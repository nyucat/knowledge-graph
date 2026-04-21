from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Adam
from torchcrf import CRF

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAG2ID = {"O": 0, "B-TRIG": 1, "I-TRIG": 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}


@dataclass
class Sentence:
    text: str
    start: int
    end: int


@dataclass
class Mention:
    canonical_name: str
    entity_type: str
    mention: str
    start: int
    end: int


@dataclass
class Rule:
    name: str
    relation: str
    triggers: list[str]
    head_canonical_in: list[str]
    head_type_in: list[str]
    tail_canonical_in: list[str]
    tail_type_in: list[str]
    pairing: str
    max_pairs: int
    head_before_tail: bool
    trigger_between: bool
    tail_side: str


@dataclass
class RegexTriple:
    head: str
    relation: str
    tail: str
    rule: str
    trigger: str
    evidence_sentence: str


class LSTMCRFTriggerTagger(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hid_dim: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim, len(TAG2ID))
        self.crf = CRF(len(TAG2ID), batch_first=True)

    def emissions(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.lstm(h)
        return self.fc(h)

    def loss(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return -self.crf(self.emissions(x), y, mask=mask, reduction="mean")

    def decode(self, x: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        return self.crf.decode(self.emissions(x), mask=mask)


def split_sentences(text: str) -> list[Sentence]:
    out: list[Sentence] = []
    s = 0
    for m in re.finditer(r"[。！？!?\n]", text):
        e = m.end()
        seg = text[s:e].strip()
        if seg:
            rs = text.find(seg, s, e)
            rs = rs if rs >= 0 else s
            out.append(Sentence(seg, rs, rs + len(seg)))
        s = e
    if s < len(text):
        seg = text[s:].strip()
        if seg:
            rs = text.find(seg, s)
            rs = rs if rs >= 0 else s
            out.append(Sentence(seg, rs, rs + len(seg)))
    return out


def load_mentions(path: Path) -> list[Mention]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    out: list[Mention] = []
    for row in payload.get("mentions_in_text_order", []):
        out.append(
            Mention(
                canonical_name=str(row.get("canonical_name", "")).strip(),
                entity_type=str(row.get("entity_type", "")).strip(),
                mention=str(row.get("mention", "")).strip(),
                start=int(row.get("start", -1)),
                end=int(row.get("end", -1)),
            )
        )
    return [m for m in out if m.canonical_name and m.start >= 0 and m.end > m.start]


def load_rules(path: Path) -> tuple[dict, list[Rule]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    settings = raw.get("settings", {})
    rules: list[Rule] = []
    for r in raw.get("rules", []):
        h = r.get("head", {}) or {}
        t = r.get("tail", {}) or {}
        rules.append(
            Rule(
                name=str(r.get("name", "")),
                relation=str(r.get("relation", "")),
                triggers=[str(x) for x in r.get("triggers", []) if str(x)],
                head_canonical_in=[str(x) for x in h.get("canonical_in", []) if str(x)],
                head_type_in=[str(x) for x in h.get("type_in", []) if str(x)],
                tail_canonical_in=[str(x) for x in t.get("canonical_in", []) if str(x)],
                tail_type_in=[str(x) for x in t.get("type_in", []) if str(x)],
                pairing=str(r.get("pairing", "nearest")),
                max_pairs=max(1, int(r.get("max_pairs", 1))),
                head_before_tail=bool(r.get("head_before_tail", False)),
                trigger_between=bool(r.get("trigger_between", False)),
                tail_side=str(r.get("tail_side", "any")),
            )
        )
    return settings, rules


def extract_literal_mentions(text: str) -> list[Mention]:
    rules = [
        (r"\d{4}年\d{1,2}月\d{1,2}日", "Date"),
        (r"\d{4}年\d{1,2}月", "Date"),
        (r"\d{4}年", "Date"),
        (r"\d+岁", "Time"),
        (r"\d+小时\d+分\d+秒", "Time"),
        (r"\d+分钟", "Time"),
        (r"第[一二三四五六七八九十百千万\d]+名", "Time"),
        (r"[约近超过]\d{1,3}(,\d{3})*人", "Time"),
        (r"\d{1,3}(,\d{3})*人", "Time"),
    ]
    out: list[Mention] = []
    seen = set()
    for pattern, entity_type in rules:
        for m in re.finditer(pattern, text):
            mention = m.group(0).strip("，。；、 ")
            if not mention:
                continue
            key = (mention, m.start(), m.end(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Mention(
                    canonical_name=mention,
                    entity_type=entity_type,
                    mention=mention,
                    start=m.start(),
                    end=m.end(),
                )
            )
    out.sort(key=lambda x: (x.start, x.end))
    return out


def normalize_name(value: str, settings: dict) -> str:
    s = value.strip()
    raw = s
    s = re.sub(r'[)\]）】」》”"，。；、]+$', '', s)
    mapping = settings.get("display_name_map", {}) or {}
    if raw in mapping:
        return str(mapping[raw]).strip()
    if s in mapping:
        return str(mapping[s]).strip()
    if " (" in s:
        s = s.split(" (", 1)[0].strip()
    if s.startswith("英国") and ("柴郡威姆斯洛" in s or "伦敦麦达维尔" in s):
        s = s[2:]
    if s.endswith("附近"):
        s = s[:-2]
    if s == "国际象棋程序":
        s = "图灵国际象棋程序"
    if s == "首个电脑国际象棋程序":
        s = "洛斯阿拉莫斯象棋"
    return s


def sentence_by_offset(text: str, offset: int) -> str:
    s = text.rfind("。", 0, offset)
    q = text.rfind("？", 0, offset)
    e = text.rfind("！", 0, offset)
    start = max(s, q, e) + 1
    end_candidates = [x for x in [text.find("。", offset), text.find("？", offset), text.find("！", offset)] if x >= 0]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()


def build_vocab(text: str) -> dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch in text:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(s: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(c, 1) for c in s]


def weak_label_trigger(sent: str, keywords: Iterable[str]) -> list[int]:
    y = [TAG2ID["O"]] * len(sent)
    for kw in sorted(set(keywords), key=len, reverse=True):
        for m in re.finditer(re.escape(kw), sent):
            s, e = m.start(), m.end()
            if s >= e:
                continue
            y[s] = TAG2ID["B-TRIG"]
            for i in range(s + 1, e):
                y[i] = TAG2ID["I-TRIG"]
    return y


def train_trigger_model(sentences: list[Sentence], keywords: list[str], vocab: dict[str, int], epochs: int) -> LSTMCRFTriggerTagger:
    model = LSTMCRFTriggerTagger(vocab_size=len(vocab))
    samples: list[tuple[list[int], list[int]]] = []
    for s in sentences:
        if len(s.text) < 2:
            continue
        y = weak_label_trigger(s.text, keywords)
        if any(t != TAG2ID["O"] for t in y):
            samples.append((encode(s.text, vocab), y))

    if not samples:
        return model

    model.train()
    opt = Adam(model.parameters(), lr=1e-3)
    for _ in range(max(1, epochs)):
        for x, y in samples:
            tx = torch.tensor([x], dtype=torch.long)
            ty = torch.tensor([y], dtype=torch.long)
            mask = torch.ones_like(tx, dtype=torch.bool)
            opt.zero_grad()
            loss = model.loss(tx, ty, mask)
            loss.backward()
            opt.step()
    return model


def decode_trigger_spans(model: LSTMCRFTriggerTagger, sent: str, vocab: dict[str, int]) -> list[tuple[int, int, str]]:
    if not sent:
        return []
    model.eval()
    tx = torch.tensor([encode(sent, vocab)], dtype=torch.long)
    mask = torch.ones_like(tx, dtype=torch.bool)
    with torch.no_grad():
        tags = model.decode(tx, mask)[0]

    spans: list[tuple[int, int, str]] = []
    st = -1
    for i, t in enumerate(tags):
        lab = ID2TAG.get(t, "O")
        if lab == "B-TRIG":
            if st != -1:
                spans.append((st, i, sent[st:i]))
            st = i
        elif lab != "I-TRIG":
            if st != -1:
                spans.append((st, i, sent[st:i]))
                st = -1
    if st != -1:
        spans.append((st, len(sent), sent[st:]))
    return spans


def sentence_mentions(sent: Sentence, mentions: list[Mention]) -> list[Mention]:
    return [m for m in mentions if sent.start <= m.start and m.end <= sent.end]


def inject_main_entity_mentions(sent: Sentence, ms: list[Mention], settings: dict, rule: Rule) -> list[Mention]:
    main_entity = str(settings.get("main_entity", "")).strip()
    if not main_entity:
        return ms
    if rule.head_canonical_in and main_entity not in rule.head_canonical_in:
        return ms
    if any(m.canonical_name == main_entity for m in ms):
        return ms
    aliases = [str(x).strip() for x in settings.get("main_aliases", []) if str(x).strip()]
    if not aliases:
        return ms
    for alias in aliases:
        idx = sent.text.find(alias)
        if idx >= 0:
            start = sent.start + idx
            end = start + len(alias)
            ms = ms + [Mention(canonical_name=main_entity, entity_type="Person", mention=alias, start=start, end=end)]
            break
    return ms


def type_match(entity_type: str, accepted: list[str]) -> bool:
    if not accepted:
        return True
    if entity_type in accepted:
        return True
    if "Location" in accepted and entity_type.startswith("Location"):
        return True
    if "Date" in accepted and entity_type in {"Date", "Time"}:
        return True
    if "Time" in accepted and entity_type in {"Date", "Time"}:
        return True
    return False


def mention_match(m: Mention, canonical_in: list[str], type_in: list[str]) -> bool:
    if canonical_in and m.canonical_name not in canonical_in:
        return False
    if not type_match(m.entity_type, type_in):
        return False
    return True


def pair_candidates(ms: list[Mention], trig_pos: int, rule: Rule) -> list[tuple[Mention, Mention]]:
    heads = [m for m in ms if mention_match(m, rule.head_canonical_in, rule.head_type_in)]
    tails = [m for m in ms if mention_match(m, rule.tail_canonical_in, rule.tail_type_in)]
    pairs: list[tuple[Mention, Mention]] = []
    for h in heads:
        for t in tails:
            if h.canonical_name == t.canonical_name:
                continue
            pairs.append((h, t))

    if not pairs:
        return []

    def score(p: tuple[Mention, Mention]) -> int:
        h, t = p
        hs = (h.start + h.end) // 2
        ts = (t.start + t.end) // 2
        base = abs(hs - trig_pos) + abs(ts - trig_pos)
        if rule.head_before_tail and h.start > t.start:
            base += 1000
        if rule.trigger_between and not (h.end <= trig_pos <= t.start):
            base += 1000
        if rule.tail_side == "before" and t.end > trig_pos:
            base += 1000
        if rule.tail_side == "after" and t.start < trig_pos:
            base += 1000
        return base

    if rule.pairing == "nearest":
        return [min(pairs, key=score)]
    pairs.sort(key=score)
    return pairs[: rule.max_pairs]


def extract_relations(text: str, mentions: list[Mention], rules: list[Rule], settings: dict, epochs: int) -> dict:
    sents = split_sentences(text)
    trigger_keywords = [kw for r in rules for kw in r.triggers]

    vocab = build_vocab(text)
    model = train_trigger_model(sents, trigger_keywords, vocab, epochs=epochs)

    relation_mentions = []
    triples = []
    seen_triples = set()

    for sent in sents:
        trig_spans = decode_trigger_spans(model, sent.text, vocab)
        for ts, te, tv in trig_spans:
            relation_mentions.append(
                {
                    "relation": "触发词",
                    "trigger": tv,
                    "sentence": sent.text,
                    "start": sent.start + ts,
                    "end": sent.start + te,
                }
            )

        for rule in rules:
            ms = inject_main_entity_mentions(sent, sentence_mentions(sent, mentions), settings, rule)
            if len(ms) < 2:
                continue
            for kw in rule.triggers:
                for m in re.finditer(re.escape(kw), sent.text):
                    trig_pos = sent.start + m.start()
                    relation_mentions.append(
                        {
                            "relation": rule.relation,
                            "trigger": kw,
                            "sentence": sent.text,
                            "start": trig_pos,
                            "end": trig_pos + len(kw),
                        }
                    )
                    for h, t in pair_candidates(ms, trig_pos, rule):
                        head_name = normalize_name(h.canonical_name, settings)
                        tail_name = normalize_name(t.canonical_name, settings)
                        key = (head_name, rule.relation, tail_name)
                        if key in seen_triples:
                            continue
                        seen_triples.add(key)
                        triples.append(
                            {
                                "head": head_name,
                                "relation": rule.relation,
                                "tail": tail_name,
                                "head_mention": h.mention,
                                "tail_mention": t.mention,
                                "trigger": kw,
                                "rule": rule.name,
                                "evidence_sentence": sent.text,
                            }
                        )

    regex_triples = extract_regex_triples(text, settings)
    for rt in regex_triples:
        key = (rt.head, rt.relation, rt.tail)
        if key in seen_triples:
            continue
        seen_triples.add(key)
        triples.append(
            {
                "head": rt.head,
                "relation": rt.relation,
                "tail": rt.tail,
                "head_mention": rt.head,
                "tail_mention": rt.tail,
                "trigger": rt.trigger,
                "rule": rt.rule,
                "evidence_sentence": rt.evidence_sentence,
            }
        )
        relation_mentions.append(
            {
                "relation": rt.relation,
                "trigger": rt.trigger,
                "sentence": rt.evidence_sentence,
                "start": text.find(rt.evidence_sentence),
                "end": text.find(rt.evidence_sentence) + len(rt.evidence_sentence),
            }
        )

    # dedupe mentions
    seen_rel = set()
    rel_unique = []
    for r in relation_mentions:
        k = (r["relation"], r["trigger"], r["start"], r["end"])
        if k in seen_rel:
            continue
        seen_rel.add(k)
        rel_unique.append(r)

    relation_types = sorted({t["relation"] for t in triples})
    return {
        "relations_only": {
            "relation_type_count": len(relation_types),
            "relation_types": relation_types,
            "relation_mention_count": len(rel_unique),
            "relation_mentions": rel_unique,
        },
        "triples": {
            "triple_count": len(triples),
            "triples": triples,
        },
    }


def extract_regex_triples(text: str, settings: dict) -> list[RegexTriple]:
    rules = settings.get("regex_rules", []) or []
    out: list[RegexTriple] = []
    main_entity = normalize_name(str(settings.get("main_entity", "")).strip(), settings)
    for rule in rules:
        name = str(rule.get("name", "")).strip()
        relation = str(rule.get("relation", "")).strip()
        pattern = str(rule.get("pattern", "")).strip()
        if not (name and relation and pattern):
            continue
        split_pattern = str(rule.get("split_pattern", "")).strip()
        head_mode = str((rule.get("head", {}) or {}).get("from", "main_entity")).strip()
        head_group = str((rule.get("head", {}) or {}).get("group", "")).strip()
        tail_groups = [str(x).strip() for x in rule.get("tail_groups", []) if str(x).strip()]
        tail_group = str(rule.get("tail_group", "")).strip()
        for m in re.finditer(pattern, text):
            if head_mode == "group" and head_group:
                head = normalize_name(m.group(head_group), settings)
            else:
                head = main_entity
            if not head:
                continue
            candidates: list[tuple[str, str]] = []
            if tail_group:
                candidates.append((relation, m.group(tail_group)))
            for tg in tail_groups:
                if "|" in tg:
                    grp, rel = tg.split("|", 1)
                    candidates.append((rel.strip(), m.group(grp.strip())))
                else:
                    candidates.append((relation, m.group(tg)))
            if not candidates:
                continue
            for rel_name, raw_tail in candidates:
                parts = [raw_tail]
                if split_pattern:
                    parts = [p for p in re.split(split_pattern, raw_tail) if p]
                for p in parts:
                    tail = normalize_name(p.strip(" ，。；、“”\""), settings)
                    if not tail or tail == head:
                        continue
                    out.append(
                        RegexTriple(
                            head=head,
                            relation=rel_name,
                            tail=tail,
                            rule=name,
                            trigger=relation,
                            evidence_sentence=sentence_by_offset(text, m.start()),
                        )
                    )
    return out


def main() -> None:
    text = (PROJECT_ROOT / "datafile" / "test.txt").read_text(encoding="utf-8", errors="ignore")
    mentions = load_mentions(PROJECT_ROOT / "outputs" / "turing_entities_disambiguated.json")
    mentions.extend(extract_literal_mentions(text))

    settings, rules = load_rules(PROJECT_ROOT / "data" / "relation_rules_turing.json")
    epochs = int(settings.get("trigger_epochs", 3))

    outputs = extract_relations(text=text, mentions=mentions, rules=rules, settings=settings, epochs=epochs)

    rel_path = PROJECT_ROOT / "outputs" / "turing_relations_only.json"
    tri_path = PROJECT_ROOT / "outputs" / "turing_kg_triples.json"

    rel_path.write_text(json.dumps(outputs["relations_only"], ensure_ascii=False, indent=2), encoding="utf-8")
    tri_path.write_text(json.dumps(outputs["triples"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "relations_output": str(rel_path),
                "triples_output": str(tri_path),
                "rule_count": len(rules),
                "relation_mention_count": outputs["relations_only"]["relation_mention_count"],
                "triple_count": outputs["triples"]["triple_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
