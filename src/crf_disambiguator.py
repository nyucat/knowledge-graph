from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

from crf_ner import FOURTEEN_LABELS, Entity, convert_bio_to_entities, load_ner_jsonl


@dataclass
class MentionRecord:
    entity_text: str
    start: int
    end: int
    coarse_label: str


class CRFEntityDisambiguator:
    """
    词袋相似度 + CRF 全局一致性消歧：
    - 训练阶段：通过极大似然估计转移参数（标签转移概率）
    - 解码阶段：使用 Loopy Belief Propagation 做全局推断
    """

    def __init__(self) -> None:
        self.labels: List[str] = sorted(FOURTEEN_LABELS)
        self.label_to_idx: Dict[str, int] = {l: i for i, l in enumerate(self.labels)}
        self.idf: Dict[str, float] = {}
        self.label_proto: Dict[str, Dict[str, float]] = {l: {} for l in self.labels}
        self.transition_logp: List[List[float]] = []
        self.is_trained = False

    @staticmethod
    def _normalize_label(raw_label: str, entity_text: str) -> str:
        label = (raw_label or "").upper().strip()
        if "." in label:
            label = label.split(".", 1)[0]
        text = (entity_text or "").strip()
        if re.search(r"%|％|百分之", text):
            return "PERCENT"
        if re.search(r"¥|￥|\$|元|美元|欧元|英镑|日元|港元|人民币|万元|亿元", text):
            return "MONEY"
        if re.search(r"\d{1,4}年|\d{1,2}月|\d{1,2}日|\d{1,2}号|世纪|年代", text):
            return "DATE"
        if re.search(r"\d{1,2}点|\d{1,2}时|\d{1,2}分|\d{1,2}秒|上午|下午|凌晨", text):
            return "TIME"
        alias = {
            "PERSON": "PER",
            "PEOPLE": "PER",
            "HUMAN": "PER",
            "ORGANIZATION": "ORG",
            "ORGNIZATION": "ORG",
            "GPE": "LOC",
            "LOCATION": "LOC",
            "PLACE": "LOC",
            "DATETIME": "DATE",
            "MONEY_VALUE": "MONEY",
            "PCT": "PERCENT",
            "DISCIPLINE": "SUBJECT",
            "PRODUCT": "PROD",
            "UNKNOWN": "ORG",
        }
        mapped = alias.get(label, label if label else "ORG")
        return mapped if mapped in FOURTEEN_LABELS else "ORG"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        if not text:
            return tokens
        lowered = text.lower()
        for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", lowered):
            if re.fullmatch(r"[a-z0-9]+", part):
                tokens.append(part)
                continue
            if len(part) == 1:
                tokens.append(part)
            else:
                for i in range(len(part) - 1):
                    tokens.append(part[i : i + 2])
        return tokens

    @staticmethod
    def _logsumexp(vals: List[float]) -> float:
        if not vals:
            return -1e9
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, va in a.items():
            vb = b.get(k)
            if vb is not None:
                dot += va * vb
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _mention_window(self, text: str, start: int, end: int) -> str:
        s = max(0, start)
        e = min(len(text), end)
        left = text[max(0, s - 24) : s]
        mid = text[s:e]
        right = text[e : min(len(text), e + 24)]
        return f"{left} {mid} {right}".strip()

    def _train_bow_prototypes(
        self,
        samples: List[Tuple[str, List[Entity]]],
    ) -> None:
        label_docs: Dict[str, List[Counter[str]]] = {l: [] for l in self.labels}
        df: Counter[str] = Counter()
        doc_cnt = 0

        for text, entities in samples:
            for ent in entities:
                label = ent.label if ent.label in self.label_to_idx else "ORG"
                window = self._mention_window(text, ent.start, ent.end)
                tokens = self._tokenize(window)
                if not tokens:
                    continue
                tf = Counter(tokens)
                label_docs[label].append(tf)
                df.update(tf.keys())
                doc_cnt += 1

        if doc_cnt == 0:
            raise ValueError("消歧训练样本为空，无法训练词袋相似度模型。")

        self.idf = {tok: math.log((1 + doc_cnt) / (1 + c)) + 1.0 for tok, c in df.items()}

        for label in self.labels:
            docs = label_docs[label]
            if not docs:
                self.label_proto[label] = {}
                continue
            agg: Dict[str, float] = defaultdict(float)
            for tf in docs:
                for tok, cnt in tf.items():
                    agg[tok] += cnt * self.idf.get(tok, 1.0)
            scale = float(len(docs))
            self.label_proto[label] = {tok: val / scale for tok, val in agg.items()}

    def _train_transition_mle(
        self,
        samples: List[Tuple[str, List[Entity]]],
    ) -> None:
        n = len(self.labels)
        counts = [[1.0 for _ in range(n)] for _ in range(n)]  # add-one smoothing

        for _, entities in samples:
            seq = [e.label if e.label in self.label_to_idx else "ORG" for e in entities]
            if len(seq) < 2:
                continue
            for a, b in zip(seq[:-1], seq[1:]):
                counts[self.label_to_idx[a]][self.label_to_idx[b]] += 1.0

        self.transition_logp = []
        for row in counts:
            s = sum(row)
            self.transition_logp.append([math.log(v / s) for v in row])

    def _load_samples_from_ner_jsonl(self, train_path: Path) -> List[Tuple[str, List[Entity]]]:
        raw = load_ner_jsonl(train_path)
        samples: List[Tuple[str, List[Entity]]] = []
        for text, bio_tags in raw:
            entities = convert_bio_to_entities(text, bio_tags)
            entities = sorted(entities, key=lambda e: (e.start, e.end))
            samples.append((text, entities))
        return samples

    def _load_samples_from_disambiguation_jsonl(self, train_path: Path) -> List[Tuple[str, List[Entity]]]:
        samples: List[Tuple[str, List[Entity]]] = []
        if not train_path.exists():
            return samples
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get("text", ""))
                mentions = row.get("mentions", [])
                if not text or not isinstance(mentions, list):
                    continue
                entities: List[Entity] = []
                for m in mentions:
                    if not isinstance(m, dict):
                        continue
                    try:
                        start = int(m.get("start", -1))
                        end = int(m.get("end", -1))
                    except (TypeError, ValueError):
                        continue
                    if start < 0 or end <= start or end > len(text):
                        continue
                    mtext = text[start:end]
                    raw_label = str(
                        m.get("label")
                        or m.get("coarse_label")
                        or m.get("entity_parent_label")
                        or ""
                    )
                    label = self._normalize_label(raw_label, mtext)
                    entities.append(Entity(text=mtext, label=label, start=start, end=end))
                if entities:
                    entities = sorted(entities, key=lambda e: (e.start, e.end))
                    samples.append((text, entities))
        return samples

    def train_from_jsonl(
        self,
        disambiguation_train_path: Path | None,
        ner_fallback_path: Path,
    ) -> str:
        samples: List[Tuple[str, List[Entity]]] = []
        train_source = "ner_fallback"
        if disambiguation_train_path is not None:
            samples = self._load_samples_from_disambiguation_jsonl(disambiguation_train_path)
            if samples:
                train_source = "disambiguation_jsonl"
        if not samples:
            samples = self._load_samples_from_ner_jsonl(ner_fallback_path)
        self._train_bow_prototypes(samples)
        self._train_transition_mle(samples)
        self.is_trained = True
        return train_source

    def save(self, path: Path) -> None:
        if not self.is_trained:
            raise RuntimeError("消歧模型尚未训练，不能保存。")
        payload = {
            "labels": self.labels,
            "idf": self.idf,
            "label_proto": self.label_proto,
            "transition_logp": self.transition_logp,
        }
        joblib.dump(payload, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.labels = list(payload["labels"])
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.idf = dict(payload["idf"])
        self.label_proto = {k: dict(v) for k, v in payload["label_proto"].items()}
        self.transition_logp = [list(row) for row in payload["transition_logp"]]
        self.is_trained = True

    def _mention_vector(self, text: str, mention: MentionRecord) -> Dict[str, float]:
        window = self._mention_window(text, mention.start, mention.end)
        tf = Counter(self._tokenize(window))
        return {tok: cnt * self.idf.get(tok, 1.0) for tok, cnt in tf.items()}

    def _unary_log_potential(self, text: str, mention: MentionRecord) -> List[float]:
        vec = self._mention_vector(text, mention)
        scores: List[float] = []
        hint = (mention.coarse_label or "").upper().strip()
        for label in self.labels:
            sim = self._cosine(vec, self.label_proto.get(label, {}))
            bonus = 0.0
            if hint == label:
                bonus += 0.25
            if hint and label == "ORG" and hint not in self.label_to_idx:
                bonus += 0.05
            score = max(1e-8, sim + bonus + 1e-6)
            scores.append(math.log(score))
        return scores

    def _decode_lbp(self, unary: List[List[float]], mentions: List[MentionRecord]) -> List[str]:
        n = len(mentions)
        lsize = len(self.labels)
        if n == 0:
            return []
        if n == 1:
            best = max(range(lsize), key=lambda i: unary[0][i])
            return [self.labels[best]]

        neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
        edge_potential: Dict[Tuple[int, int], List[List[float]]] = {}

        # 邻接边：CRF 全局转移势
        for i in range(n - 1):
            j = i + 1
            neighbors[i].append(j)
            neighbors[j].append(i)
            forward = self.transition_logp
            backward = [[forward[b][a] for b in range(lsize)] for a in range(lsize)]
            edge_potential[(i, j)] = forward
            edge_potential[(j, i)] = backward

        # 共指边：同文本 mention 强一致性约束（引入环，使用 LBP）
        text_to_nodes: Dict[str, List[int]] = defaultdict(list)
        for i, m in enumerate(mentions):
            key = m.entity_text.strip().lower()
            if key:
                text_to_nodes[key].append(i)
        same_matrix = [
            [0.0 if a == b else -2.0 for b in range(lsize)]
            for a in range(lsize)
        ]
        for nodes in text_to_nodes.values():
            if len(nodes) <= 1:
                continue
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    i, j = nodes[a], nodes[b]
                    if j in neighbors[i]:
                        continue
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                    edge_potential[(i, j)] = same_matrix
                    edge_potential[(j, i)] = same_matrix

        messages: Dict[Tuple[int, int], List[float]] = {}
        for i, nbs in neighbors.items():
            for j in nbs:
                messages[(i, j)] = [0.0] * lsize

        max_iter = 12
        for _ in range(max_iter):
            new_messages: Dict[Tuple[int, int], List[float]] = {}
            for (i, j), old_msg in messages.items():
                incoming = [0.0] * lsize
                for k in neighbors[i]:
                    if k == j:
                        continue
                    kin = messages[(k, i)]
                    for li in range(lsize):
                        incoming[li] += kin[li]

                pair = edge_potential[(i, j)]
                out = [0.0] * lsize
                for lj in range(lsize):
                    vals = []
                    for li in range(lsize):
                        vals.append(unary[i][li] + incoming[li] + pair[li][lj])
                    out[lj] = self._logsumexp(vals)
                norm = self._logsumexp(out)
                new_messages[(i, j)] = [v - norm for v in out]
            messages = new_messages

        beliefs: List[List[float]] = []
        for i in range(n):
            b = unary[i][:]
            for k in neighbors[i]:
                kin = messages[(k, i)]
                for li in range(lsize):
                    b[li] += kin[li]
            beliefs.append(b)

        pred: List[str] = []
        for b in beliefs:
            best = max(range(lsize), key=lambda idx: b[idx])
            pred.append(self.labels[best])
        return pred

    def predict_parent_labels(
        self,
        entity_records: List[dict],
        text_by_url: Dict[str, str],
    ) -> List[str]:
        if not self.is_trained:
            return [str(row.get("entity_parent_label", "ORG")) for row in entity_records]

        grouped_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(entity_records):
            grouped_indices[str(row.get("page_url", ""))].append(idx)

        predicted: List[str] = [str(row.get("entity_parent_label", "ORG")) for row in entity_records]

        for page_url, indices in grouped_indices.items():
            doc_text = text_by_url.get(page_url, "")
            ordered = sorted(
                indices,
                key=lambda i: (
                    int(entity_records[i].get("start", -1)),
                    int(entity_records[i].get("end", -1)),
                ),
            )
            seq: List[MentionRecord] = []
            valid_indices: List[int] = []
            for i in ordered:
                row = entity_records[i]
                text = str(row.get("entity_text", "")).strip()
                if not text:
                    continue
                seq.append(
                    MentionRecord(
                        entity_text=text,
                        start=int(row.get("start", -1)),
                        end=int(row.get("end", -1)),
                        coarse_label=str(row.get("entity_parent_label", "ORG")),
                    )
                )
                valid_indices.append(i)
            if not seq:
                continue

            unary = [self._unary_log_potential(doc_text, m) for m in seq]
            seq_pred = self._decode_lbp(unary, seq)
            for i, label in zip(valid_indices, seq_pred):
                predicted[i] = label
        return predicted
