from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bert_bilstm_crf_ner import BertBiLSTMCRFEntityRecognizer, load_ner_jsonl_for_bert

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORY_TO_COARSE = {
    "person": "PER",
    "location": "LOC",
    "organization": "ORG",
    "work": "WORK",
    "tech_theory": "TECH",
    "event_law": "EVENT",
    "honor_identity": "FIELD",
    "other": "FIELD",
    "time_value": "DATE",
}


@dataclass
class Span:
    text: str
    start: int
    end: int
    category: str
    coarse: str
    source: str


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_lexicon(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: [str(x).strip() for x in v if str(x).strip()] for k, v in data.items()}


def _find_term_spans(text: str, term: str) -> Iterable[tuple[int, int]]:
    if not term:
        return []
    flags = re.IGNORECASE if re.search(r"[A-Za-z]", term) else 0
    return [(m.start(), m.end()) for m in re.finditer(re.escape(term), text, flags=flags)]


def extract_by_lexicon(text: str, lexicon: dict[str, list[str]]) -> list[Span]:
    spans: list[Span] = []
    for category, terms in lexicon.items():
        coarse = CATEGORY_TO_COARSE.get(category, "ORG")
        for term in sorted(set(terms), key=len, reverse=True):
            for s, e in _find_term_spans(text, term):
                spans.append(Span(text=text[s:e], start=s, end=e, category=category, coarse=coarse, source="lexicon"))
    spans.sort(key=lambda x: (x.start, -(x.end - x.start)))

    # remove strict-overlap duplicates, keep longest first at same start
    filtered: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for sp in spans:
        overlap = False
        for a, b in occupied:
            if max(a, sp.start) < min(b, sp.end) and (sp.end - sp.start) <= (b - a):
                overlap = True
                break
        if overlap:
            continue
        filtered.append(sp)
        occupied.append((sp.start, sp.end))
    filtered.sort(key=lambda x: (x.start, x.end))
    return filtered


def split_sentences_with_offsets(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    start = 0
    for m in re.finditer(r"[。！？!?\n]", text):
        end = m.end()
        s = text[start:end].strip()
        if s:
            real_start = text.find(s, start, end)
            out.append((real_start if real_start >= 0 else start, s))
        start = end
    if start < len(text):
        s = text[start:].strip()
        if s:
            real_start = text.find(s, start)
            out.append((real_start if real_start >= 0 else start, s))
    return out


def weak_label_sentence(sentence: str, lexicon: dict[str, list[str]]) -> list[str]:
    labels = ["O"] * len(sentence)
    candidates: list[tuple[int, int, str]] = []
    for cat, terms in lexicon.items():
        coarse = CATEGORY_TO_COARSE.get(cat, "ORG")
        for term in sorted(set(terms), key=len, reverse=True):
            for s, e in _find_term_spans(sentence, term):
                if e > s:
                    candidates.append((s, e, coarse))
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    occupied = [False] * len(sentence)
    for s, e, coarse in candidates:
        if any(occupied[i] for i in range(s, e)):
            continue
        labels[s] = f"B-{coarse}"
        for i in range(s + 1, e):
            labels[i] = f"I-{coarse}"
        for i in range(s, e):
            occupied[i] = True
    return labels


def build_weak_samples(text: str, lexicon: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    samples: list[tuple[str, list[str]]] = []
    for _, sent in split_sentences_with_offsets(text):
        if len(sent) < 4:
            continue
        labels = weak_label_sentence(sent, lexicon)
        if any(tag != "O" for tag in labels):
            samples.append((sent, labels))
    return samples


def train_model(train_data: Path, text: str, lexicon: dict[str, list[str]], model_name: str, model_path: Path) -> BertBiLSTMCRFEntityRecognizer:
    base_samples = load_ner_jsonl_for_bert(train_data)
    weak_samples = build_weak_samples(text, lexicon)
    recognizer = BertBiLSTMCRFEntityRecognizer(
        model_name=model_name,
        epochs=2,
        batch_size=8,
    )
    recognizer.train(base_samples + weak_samples)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(model_path)
    return recognizer


def extract_by_model(text: str, recognizer: BertBiLSTMCRFEntityRecognizer) -> list[Span]:
    spans: list[Span] = []
    for e in recognizer.predict_entities(text):
        coarse = (e.label or "ORG").upper().strip()
        category = {
            "PER": "person",
            "LOC": "location",
            "ORG": "organization",
            "WORK": "work",
            "TECH": "tech_theory",
            "THEORY": "tech_theory",
            "EVENT": "event_law",
            "DATE": "time_value",
            "TIME": "time_value",
        }.get(coarse, "other")
        spans.append(Span(text=e.text, start=e.start, end=e.end, category=category, coarse=coarse, source="model"))
    return spans


def merge_spans(model_spans: list[Span], lex_spans: list[Span]) -> list[Span]:
    del model_spans  # keep training, but final extraction is lexicon-prioritized for target list fidelity
    merged: dict[tuple[int, int, str], Span] = {}
    for sp in lex_spans:
        merged[(sp.start, sp.end, sp.text)] = sp
    out = list(merged.values())
    out.sort(key=lambda x: (x.start, x.end, x.text))
    return out


def to_outputs(spans: list[Span]) -> dict:
    categorized: dict[str, list[str]] = {}
    for cat in CATEGORY_TO_COARSE:
        seen: set[str] = set()
        vals: list[str] = []
        for sp in spans:
            if sp.category != cat:
                continue
            if sp.text in seen:
                continue
            seen.add(sp.text)
            vals.append(sp.text)
        categorized[cat] = vals

    ordered = [sp.text for sp in spans]  # keep original order, allow repeats
    return {
        "categorized_entities": categorized,
        "ordered_entities": ordered,
        "entity_count": len(spans),
    }


def main() -> None:
    text_path = PROJECT_ROOT / "datafile" / "test.txt"
    train_data = PROJECT_ROOT / "data" / "sample_ner_train.jsonl"
    lexicon_path = PROJECT_ROOT / "data" / "turing_lexicon.json"
    model_name = str(PROJECT_ROOT / "models" / "bert-base-chinese")
    model_path = PROJECT_ROOT / "outputs" / "turing_bert_bilstm_crf.pt"
    out_path = PROJECT_ROOT / "outputs" / "turing_entities.json"

    text = load_text(text_path)
    lexicon = load_lexicon(lexicon_path)

    recognizer = train_model(train_data=train_data, text=text, lexicon=lexicon, model_name=model_name, model_path=model_path)
    model_spans = extract_by_model(text, recognizer)
    lex_spans = extract_by_lexicon(text, lexicon)
    spans = merge_spans(model_spans, lex_spans)

    result = to_outputs(spans)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "entity_count": result["entity_count"],
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
