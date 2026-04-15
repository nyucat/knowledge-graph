from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import sklearn_crfsuite

FOURTEEN_LABELS = {
    "EVENT",
    "FIELD",
    "LOC",
    "ORG",
    "PER",
    "PROD",
    "TECH",
    "THEORY",
    "TIME",
    "WORK",
    "MONEY",
    "PERCENT",
    "DATE",
    "SUBJECT",
}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


def sentence_to_features(chars: Sequence[str], idx: int) -> Dict[str, object]:
    ch = chars[idx]
    features: Dict[str, object] = {
        "bias": 1.0,
        "ch": ch,
        "is_digit": ch.isdigit(),
        "is_alpha": ch.isalpha(),
        "is_upper": ch.isupper(),
        "is_chinese": bool(re.match(r"[\u4e00-\u9fff]", ch)),
    }
    if idx > 0:
        features["-1:bigram"] = chars[idx - 1] + ch
    if idx < len(chars) - 1:
        features["+1:bigram"] = ch + chars[idx + 1]
    if idx > 0:
        prev = chars[idx - 1]
        features.update(
            {
                "-1:ch": prev,
                "-1:is_digit": prev.isdigit(),
                "-1:is_alpha": prev.isalpha(),
            }
        )
    else:
        features["BOS"] = True
    if idx > 1:
        prev2 = chars[idx - 2]
        features.update(
            {
                "-2:ch": prev2,
                "-2:is_digit": prev2.isdigit(),
                "-2:is_alpha": prev2.isalpha(),
            }
        )
    if idx < len(chars) - 1:
        nxt = chars[idx + 1]
        features.update(
            {
                "+1:ch": nxt,
                "+1:is_digit": nxt.isdigit(),
                "+1:is_alpha": nxt.isalpha(),
            }
        )
    else:
        features["EOS"] = True
    if idx < len(chars) - 2:
        nxt2 = chars[idx + 2]
        features.update(
            {
                "+2:ch": nxt2,
                "+2:is_digit": nxt2.isdigit(),
                "+2:is_alpha": nxt2.isalpha(),
            }
        )
    return features


def convert_bio_to_entities(text: str, labels: Sequence[str]) -> List[Entity]:
    entities: List[Entity] = []
    start = -1
    ent_label = ""
    max_span_len = 20

    def close_entity(end_idx: int) -> None:
        nonlocal start, ent_label
        if start == -1:
            return
        if end_idx - start <= max_span_len:
            entities.append(Entity(text=text[start:end_idx], label=ent_label, start=start, end=end_idx))
        start = -1
        ent_label = ""

    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            close_entity(i)
            start = i
            ent_label = tag[2:]
        elif tag.startswith("I-"):
            current = tag[2:]
            # 若 I- 标签前面没有对应实体，则按 B- 处理，修正非法 BIO 序列。
            if start == -1:
                start = i
                ent_label = current
            elif ent_label != current:
                close_entity(i)
                start = i
                ent_label = current
            continue
        else:
            close_entity(i)
    close_entity(len(text))
    entities = repair_numeric_boundaries(text, entities)
    entities = repair_common_prefix_boundaries(text, entities)
    return entities


def repair_numeric_boundaries(text: str, entities: List[Entity]) -> List[Entity]:

    repaired: List[Entity] = []
    for ent in entities:
        if (
            ent.start > 0
            and re.match(r"^\d{3}(年|月|日|届)", ent.text)
            and text[ent.start - 1].isdigit()
        ):
            ent.start -= 1
            ent.text = text[ent.start : ent.end]
        repaired.append(ent)
    return repaired


def repair_common_prefix_boundaries(text: str, entities: List[Entity]) -> List[Entity]:
    repaired: List[Entity] = []
    suffixes = {"同校", "制度", "学制", "年制", "分制", "委员会", "出版社"}
    for ent in entities:
        if ent.start > 0 and len(ent.text) >= 2:
            prev = text[ent.start - 1]
            if (
                re.match(r"[\u4e00-\u9fff]", prev)
                and not re.match(r"[，。！？；：、\s]", prev)
                and any(ent.text.endswith(s) for s in suffixes)
            ):
                candidate = prev + ent.text
                if 2 <= len(candidate) <= 12:
                    ent.start -= 1
                    ent.text = candidate
        repaired.append(ent)
    return repaired


def merge_adjacent_entities(text: str, entities: List[Entity]) -> List[Entity]:
    if not entities:
        return entities
    entities = sorted(entities, key=lambda e: (e.start, e.end))
    merged: List[Entity] = []
    current = entities[0]

    merge_suffixes = {
        "委员会",
        "出版社",
        "研究院",
        "研究所",
        "实验室",
        "大学",
        "学院",
        "集团",
        "公司",
        "协会",
        "中心",
        "银行",
        "电视台",
        "几等奖",
        "一等奖",
        "二等奖",
        "三等奖",
        "特等奖",
    }
    mergeable_labels = {"ORG", "FIELD", "TECH", "SUBJECT", "WORK", "PROD", "THEORY", "EVENT"}

    def should_merge(a: Entity, b: Entity) -> bool:
        gap_text = text[a.end : b.start]
        if len(gap_text) > 1:
            return False
        if gap_text and gap_text not in {" ", "-", "/", "—"}:
            return False
        if a.label != b.label:
            # 仅在知识类标签内允许跨标签合并，避免误拼人名地名。
            if not (a.label in mergeable_labels and b.label in mergeable_labels):
                return False
        candidate = text[a.start : b.end]
        if any(candidate.endswith(s) for s in merge_suffixes):
            return True
        if len(a.text) <= 3 and len(b.text) <= 3 and len(candidate) <= 10:
            return True
        if re.search(r"第?[一二三四五六七八九十0-9]+等奖$", candidate):
            return True
        return False

    for nxt in entities[1:]:
        if should_merge(current, nxt):
            current = Entity(
                text=text[current.start : nxt.end],
                label=current.label,
                start=current.start,
                end=nxt.end,
            )
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


class CRFEntityRecognizer:
    def __init__(self) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        self.is_trained = False

    def train(self, samples: Iterable[Tuple[str, List[str]]]) -> None:
        x_train = []
        y_train = []
        for text, labels in samples:
            chars = list(text)
            x_train.append([sentence_to_features(chars, i) for i in range(len(chars))])
            y_train.append(labels)
        if not x_train:
            raise ValueError("训练样本为空，无法训练 CRF。")
        self.model.fit(x_train, y_train)
        self.is_trained = True

    def predict_labels(self, text: str) -> List[str]:
        if not self.is_trained:
            raise RuntimeError("CRF 模型尚未训练。")
        if not text:
            return []
        chars = list(text)
        x = [[sentence_to_features(chars, i) for i in range(len(chars))]]
        return self.model.predict(x)[0]

    def predict_entities(self, text: str) -> List[Entity]:
        labels = self.predict_labels(text)
        entities = convert_bio_to_entities(text, labels)
        entities = merge_adjacent_entities(text, entities)
        return entities

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，不能保存。")
        joblib.dump(self.model, Path(path))

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(Path(path))
        self.is_trained = True


def load_ner_jsonl(path: str | Path) -> List[Tuple[str, List[str]]]:
    label_alias = {
        "PERSON": "PER",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOC",
        "GPE": "LOC",
        "DATETIME": "DATE",
        "DISCIPLINE": "SUBJECT",
        "PRODUCT": "PROD",
    }
    samples: List[Tuple[str, List[str]]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            raw_labels = row["labels"]
            labels: List[str] = []
            for tag in raw_labels:
                if tag == "O":
                    labels.append(tag)
                    continue
                if "-" not in tag:
                    labels.append("O")
                    continue
                bio, entity_type = tag.split("-", 1)
                mapped = label_alias.get(entity_type, entity_type)
                if mapped not in FOURTEEN_LABELS:
                    mapped = "ORG"
                labels.append(f"{bio}-{mapped}")
            if len(text) != len(labels):
                raise ValueError("每条样本的 text 与 labels 长度必须一致。")
            samples.append((text, labels))
    return samples


def ensure_label_coverage(
    samples: List[Tuple[str, List[str]]],
    expected_labels: set[str] | None = None,
) -> None:
    expected = expected_labels or FOURTEEN_LABELS
    present: set[str] = set()
    for _, labels in samples:
        for tag in labels:
            if tag != "O" and "-" in tag:
                present.add(tag.split("-", 1)[1])
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"训练集中缺少以下标签: {', '.join(missing)}")
