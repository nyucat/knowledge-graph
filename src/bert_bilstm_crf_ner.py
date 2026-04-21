from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

from crf_ner import Entity, FOURTEEN_LABELS, convert_bio_to_entities

BIO_ENTITY_TYPES = sorted(FOURTEEN_LABELS)
BIO_TAGS = ["O"] + [f"{prefix}-{label}" for label in BIO_ENTITY_TYPES for prefix in ("B", "I")]


def _to_i_tag(tag: str) -> str:
    if tag.startswith("B-"):
        return "I-" + tag[2:]
    return tag


def _build_tag_mappings() -> tuple[dict[str, int], dict[int, str]]:
    tag2id = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag


class CharTokenClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Tuple[str, List[str]]],
        tokenizer,
        tag2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        for text, labels in samples:
            if not text:
                continue
            if len(text) != len(labels):
                continue
            self.items.append(
                self._encode_sample(
                    text=text,
                    labels=labels,
                    tokenizer=tokenizer,
                    tag2id=tag2id,
                    max_length=max_length,
                )
            )

    @staticmethod
    def _encode_sample(
        text: str,
        labels: Sequence[str],
        tokenizer,
        tag2id: dict[str, int],
        max_length: int,
    ) -> dict[str, list[int]]:
        chars = list(text)
        encoded = tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

        word_ids = encoded.word_ids()
        prev_word_id = None
        tag_ids: list[int] = []
        label_mask: list[int] = []

        for wid in word_ids:
            if wid is None or wid >= len(labels):
                tag_ids.append(tag2id["O"])
                label_mask.append(0)
                prev_word_id = wid
                continue

            raw_tag = labels[wid]
            if wid == prev_word_id:
                raw_tag = _to_i_tag(raw_tag)
            tag_ids.append(tag2id.get(raw_tag, tag2id["O"]))
            label_mask.append(1)
            prev_word_id = wid

        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "tag_ids": tag_ids,
            "label_mask": label_mask,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tags: torch.Tensor
    crf_mask: torch.Tensor


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> _Batch:
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(seq: list[int], value: int) -> list[int]:
        return seq + [value] * (max_len - len(seq))

    input_ids = torch.tensor([pad(item["input_ids"], pad_id) for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long)
    tags = torch.tensor([pad(item["tag_ids"], 0) for item in batch], dtype=torch.long)
    crf_mask = attention_mask > 0
    return _Batch(input_ids=input_ids, attention_mask=attention_mask, tags=tags, crf_mask=crf_mask)


class BertBiLSTMCRFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_tags: int,
        lstm_hidden_size: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.bert.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        logits = self.classifier(self.dropout(lstm_out))
        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: torch.Tensor,
        crf_mask: torch.Tensor,
    ) -> torch.Tensor:
        emissions = self.emissions(input_ids=input_ids, attention_mask=attention_mask)
        log_likelihood = self.crf(emissions, tags, mask=crf_mask, reduction="mean")
        return -log_likelihood

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, crf_mask: torch.Tensor) -> list[list[int]]:
        emissions = self.emissions(input_ids=input_ids, attention_mask=attention_mask)
        return self.crf.decode(emissions, mask=crf_mask)


class BertBiLSTMCRFEntityRecognizer:
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 256,
        lr: float = 3e-5,
        epochs: int = 3,
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tag2id, self.id2tag = _build_tag_mappings()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertBiLSTMCRFModel(model_name=model_name, num_tags=len(self.tag2id)).to(self.device)
        self.is_trained = False

    def train(self, samples: Iterable[Tuple[str, List[str]]]) -> None:
        dataset = CharTokenClassificationDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            tag2id=self.tag2id,
            max_length=self.max_length,
        )
        if len(dataset) == 0:
            raise ValueError("BERT-BiLSTM-CRF 训练样本为空。")

        pad_id = self.tokenizer.pad_token_id or 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: _collate(x, pad_id),
        )

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()

        for _ in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                loss = self.model.compute_loss(
                    input_ids=batch.input_ids.to(self.device),
                    attention_mask=batch.attention_mask.to(self.device),
                    tags=batch.tags.to(self.device),
                    crf_mask=batch.crf_mask.to(self.device),
                )
                loss.backward()
                optimizer.step()

        self.is_trained = True

    def _predict_chunk_labels(self, text: str) -> list[str]:
        if not text:
            return []
        chars = list(text)
        encoded = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        crf_mask = attention_mask > 0

        self.model.eval()
        with torch.no_grad():
            pred = self.model.decode(input_ids=input_ids, attention_mask=attention_mask, crf_mask=crf_mask)[0]

        char_labels = ["O"] * len(chars)
        token_idx = 0
        seen_words: set[int] = set()
        for pos, wid in enumerate(word_ids):
            if int(attention_mask[0, pos].item()) == 0:
                continue
            if token_idx >= len(pred):
                break
            tag_id = pred[token_idx]
            token_idx += 1
            if wid is None:
                continue
            if wid in seen_words:
                continue
            char_labels[wid] = self.id2tag.get(tag_id, "O")
            seen_words.add(wid)
        return char_labels

    def predict_labels(self, text: str, infer_chunk_chars: int = 220) -> List[str]:
        if not self.is_trained:
            raise RuntimeError("BERT-BiLSTM-CRF 模型尚未训练。")
        if not text:
            return []

        labels: list[str] = []
        cursor = 0
        while cursor < len(text):
            chunk = text[cursor : cursor + infer_chunk_chars]
            chunk_labels = self._predict_chunk_labels(chunk)
            labels.extend(chunk_labels)
            cursor += len(chunk)
        return labels

    def predict_entities(self, text: str) -> List[Entity]:
        labels = self.predict_labels(text)
        return convert_bio_to_entities(text, labels)

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法保存。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "tag2id": self.tag2id,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.model_name = str(payload.get("model_name", self.model_name))
        self.max_length = int(payload.get("max_length", self.max_length))
        self.lr = float(payload.get("lr", self.lr))
        self.epochs = int(payload.get("epochs", self.epochs))
        self.batch_size = int(payload.get("batch_size", self.batch_size))

        self.tag2id = {str(k): int(v) for k, v in payload["tag2id"].items()}
        self.id2tag = {int(v): str(k) for k, v in self.tag2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertBiLSTMCRFModel(model_name=self.model_name, num_tags=len(self.tag2id)).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.is_trained = True


def load_ner_jsonl_for_bert(path: str | Path) -> list[tuple[str, list[str]]]:
    samples: list[tuple[str, list[str]]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = str(row.get("text", ""))
            labels = [str(x) for x in row.get("labels", [])]
            if not text or not labels or len(text) != len(labels):
                continue
            samples.append((text, labels))
    return samples
