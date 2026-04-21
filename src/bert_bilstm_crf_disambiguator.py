from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

from crf_ner import FOURTEEN_LABELS, Entity, convert_bio_to_entities, load_ner_jsonl


@dataclass
class MentionRecord:
    entity_text: str
    start: int
    end: int
    coarse_label: str


class MentionSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Tuple[str, List[Entity]]],
        tokenizer,
        label2id: dict[str, int],
        context_window: int,
        mention_max_length: int,
    ) -> None:
        self.rows: list[dict] = []
        for text, mentions in samples:
            sorted_mentions = sorted(mentions, key=lambda x: (x.start, x.end))
            if not sorted_mentions:
                continue
            contexts: list[list[int]] = []
            attns: list[list[int]] = []
            tags: list[int] = []
            for m in sorted_mentions:
                label = m.label if m.label in label2id else "ORG"
                tags.append(label2id[label])

                left = max(0, m.start - context_window)
                right = min(len(text), m.end + context_window)
                mention_context = text[left:right]
                encoded = tokenizer(
                    mention_context,
                    truncation=True,
                    max_length=mention_max_length,
                    return_attention_mask=True,
                )
                contexts.append(list(encoded["input_ids"]))
                attns.append(list(encoded["attention_mask"]))

            self.rows.append(
                {
                    "contexts": contexts,
                    "attentions": attns,
                    "tags": tags,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _collate_mentions(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_mentions = max(len(item["tags"]) for item in batch)
    max_ctx_len = max(len(ctx) for item in batch for ctx in item["contexts"])

    context_ids = torch.full((batch_size, max_mentions, max_ctx_len), pad_id, dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_mentions, max_ctx_len), dtype=torch.long)
    tags = torch.zeros((batch_size, max_mentions), dtype=torch.long)
    seq_mask = torch.zeros((batch_size, max_mentions), dtype=torch.bool)

    for bi, item in enumerate(batch):
        mlen = len(item["tags"])
        for mi in range(mlen):
            ids = item["contexts"][mi]
            attn = item["attentions"][mi]
            context_ids[bi, mi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            context_mask[bi, mi, : len(attn)] = torch.tensor(attn, dtype=torch.long)
            tags[bi, mi] = int(item["tags"][mi])
            seq_mask[bi, mi] = True

    return {
        "context_ids": context_ids,
        "context_mask": context_mask,
        "tags": tags,
        "seq_mask": seq_mask,
    }


class MentionBertBiLSTMCRF(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lstm_hidden_size: int = 256,
        dropout: float = 0.2,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.bert.config.hidden_size)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _encode_mentions(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        bsz, mention_len, seq_len = context_ids.shape
        flat_ids = context_ids.view(bsz * mention_len, seq_len)
        flat_mask = context_mask.view(bsz * mention_len, seq_len)

        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb.view(bsz, mention_len, -1)

    def emissions(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        mention_emb = self._encode_mentions(context_ids, context_mask)
        lstm_out, _ = self.lstm(mention_emb)
        return self.classifier(self.dropout(lstm_out))

    def compute_loss(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        tags: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        emissions = self.emissions(context_ids, context_mask)
        log_likelihood = self.crf(emissions, tags, mask=seq_mask, reduction="mean")
        return -log_likelihood

    def decode(self, context_ids: torch.Tensor, context_mask: torch.Tensor, seq_mask: torch.Tensor) -> list[list[int]]:
        emissions = self.emissions(context_ids, context_mask)
        return self.crf.decode(emissions, mask=seq_mask)


class BertBiLSTMCRFEntityDisambiguator:
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        context_window: int = 36,
        mention_max_length: int = 96,
        epochs: int = 4,
        batch_size: int = 4,
        lr: float = 1e-3,
        freeze_bert: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.context_window = context_window
        self.mention_max_length = mention_max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.freeze_bert = freeze_bert
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.labels = sorted(FOURTEEN_LABELS)
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MentionBertBiLSTMCRF(
            model_name=model_name,
            num_labels=len(self.labels),
            freeze_bert=freeze_bert,
        ).to(self.device)
        self.is_trained = False

    @staticmethod
    def _normalize_label(label: str) -> str:
        norm = (label or "").upper().strip()
        if "." in norm:
            norm = norm.split(".", 1)[0]
        if norm in FOURTEEN_LABELS:
            return norm
        alias = {
            "PERSON": "PER",
            "PEOPLE": "PER",
            "HUMAN": "PER",
            "ORGANIZATION": "ORG",
            "LOCATION": "LOC",
            "GPE": "LOC",
            "DATETIME": "DATE",
            "DISCIPLINE": "SUBJECT",
            "PRODUCT": "PROD",
        }
        return alias.get(norm, "ORG")

    def _load_samples_from_disambiguation_jsonl(self, path: Path) -> list[tuple[str, list[Entity]]]:
        if not path.exists():
            return []
        docs: list[tuple[str, list[Entity]]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = str(row.get("text", ""))
                mentions = row.get("mentions", [])
                if not text or not isinstance(mentions, list):
                    continue
                entities: list[Entity] = []
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
                    label = self._normalize_label(str(m.get("label", "")))
                    entities.append(Entity(text=text[start:end], label=label, start=start, end=end))
                if entities:
                    docs.append((text, sorted(entities, key=lambda x: (x.start, x.end))))
        return docs

    def _load_samples_from_ner_jsonl(self, path: Path) -> list[tuple[str, list[Entity]]]:
        samples = load_ner_jsonl(path)
        docs: list[tuple[str, list[Entity]]] = []
        for text, tags in samples:
            entities = convert_bio_to_entities(text, tags)
            if entities:
                docs.append((text, sorted(entities, key=lambda x: (x.start, x.end))))
        return docs

    def train_from_jsonl(self, disambiguation_train_path: Path | None, ner_fallback_path: Path) -> str:
        train_samples: list[tuple[str, list[Entity]]] = []
        source = "ner_fallback"

        if disambiguation_train_path is not None:
            train_samples = self._load_samples_from_disambiguation_jsonl(disambiguation_train_path)
            if train_samples:
                source = "disambiguation_jsonl"

        if not train_samples:
            train_samples = self._load_samples_from_ner_jsonl(ner_fallback_path)

        dataset = MentionSequenceDataset(
            samples=train_samples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            context_window=self.context_window,
            mention_max_length=self.mention_max_length,
        )
        if len(dataset) == 0:
            raise ValueError("消歧训练样本为空。")

        pad_id = self.tokenizer.pad_token_id or 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: _collate_mentions(x, pad_id),
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr)
        self.model.train()

        for _ in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                loss = self.model.compute_loss(
                    context_ids=batch["context_ids"].to(self.device),
                    context_mask=batch["context_mask"].to(self.device),
                    tags=batch["tags"].to(self.device),
                    seq_mask=batch["seq_mask"].to(self.device),
                )
                loss.backward()
                optimizer.step()

        self.is_trained = True
        return source

    def save(self, path: Path) -> None:
        if not self.is_trained:
            raise RuntimeError("消歧模型尚未训练。")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "context_window": self.context_window,
            "mention_max_length": self.mention_max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "freeze_bert": self.freeze_bert,
            "labels": self.labels,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model_name = str(payload.get("model_name", self.model_name))
        self.context_window = int(payload.get("context_window", self.context_window))
        self.mention_max_length = int(payload.get("mention_max_length", self.mention_max_length))
        self.epochs = int(payload.get("epochs", self.epochs))
        self.batch_size = int(payload.get("batch_size", self.batch_size))
        self.lr = float(payload.get("lr", self.lr))
        self.freeze_bert = bool(payload.get("freeze_bert", self.freeze_bert))

        self.labels = [str(x) for x in payload["labels"]]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MentionBertBiLSTMCRF(
            model_name=self.model_name,
            num_labels=len(self.labels),
            freeze_bert=self.freeze_bert,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.is_trained = True

    def _build_sequence_batch(self, mentions: list[MentionRecord], doc_text: str) -> dict[str, torch.Tensor]:
        contexts: list[list[int]] = []
        attns: list[list[int]] = []
        for m in mentions:
            left = max(0, m.start - self.context_window)
            right = min(len(doc_text), m.end + self.context_window)
            segment = doc_text[left:right]
            encoded = self.tokenizer(
                segment,
                truncation=True,
                max_length=self.mention_max_length,
                return_attention_mask=True,
            )
            contexts.append(list(encoded["input_ids"]))
            attns.append(list(encoded["attention_mask"]))

        max_ctx_len = max(len(x) for x in contexts)
        mlen = len(contexts)
        pad_id = self.tokenizer.pad_token_id or 0

        context_ids = torch.full((1, mlen, max_ctx_len), pad_id, dtype=torch.long)
        context_mask = torch.zeros((1, mlen, max_ctx_len), dtype=torch.long)
        seq_mask = torch.ones((1, mlen), dtype=torch.bool)

        for i in range(mlen):
            ids = contexts[i]
            am = attns[i]
            context_ids[0, i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            context_mask[0, i, : len(am)] = torch.tensor(am, dtype=torch.long)

        return {
            "context_ids": context_ids.to(self.device),
            "context_mask": context_mask.to(self.device),
            "seq_mask": seq_mask.to(self.device),
        }

    def predict_parent_labels(self, entity_records: list[dict], text_by_url: Dict[str, str]) -> list[str]:
        if not self.is_trained:
            return [str(row.get("entity_parent_label", "ORG")) for row in entity_records]

        output = [str(row.get("entity_parent_label", "ORG")) for row in entity_records]
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(entity_records):
            grouped_indices[str(row.get("page_url", ""))].append(i)

        self.model.eval()
        with torch.no_grad():
            for page_url, indices in grouped_indices.items():
                doc_text = text_by_url.get(page_url, "")
                ordered = sorted(indices, key=lambda idx: (int(entity_records[idx].get("start", -1)), int(entity_records[idx].get("end", -1))))
                mentions: list[MentionRecord] = []
                valid_idx: list[int] = []
                for idx in ordered:
                    row = entity_records[idx]
                    text = str(row.get("entity_text", "")).strip()
                    if not text:
                        continue
                    mentions.append(
                        MentionRecord(
                            entity_text=text,
                            start=int(row.get("start", -1)),
                            end=int(row.get("end", -1)),
                            coarse_label=str(row.get("entity_parent_label", "ORG")),
                        )
                    )
                    valid_idx.append(idx)

                if not mentions:
                    continue

                batch = self._build_sequence_batch(mentions, doc_text)
                pred_ids = self.model.decode(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    seq_mask=batch["seq_mask"],
                )[0]
                for i, pid in enumerate(pred_ids):
                    if i >= len(valid_idx):
                        break
                    output[valid_idx[i]] = self.id2label.get(pid, "ORG")

        return output
