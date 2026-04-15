from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


class KnowledgeGraphBuilder:
    def __init__(self) -> None:
        self.entities: Set[str] = set()
        self.triples: Set[Triple] = set()

    def add_taxonomy_from_wiki(self, page_title: str, categories: Iterable[str]) -> None:
        self.entities.add(page_title)
        for cat in categories:
            self.entities.add(cat)
            self.triples.add(Triple(head=page_title, relation="属于类别", tail=cat))

    def add_entity_mentions(self, page_title: str, entities: Iterable[Dict[str, str]]) -> None:
        self.entities.add(page_title)
        for ent in entities:
            ent_text = ent.get("text", "").strip()
            ent_label = ent.get("label", "UNKNOWN").strip() or "UNKNOWN"
            if not ent_text:
                continue
            typed_entity = f"{ent_text}::{ent_label}"
            self.entities.add(typed_entity)
            self.triples.add(Triple(head=page_title, relation="提及实体", tail=typed_entity))

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "entities": [{"id": e} for e in sorted(self.entities)],
            "triples": [
                {"head": t.head, "relation": t.relation, "tail": t.tail}
                for t in sorted(self.triples, key=lambda x: (x.head, x.relation, x.tail))
            ],
        }

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
