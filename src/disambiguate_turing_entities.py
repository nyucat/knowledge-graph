from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Mention:
    canonical: str
    entity_type: str
    mention: str
    start: int
    end: int


def load_mapping(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_mentions(text: str, mapping: list[dict]) -> list[Mention]:
    # longest alias first to reduce nested overlaps at same position
    alias_records: list[tuple[str, str, str]] = []
    for row in mapping:
        canonical = str(row["canonical"])
        etype = str(row["type"])
        for alias in row.get("aliases", []):
            a = str(alias).strip()
            if not a:
                continue
            alias_records.append((a, canonical, etype))
    alias_records.sort(key=lambda x: len(x[0]), reverse=True)

    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []

    for alias, canonical, etype in alias_records:
        flags = re.IGNORECASE if re.search(r"[A-Za-z]", alias) else 0
        for m in re.finditer(re.escape(alias), text, flags=flags):
            s, e = m.start(), m.end()
            # drop if fully covered by an existing longer match
            if any(s >= a and e <= b for a, b in occupied):
                continue
            mentions.append(Mention(canonical=canonical, entity_type=etype, mention=text[s:e], start=s, end=e))
            occupied.append((s, e))

    mentions.sort(key=lambda x: (x.start, x.end, x.canonical))
    return mentions


def build_output(mentions: list[Mention]) -> dict:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen_variant: dict[tuple[str, str], set[str]] = defaultdict(set)
    for m in mentions:
        k = (m.canonical, m.entity_type)
        if m.mention not in seen_variant[k]:
            seen_variant[k].add(m.mention)
            grouped[k].append(m.mention)

    canonical_entities = [
        {
            "canonical_name": k[0],
            "entity_type": k[1],
            "variants_found": v,
            "mention_count": sum(1 for x in mentions if x.canonical == k[0] and x.entity_type == k[1]),
        }
        for k, v in grouped.items()
    ]
    canonical_entities.sort(key=lambda x: (x["entity_type"], x["canonical_name"]))

    mentions_ordered = [
        {
            "canonical_name": m.canonical,
            "entity_type": m.entity_type,
            "mention": m.mention,
            "start": m.start,
            "end": m.end,
        }
        for m in mentions
    ]

    return {
        "canonical_entity_count": len(canonical_entities),
        "mention_count": len(mentions_ordered),
        "canonical_entities": canonical_entities,
        "mentions_in_text_order": mentions_ordered,
    }


def main() -> None:
    text_path = PROJECT_ROOT / "datafile" / "test.txt"
    mapping_path = PROJECT_ROOT / "data" / "turing_disambiguation.json"
    output_path = PROJECT_ROOT / "outputs" / "turing_entities_disambiguated.json"

    text = text_path.read_text(encoding="utf-8", errors="ignore")
    mapping = load_mapping(mapping_path)
    mentions = find_mentions(text, mapping)
    result = build_output(mentions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "canonical_entity_count": result["canonical_entity_count"],
        "mention_count": result["mention_count"],
        "source_kept": str(PROJECT_ROOT / "outputs" / "turing_entities.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
