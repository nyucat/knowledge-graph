from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cypher_quote(value: str) -> str:
    s = value.replace("\\", "\\\\").replace("'", "\\'")
    s = s.replace("\r", "\\r").replace("\n", "\\n")
    return f"'{s}'"


def load_triples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    triples = payload.get("triples", [])
    if not isinstance(triples, list):
        return []
    out = []
    for row in triples:
        if not isinstance(row, dict):
            continue
        head = str(row.get("head", "")).strip()
        relation = str(row.get("relation", "")).strip()
        tail = str(row.get("tail", "")).strip()
        if not head or not relation or not tail:
            continue
        out.append(row)
    return out


def load_entity_type_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    out: dict[str, str] = {}
    for row in payload.get("canonical_entities", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("canonical_name", "")).strip()
        et = str(row.get("entity_type", "")).strip()
        if name and et:
            out[name] = et
    for row in payload.get("mentions_in_text_order", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("canonical_name", "")).strip()
        et = str(row.get("entity_type", "")).strip()
        if name and et and name not in out:
            out[name] = et
    return out


TYPE_TO_LABEL = {
    "Person": "Person",
    "Location": "Location",
    "Location/Organization": "Organization",
    "Organization": "Organization",
    "Work": "Work",
    "Concept": "Concept",
    "Device": "Device",
    "Event": "Event",
    "Law": "Event",
    "Law/Event": "Event",
    "Award": "Award",
    "Title": "Award",
    "Degree": "Award",
    "Sport": "Sport",
    "Country": "Country",
    "Disease": "Disease",
    "Chemical": "Substance",
    "Date": "Date",
    "Time": "Numeric",
}

LABEL_ORDER = [
    "Person",
    "Location",
    "Organization",
    "Work",
    "Concept",
    "Device",
    "Event",
    "Award",
    "Sport",
    "Country",
    "Disease",
    "Substance",
    "Numeric",
    "Year",
    "Date",
    "Entity",
]

LABEL_CN = {
    "Person": "人物",
    "Location": "地点",
    "Organization": "组织",
    "Work": "作品",
    "Concept": "概念",
    "Device": "设备",
    "Event": "事件",
    "Award": "奖项/头衔",
    "Sport": "运动",
    "Country": "国家",
    "Disease": "疾病",
    "Substance": "物质",
    "Numeric": "数值",
    "Year": "年份",
    "Date": "日期",
    "Entity": "实体",
}

VAR_PREFIX = {
    "Person": "p",
    "Location": "l",
    "Organization": "o",
    "Work": "w",
    "Concept": "c",
    "Device": "d",
    "Event": "e",
    "Award": "a",
    "Sport": "s",
    "Country": "co",
    "Disease": "di",
    "Substance": "sb",
    "Numeric": "n",
    "Year": "y",
    "Date": "dt",
    "Entity": "x",
}

VALUE_KEY_LABELS = {"Numeric", "Year", "Date"}


def infer_label(name: str, entity_type_map: dict[str, str]) -> str:
    if name in entity_type_map:
        return TYPE_TO_LABEL.get(entity_type_map[name], "Entity")
    if re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", name):
        return "Date"
    if re.fullmatch(r"\d{4}年", name):
        return "Year"
    if re.search(r"\d", name):
        return "Numeric"
    return "Entity"


def rel_type(value: str) -> str:
    return value.replace("`", "")


def build_cypher(triples: list[dict], entity_type_map: dict[str, str]) -> str:
    # 1) de-duplicate triples
    uniq: list[tuple[str, str, str]] = []
    seen = set()
    for row in triples:
        h = str(row.get("head", "")).strip()
        r = str(row.get("relation", "")).strip()
        t = str(row.get("tail", "")).strip()
        if not h or not r or not t:
            continue
        k = (h, r, t)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)

    # 2) collect nodes and assign labels
    node_label: dict[str, str] = {}
    for h, _, t in uniq:
        node_label.setdefault(h, infer_label(h, entity_type_map))
        node_label.setdefault(t, infer_label(t, entity_type_map))

    # 3) assign vars by label groups
    grouped_nodes: dict[str, list[str]] = {lb: [] for lb in LABEL_ORDER}
    for name, lb in node_label.items():
        if lb not in grouped_nodes:
            grouped_nodes.setdefault("Entity", []).append(name)
        else:
            grouped_nodes[lb].append(name)
    for lb in grouped_nodes:
        grouped_nodes[lb] = sorted(set(grouped_nodes[lb]))

    node_var: dict[str, str] = {}
    for lb in LABEL_ORDER:
        prefix = VAR_PREFIX.get(lb, "x")
        for i, name in enumerate(grouped_nodes.get(lb, []), start=1):
            node_var[name] = f"{prefix}{i}"

    lines: list[str] = []
    used_labels = [lb for lb in LABEL_ORDER if grouped_nodes.get(lb)]
    lines.append("// 创建节点")
    lines.append("")
    for lb in used_labels:
        nodes = grouped_nodes.get(lb, [])
        if not nodes:
            continue
        cn = LABEL_CN.get(lb, lb)
        lines.append(f"// ---- {cn} ({lb}) ----")
        key = "value" if lb in VALUE_KEY_LABELS else "name"
        for name in nodes:
            lines.append(f"CREATE (:{lb} {{{key}: {cypher_quote(name)}}});")
        lines.append("")

    lines.append("// 创建关系")
    lines.append("")
    # MATCH section
    match_rows: list[str] = []
    for name in sorted(node_var.keys(), key=lambda x: node_var[x]):
        lb = node_label[name]
        var = node_var[name]
        key = "value" if lb in VALUE_KEY_LABELS else "name"
        match_rows.append(f"({var}:{lb} {{{key}: {cypher_quote(name)}}})")

    lines.append("MATCH")
    for i, row in enumerate(match_rows):
        suffix = "," if i < len(match_rows) - 1 else ""
        lines.append(f"  {row}{suffix}")
    lines.append("")

    # CREATE relationships section in same style
    lines.append("CREATE")
    for i, (h, r, t) in enumerate(uniq):
        hv = node_var[h]
        tv = node_var[t]
        rel = rel_type(r)
        suffix = "," if i < len(uniq) - 1 else ";"
        lines.append(f"  ({hv})-[:`{rel}`]->({tv}){suffix}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cypher commands from turing_kg_triples.json")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "outputs" / "turing_kg_triples.json"),
        help="Input triples JSON path",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "CY.txt"),
        help="Output Cypher text path",
    )
    parser.add_argument(
        "--entity-types",
        default=str(PROJECT_ROOT / "outputs" / "turing_entities_disambiguated.json"),
        help="Entity type mapping JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    entity_types_path = Path(args.entity_types)

    triples = load_triples(input_path)
    entity_type_map = load_entity_type_map(entity_types_path)
    cypher_text = build_cypher(triples=triples, entity_type_map=entity_type_map)
    output_path.write_text(cypher_text, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "entity_types": str(entity_types_path),
                "triple_count_in": len(triples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
