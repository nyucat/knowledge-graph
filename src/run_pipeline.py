from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import jieba.posseg as pseg
from cogie_adapter import CogIEEntityRecognizer
from crf_disambiguator import CRFEntityDisambiguator
from crf_ner import CRFEntityRecognizer, ensure_label_coverage, load_ner_jsonl
from fine_grained_typing import FineGrainedEntityTyper
from text_sources import TextDocument, build_web_txt_from_urls, load_local_txt_documents, parse_urls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERAL_STOPWORDS = {
    "的",
    "了",
    "在",
    "和",
    "与",
    "及",
    "或",
    "并",
    "而",
    "是",
    "有",
    "就",
    "都",
    "也",
    "中",
    "上",
    "下",
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def train_or_load_crf(
    train_path: Path, model_path: Path, reuse_existing_model: bool
) -> CRFEntityRecognizer:
    crf = CRFEntityRecognizer()
    if reuse_existing_model and model_path.exists():
        crf.load(model_path)
        return crf
    samples = load_ner_jsonl(train_path)
    ensure_label_coverage(samples)
    crf.train(samples)
    crf.save(model_path)
    return crf


def train_or_load_disambiguator(
    disambiguation_train_path: Path | None,
    ner_fallback_path: Path,
    model_path: Path,
    reuse_existing_model: bool,
) -> tuple[CRFEntityDisambiguator, str]:
    disambiguator = CRFEntityDisambiguator()
    if reuse_existing_model and model_path.exists():
        disambiguator.load(model_path)
        return disambiguator, "cached_model"
    train_source = disambiguator.train_from_jsonl(
        disambiguation_train_path=disambiguation_train_path,
        ner_fallback_path=ner_fallback_path,
    )
    disambiguator.save(model_path)
    return disambiguator, train_source


def generic_fine_label_for_parent(parent_label: str) -> str:
    mapping = {
        "PER": "PER.Scientist",
        "ORG": "ORG.Company",
        "LOC": "LOC.City",
        "TIME": "TIME.ClockTime",
        "DATE": "DATE.CalendarDate",
        "MONEY": "MONEY.Currency",
        "PERCENT": "PERCENT.Ratio",
        "SUBJECT": "SUBJECT.ComputerScience",
        "EVENT": "EVENT.Conference",
        "WORK": "WORK.Book",
        "PROD": "PROD.SoftwareOrModel",
        "THEORY": "THEORY.LawOrTheorem",
        "FIELD": "FIELD.ResearchField",
        "TECH": "TECH.Technology",
    }
    return mapping.get(parent_label, "ORG.Company")


def apply_crf_disambiguation(
    entity_records: list[dict],
    docs: list[TextDocument],
    disambiguator: CRFEntityDisambiguator,
) -> list[dict]:
    if not entity_records:
        return entity_records
    text_by_url = {doc.source_url: doc.text for doc in docs}
    predicted_parents = disambiguator.predict_parent_labels(entity_records, text_by_url)
    for row, predicted_parent in zip(entity_records, predicted_parents):
        if not predicted_parent:
            continue
        old_parent = str(row.get("entity_parent_label", ""))
        if predicted_parent == old_parent:
            pass
        else:
            row["entity_parent_label"] = predicted_parent
            row["entity_label"] = generic_fine_label_for_parent(predicted_parent)

        # 人名别名规范：X·Y / X·Y·Z 链接为最后一段（例如“艾伦·麦席森·图灵” -> “图灵”）。
        if str(row.get("entity_parent_label", "")) == "PER":
            mention = str(row.get("entity_text", "")).strip()
            if "·" in mention:
                parts = [p.strip() for p in mention.split("·") if p.strip()]
                if parts:
                    canonical = parts[-1]
                    if re.fullmatch(r"[\u4e00-\u9fff]{1,10}", canonical):
                        row["entity_text"] = canonical
                        try:
                            end = int(row.get("end", -1))
                            if end >= 0:
                                row["start"] = max(0, end - len(canonical))
                        except (TypeError, ValueError):
                            pass
    return entity_records


def prune_middle_dot_person_segments(entity_records: list[dict], docs: list[TextDocument]) -> list[dict]:
    text_by_url = {doc.source_url: doc.text for doc in docs}
    pruned: list[dict] = []
    linked_name_pattern = re.compile(
        r"[\u4e00-\u9fff]{1,8}[·•・‧∙･][\u4e00-\u9fff]{1,10}(?:[·•・‧∙･][\u4e00-\u9fff]{1,10}){0,2}"
    )
    for row in entity_records:
        if str(row.get("entity_parent_label", "")) != "PER":
            pruned.append(row)
            continue
        page_url = str(row.get("page_url", ""))
        text = text_by_url.get(page_url, "")
        try:
            start = int(row.get("start", -1))
            end = int(row.get("end", -1))
        except (TypeError, ValueError):
            pruned.append(row)
            continue
        if start >= 0 and end > start:
            ws = max(0, start - 10)
            we = min(len(text), end + 10)
            window = text[ws:we]
            for m in linked_name_pattern.finditer(window):
                abs_s = ws + m.start()
                abs_e = ws + m.end()
                if abs_s <= start and end <= abs_e:
                    # 位于点连接姓名内部，但不是尾段（尾段通常作为规范链接保留）。
                    if end < abs_e:
                        start = -1
                    break
            if start == -1:
                continue
        pruned.append(row)
    return pruned


def collapse_conflicting_entities(entity_records: list[dict]) -> list[dict]:
    priority = {
        "PER": 9,
        "ORG": 8,
        "LOC": 7,
        "EVENT": 6,
        "WORK": 5,
        "PROD": 5,
        "THEORY": 4,
        "TECH": 4,
        "SUBJECT": 3,
        "FIELD": 2,
        "DATE": 2,
        "TIME": 2,
        "MONEY": 2,
        "PERCENT": 2,
    }
    best: dict[tuple, dict] = {}
    for row in entity_records:
        key = (
            row.get("page_url", ""),
            row.get("start", -1),
            row.get("end", -1),
            row.get("entity_text", ""),
        )
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        cur_parent = str(current.get("entity_parent_label", ""))
        new_parent = str(row.get("entity_parent_label", ""))
        cur_score = (priority.get(cur_parent, 0), float(current.get("typing_score", 0.0)))
        new_score = (priority.get(new_parent, 0), float(row.get("typing_score", 0.0)))
        if new_score > cur_score:
            best[key] = row
    return list(best.values())


def save_entities_json(entity_records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(entity_records, f, ensure_ascii=False, indent=2)


def save_entities_csv(entity_records: list[dict], path: Path) -> None:
    fieldnames = [
        "entity_text",
        "entity_label",
        "entity_parent_label",
        "typing_score",
        "start",
        "end",
        "page_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entity_records:
            writer.writerow(row)


def is_valid_entity_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 30:
        return False
    if re.search(r"\[\s*\d+\s*\]", cleaned):
        return False
    if any(ch in cleaned for ch in "[]{}()\"'，。！？；："):
        return False
    if cleaned.count(" ") > 1:
        return False
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", cleaned):
        return False
    return True


def normalize_coarse_label(raw_label: str, entity_text: str) -> str:
    label = (raw_label or "").upper().strip()
    text = entity_text.strip()
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
    return alias.get(label, label if label else "ORG")


def extract_jieba_entities(text: str, high_recall: bool = False) -> list[tuple[str, int, int, str]]:
    """
    使用 jieba 对文本进行实体候选检测，并返回：
    (实体文本, start, end, 粗粒度标签)
    """
    results: list[tuple[str, int, int, str]] = []
    cursor = 0
    allowed_flags = {"nr", "ns", "s", "nt", "nz", "n", "vn", "an", "eng"}
    if high_recall:
        allowed_flags |= {"j", "l", "i", "v", "t", "m"}
    # 使用通用形态线索，避免依赖具体词条白名单。
    strong_shape_pattern = re.compile(
        r"(?:"
        r"委员会|出版社|研究院|研究所|实验室|大学|学院|集团|公司|协会|联盟|中心|银行|机构|"
        r"制度|机制|体系|学制|方案|规划|平台|技术|模型|系统|工程|项目"
        r")$"
    )
    for token in pseg.cut(text):
        word = token.word.strip()
        flag = token.flag
        if not word:
            continue
        has_strong_suffix = bool(strong_shape_pattern.search(word))
        if flag not in allowed_flags and not has_strong_suffix:
            continue
        if word in GENERAL_STOPWORDS:
            continue
        if len(word) <= 1:
            continue
        # 默认过滤大量“两字泛词”，仅保留更像实体的候选。
        is_short_allowed = bool(
            re.fullmatch(r"\d+[Vv]\d+", word)
            or word.lower() in {"ai", "ag", "gpt", "cv", "nlp", "kg", "re", "ner", "ie"}
            or word.lower() in {"buff", "debuff"}
            or (high_recall and word.lower() in {"llm", "rag", "iot", "api"})
        )
        min_len = 2 if high_recall else 3
        if len(word) < min_len and not (is_short_allowed or has_strong_suffix):
            continue
        # 忽略纯标点
        if re.fullmatch(r"\W+", word):
            continue
        if re.fullmatch(r"\d+", word):
            continue
        if re.fullmatch(r"[A-Za-z]{1,2}", word):
            continue

        idx = text.find(word, cursor)
        if idx < 0:
            idx = text.find(word)
        if idx < 0:
            continue
        cursor = idx + len(word)
        if not is_valid_entity_text(word):
            continue

        coarse = "ORG"
        if flag == "nr":
            coarse = "PER"
        elif flag in {"ns", "s"}:
            coarse = "LOC"
        elif flag in {"nt", "nz"}:
            coarse = "ORG"
        elif flag in {"t"}:
            coarse = "DATE"
        elif flag in {"m"} and re.search(r"\d", word):
            coarse = "TIME"
        elif flag in {"n", "vn", "an"} or has_strong_suffix:
            coarse = "FIELD"

        coarse = normalize_coarse_label(coarse, word)
        results.append((word, idx, idx + len(word), coarse))
    return results


def extract_rule_entities(text: str, high_recall: bool = False) -> list[tuple[str, int, int, str]]:
    results: list[tuple[str, int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    def add_candidate(start: int, end: int, coarse: str) -> None:
        if start < 0 or end <= start:
            return
        word = text[start:end].strip()
        if not word or not is_valid_entity_text(word):
            return
        key = (start, end, coarse)
        if key in seen:
            return
        seen.add(key)
        results.append((word, start, end, normalize_coarse_label(coarse, word)))

    # 姓名连接符规则始终开启：避免“艾伦·麦席森·图灵”被切碎。
    linked_name_rules: list[tuple[str, str]] = [
        (r"[\u4e00-\u9fff]{1,6}(?:·[\u4e00-\u9fff]{1,10}){1,3}", "PER"),
    ]
    for pattern, coarse in linked_name_rules:
        for m in re.finditer(pattern, text):
            add_candidate(m.start(), m.end(), coarse)

    if not high_recall:
        results.sort(key=lambda x: (x[1], x[2], x[0]))
        return results

    regex_rules: list[tuple[str, str]] = [
        (
            r"[\u4e00-\u9fffA-Za-z0-9]{2,30}"
            r"(?:委员会|出版社|研究院|研究所|实验室|大学|学院|集团|公司|协会|联盟|中心|银行|机构)",
            "ORG",
        ),
        (
            r"[\u4e00-\u9fffA-Za-z0-9]{2,30}"
            r"(?:一等奖|二等奖|三等奖|特等奖|优胜奖|金奖|银奖|铜奖|几等奖)",
            "EVENT",
        ),
        (
            r"[\u4e00-\u9fffA-Za-z0-9]{2,24}"
            r"(?:制度|机制|体系|学制|方案|规划|工程|项目|框架|流程|标准)",
            "FIELD",
        ),
    ]
    for pattern, coarse in regex_rules:
        for m in re.finditer(pattern, text):
            add_candidate(m.start(), m.end(), coarse)

    results.sort(key=lambda x: (x[1], x[2], x[0]))
    return results


def collect_documents(data_dir: Path, urls: list[str]) -> tuple[list[TextDocument], list[TextDocument]]:
    local_docs = load_local_txt_documents(data_dir)
    if not local_docs:
        print("无本地数据")

    web_txt_path = PROJECT_ROOT / "webfile" / "web.txt"
    web_doc = build_web_txt_from_urls(urls, web_txt_path)
    web_docs = [web_doc] if web_doc is not None else []
    return local_docs, web_docs


def run(
    urls: list[str],
    data_dir: Path,
    ner_backend: str,
    train_data: Path,
    crf_model: Path,
    disambiguation_train_data: Path | None,
    reuse_existing_model: bool,
    output_dir: Path,
    high_recall: bool,
    use_crf_disambiguation: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    typer = FineGrainedEntityTyper()
    local_docs, web_docs = collect_documents(data_dir=data_dir, urls=urls)
    docs = local_docs + web_docs
    total_entities = 0
    entity_records: list[dict] = []
    seen: set[tuple] = set()
    boundary_punct = " \t\r\n\"'，。！？；：、,.!?;:()（）[]{}<>《》"
    weak_boundary_chars = set("的了和与在并及或而是有被把将于对从向按以为及其该这那")

    def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        if min(a_start, a_end, b_start, b_end) < 0:
            return False
        return max(a_start, b_start) < min(a_end, b_end)

    def recover_fragment_entity(
        source_text: str,
        start: int,
        end: int,
        cleaned: str,
    ) -> tuple[int, int, str]:
        """
        通用碎片修复：
        - 先尝试把明显后缀残片向左补全（如 大学/年制/分制 等）
        - 若仍是无语义后缀残片，则丢弃
        """
        if start < 0 or end < 0 or start >= len(source_text):
            return start, end, cleaned

        suffix_min_len = {
            "大学": 4,
            "学院": 4,
            "公司": 4,
            "集团": 4,
            "银行": 4,
            "委员会": 5,
            "出版社": 5,
            "研究院": 5,
            "研究所": 5,
            "实验室": 5,
            "制度": 4,
            "机制": 4,
            "体系": 4,
            "学制": 4,
            "同校": 4,
            "年制": 3,
            "分制": 3,
        }

        matched_suffix = ""
        target_len = 0
        for suffix, min_len in suffix_min_len.items():
            if cleaned.endswith(suffix):
                matched_suffix = suffix
                target_len = min_len
                break
        if not matched_suffix:
            return start, end, cleaned

        # 左扩补全到最小语义长度，最多扩展 8 个字符，避免吞句。
        left = start
        max_expand = 8
        while left > 0 and len(cleaned) < target_len and max_expand > 0:
            prev = source_text[left - 1]
            if prev in boundary_punct or prev in weak_boundary_chars:
                break
            if not re.match(r"[\u4e00-\u9fffA-Za-z0-9]", prev):
                break
            cleaned = prev + cleaned
            left -= 1
            max_expand -= 1

        # 仍然是后缀残片则标记为空，让上游直接丢弃。
        suffix_only = {
            "大学",
            "学院",
            "公司",
            "集团",
            "银行",
            "委员会",
            "出版社",
            "研究院",
            "研究所",
            "实验室",
            "制度",
            "机制",
            "体系",
            "学制",
            "同校",
            "年制",
            "分制",
        }
        if cleaned in suffix_only:
            return left, left, ""

        return left, left + len(cleaned), cleaned

    def append_record(
        entity_text: str,
        entity_label: str,
        entity_parent_label: str,
        typing_score: float,
        start: int,
        end: int,
        page_url: str,
        dedupe_scope: str,
        source_text: str,
    ) -> None:
        raw_text = entity_text.strip()
        if not raw_text:
            return

        left = 0
        right = len(raw_text)
        while left < right and raw_text[left] in boundary_punct:
            left += 1
        while right > left and raw_text[right - 1] in boundary_punct:
            right -= 1
        cleaned = raw_text[left:right].strip()
        if not cleaned:
            return

        if start >= 0 and end >= 0:
            start = start + left
            end = start + len(cleaned)
            if end <= start:
                return

        start, end, cleaned = recover_fragment_entity(
            source_text=source_text,
            start=start,
            end=end,
            cleaned=cleaned,
        )
        if not cleaned:
            return

        if not cleaned:
            return
        # 轻量去噪：不过滤分数，仅过滤明显噪声
        if cleaned in GENERAL_STOPWORDS:
            return
        if len(cleaned) == 1:
            return
        # 用户要求：抽取实体不包含英文字符。
        if re.search(r"[A-Za-zＡ-Ｚａ-ｚ]", cleaned):
            return
        if re.fullmatch(r"\d+", cleaned):
            return
        if re.fullmatch(r"[A-Za-z]{1,2}", cleaned):
            return
        if not is_valid_entity_text(cleaned):
            return

        key = (dedupe_scope, cleaned, entity_label, start, end)
        if key in seen:
            return
        seen.add(key)
        entity_records.append(
            {
                "entity_text": cleaned,
                "entity_label": entity_label,
                "entity_parent_label": entity_parent_label,
                "typing_score": round(typing_score, 3),
                "start": start,
                "end": end,
                "page_url": page_url,
            }
        )

    if ner_backend == "crf":
        recognizer = train_or_load_crf(
            train_path=train_data,
            model_path=crf_model,
            reuse_existing_model=reuse_existing_model,
        )
        for doc in docs:
            entities = recognizer.predict_entities(doc.text)
            crf_spans: list[tuple[int, int]] = []
            for e in entities:
                coarse_label = normalize_coarse_label(e.label, e.text)
                typing = typer.predict(
                    entity_text=e.text,
                    page_text=doc.text,
                    start=e.start,
                    end=e.end,
                    coarse_label=coarse_label,
                )
                append_record(
                    entity_text=e.text,
                    entity_label=typing.fine_label,
                    entity_parent_label=typing.parent_label,
                    typing_score=typing.score,
                    start=e.start,
                    end=e.end,
                    page_url=doc.source_url,
                    dedupe_scope=doc.title,
                    source_text=doc.text,
                )
                crf_spans.append((e.start, e.end))
            supplement_spans = list(crf_spans)
            supplement_candidates = extract_jieba_entities(doc.text, high_recall=high_recall)
            supplement_candidates += extract_rule_entities(doc.text, high_recall=high_recall)
            for word, start, end, coarse_label in supplement_candidates:
                # 补充候选只在未覆盖跨度上生效，避免重复堆叠。
                overlap_exists = any(spans_overlap(start, end, s, e) for s, e in supplement_spans)
                if overlap_exists and not (coarse_label == "PER" and "·" in word):
                    continue
                typing = typer.predict(
                    entity_text=word,
                    page_text=doc.text,
                    start=start,
                    end=end,
                    coarse_label=coarse_label,
                )
                append_record(
                    entity_text=word,
                    entity_label=typing.fine_label,
                    entity_parent_label=typing.parent_label,
                    typing_score=typing.score,
                    start=start,
                    end=end,
                    page_url=doc.source_url,
                    dedupe_scope=doc.title,
                    source_text=doc.text,
                )
                supplement_spans.append((start, end))
            total_entities = len(entity_records)
    else:
        recognizer = CogIEEntityRecognizer(language="english", corpus="trex")
        for doc in docs:
            entities = recognizer.predict(doc.text)
            base_spans: list[tuple[int, int]] = []
            for e in entities:
                text = e.get("text", "")
                coarse_label = normalize_coarse_label(e.get("label", "UNKNOWN"), text)
                idx = doc.text.find(text)
                start = idx if idx >= 0 else -1
                end = (idx + len(text)) if idx >= 0 else -1
                typing = typer.predict(
                    entity_text=text,
                    page_text=doc.text,
                    start=start,
                    end=end,
                    coarse_label=coarse_label,
                )
                append_record(
                    entity_text=text,
                    entity_label=typing.fine_label,
                    entity_parent_label=typing.parent_label,
                    typing_score=typing.score,
                    start=start,
                    end=end,
                    page_url=doc.source_url,
                    dedupe_scope=doc.title,
                    source_text=doc.text,
                )
                base_spans.append((start, end))
            supplement_spans = list(base_spans)
            supplement_candidates = extract_jieba_entities(doc.text, high_recall=high_recall)
            supplement_candidates += extract_rule_entities(doc.text, high_recall=high_recall)
            for word, start, end, coarse_label in supplement_candidates:
                overlap_exists = any(spans_overlap(start, end, s, e) for s, e in supplement_spans)
                if overlap_exists and not (coarse_label == "PER" and "·" in word):
                    continue
                typing = typer.predict(
                    entity_text=word,
                    page_text=doc.text,
                    start=start,
                    end=end,
                    coarse_label=coarse_label,
                )
                append_record(
                    entity_text=word,
                    entity_label=typing.fine_label,
                    entity_parent_label=typing.parent_label,
                    typing_score=typing.score,
                    start=start,
                    end=end,
                    page_url=doc.source_url,
                    dedupe_scope=doc.title,
                    source_text=doc.text,
                )
                supplement_spans.append((start, end))
            total_entities = len(entity_records)

    json_path = output_dir / "entities.json"
    csv_path = output_dir / "entities.csv"

    # 按实体在原文中的位置输出，避免 CRF/jieba 结果混排导致同类实体堆叠。
    entity_records.sort(
        key=lambda r: (
            r.get("page_url", ""),
            r.get("start", -1) if r.get("start", -1) >= 0 else 10**9,
            r.get("end", -1) if r.get("end", -1) >= 0 else 10**9,
            r.get("entity_text", ""),
        )
    )

    if use_crf_disambiguation and entity_records:
        disamb_model_path = output_dir / "crf_disambiguator.bin"
        disambiguator, disamb_train_source = train_or_load_disambiguator(
            disambiguation_train_path=disambiguation_train_data,
            ner_fallback_path=train_data,
            model_path=disamb_model_path,
            reuse_existing_model=reuse_existing_model,
        )
        entity_records = apply_crf_disambiguation(entity_records, docs, disambiguator)
    else:
        disamb_train_source = "disabled"

    entity_records = prune_middle_dot_person_segments(entity_records, docs)
    entity_records = collapse_conflicting_entities(entity_records)

    save_entities_json(entity_records, json_path)
    save_entities_csv(entity_records, csv_path)

    # 清理旧版输出，避免 outputs 中出现无关文件造成困惑。
    for obsolete in ["kg.json", "wiki_pages.jsonl", "report.json"]:
        old_file = output_dir / obsolete
        if old_file.exists():
            old_file.unlink()

    result = {
        "local_documents": len(local_docs),
        "web_documents": len(web_docs),
        "total_documents": len(docs),
        "ner_backend": ner_backend,
        "high_recall": high_recall,
        "crf_disambiguation": use_crf_disambiguation,
        "disambiguation_method": "bow_cosine + crf_global_lbp" if use_crf_disambiguation else "disabled",
        "disambiguation_train_source": disamb_train_source,
        "entities_extracted": total_entities,
        "entities_json_path": str(json_path),
        "entities_csv_path": str(csv_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地文本+链接文本实体抽取（CRF/CogIE）")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="0到多个网页链接（可粘贴一段含链接文本）",
    )
    parser.add_argument(
        "--data-dir",
        default="datafile",
        help="本地txt目录，默认 datafile",
    )
    parser.add_argument(
        "--ner-backend",
        choices=["crf", "cogie"],
        default="crf",
        help="实体识别后端，默认 crf",
    )
    parser.add_argument(
        "--train-data",
        default="data/sample_ner_train.jsonl",
        help="CRF 训练数据（jsonl）",
    )
    parser.add_argument(
        "--crf-model",
        default="outputs/crf_model.bin",
        help="CRF 模型输出路径",
    )
    parser.add_argument(
        "--disambiguation-train-data",
        default="data/sample_disambiguation_train.jsonl",
        help="消歧训练数据（jsonl，含 mentions）；缺失时自动回退到 --train-data",
    )
    parser.add_argument(
        "--reuse-crf-model",
        action="store_true",
        help="若提供该参数，则优先加载已存在的 CRF 模型，不重新训练",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="高召回模式：放宽 jieba 候选并启用规则词典补抽取",
    )
    parser.add_argument(
        "--disable-crf-disambiguation",
        action="store_true",
        help="关闭实体消歧（默认使用词袋余弦 + CRF全局LBP）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_data_path = resolve_path(args.train_data)
    disambiguation_train_path = resolve_path(args.disambiguation_train_data)
    crf_model_path = resolve_path(args.crf_model)
    output_dir_path = resolve_path(args.output_dir)
    data_dir_path = resolve_path(args.data_dir)
    urls = parse_urls(" ".join(args.urls))
    if not urls and not args.urls:
        try:
            raw = input("请输入0到多个链接（可直接回车跳过）: ").strip()
        except EOFError:
            raw = ""
        urls = parse_urls(raw)

    run(
        urls=urls,
        data_dir=data_dir_path,
        ner_backend=args.ner_backend,
        train_data=train_data_path,
        crf_model=crf_model_path,
        disambiguation_train_data=disambiguation_train_path,
        reuse_existing_model=args.reuse_crf_model,
        output_dir=output_dir_path,
        high_recall=args.high_recall,
        use_crf_disambiguation=not args.disable_crf_disambiguation,
    )
