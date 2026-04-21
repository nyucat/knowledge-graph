from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from bert_bilstm_crf_disambiguator import BertBiLSTMCRFEntityDisambiguator
from bert_bilstm_crf_ner import BertBiLSTMCRFEntityRecognizer, load_ner_jsonl_for_bert
from cogie_adapter import CogIEEntityRecognizer
from crf_disambiguator import CRFEntityDisambiguator
from crf_ner import CRFEntityRecognizer, ensure_label_coverage, load_ner_jsonl
from fine_grained_typing import FineGrainedEntityTyper
from text_sources import TextDocument, build_web_txt_from_urls, load_local_txt_documents, parse_urls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_coarse_label(raw_label: str, entity_text: str) -> str:
    label = (raw_label or "").upper().strip()
    text = (entity_text or "").strip()
    if re.search(r"%|百分之", text):
        return "PERCENT"
    if re.search(r"元|美元|欧元|英镑|日元|港元", text):
        return "MONEY"
    if re.search(r"\d{1,4}年|\d{1,2}月|\d{1,2}日", text):
        return "DATE"
    if re.search(r"\d{1,2}点|\d{1,2}:\d{1,2}", text):
        return "TIME"
    alias = {
        "PERSON": "PER",
        "PEOPLE": "PER",
        "HUMAN": "PER",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOC",
        "PLACE": "LOC",
        "GPE": "LOC",
        "DATETIME": "DATE",
        "DISCIPLINE": "SUBJECT",
        "PRODUCT": "PROD",
        "UNKNOWN": "ORG",
    }
    return alias.get(label, label if label else "ORG")


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


def train_or_load_crf(train_path: Path, model_path: Path, reuse_existing_model: bool) -> CRFEntityRecognizer:
    crf = CRFEntityRecognizer()
    if reuse_existing_model and model_path.exists():
        crf.load(model_path)
        return crf
    samples = load_ner_jsonl(train_path)
    ensure_label_coverage(samples)
    crf.train(samples)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    crf.save(model_path)
    return crf


def train_or_load_bert_ner(
    train_path: Path,
    model_path: Path,
    reuse_existing_model: bool,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> BertBiLSTMCRFEntityRecognizer:
    try:
        recognizer = BertBiLSTMCRFEntityRecognizer(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
        )
    except OSError as exc:
        raise RuntimeError(
            "无法加载 BERT 模型。请将 --bert-model-name 指向本地已下载模型目录（离线环境不可自动下载）。"
        ) from exc
    if reuse_existing_model and model_path.exists():
        recognizer.load(model_path)
        return recognizer
    samples = load_ner_jsonl_for_bert(train_path)
    recognizer.train(samples)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(model_path)
    return recognizer


def train_or_load_crf_disambiguator(
    disambiguation_train_path: Path | None,
    ner_fallback_path: Path,
    model_path: Path,
    reuse_existing_model: bool,
) -> tuple[CRFEntityDisambiguator, str]:
    disambiguator = CRFEntityDisambiguator()
    if reuse_existing_model and model_path.exists():
        disambiguator.load(model_path)
        return disambiguator, "cached_model"
    source = disambiguator.train_from_jsonl(
        disambiguation_train_path=disambiguation_train_path,
        ner_fallback_path=ner_fallback_path,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    disambiguator.save(model_path)
    return disambiguator, source


def train_or_load_bert_disambiguator(
    disambiguation_train_path: Path | None,
    ner_fallback_path: Path,
    model_path: Path,
    reuse_existing_model: bool,
    model_name: str,
    epochs: int,
) -> tuple[BertBiLSTMCRFEntityDisambiguator, str]:
    try:
        disambiguator = BertBiLSTMCRFEntityDisambiguator(model_name=model_name, epochs=epochs)
    except OSError as exc:
        raise RuntimeError(
            "无法加载消歧所需 BERT 模型。请将 --bert-model-name 指向本地已下载模型目录。"
        ) from exc
    if reuse_existing_model and model_path.exists():
        disambiguator.load(model_path)
        return disambiguator, "cached_model"
    source = disambiguator.train_from_jsonl(
        disambiguation_train_path=disambiguation_train_path,
        ner_fallback_path=ner_fallback_path,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    disambiguator.save(model_path)
    return disambiguator, source


def collect_documents(data_dir: Path, urls: list[str]) -> tuple[list[TextDocument], list[TextDocument]]:
    local_docs = load_local_txt_documents(data_dir)
    web_txt_path = PROJECT_ROOT / "webfile" / "web.txt"
    web_doc = build_web_txt_from_urls(urls, web_txt_path)
    web_docs = [web_doc] if web_doc is not None else []
    return local_docs, web_docs


def append_typed_record(
    entity_records: list[dict],
    seen: set[tuple],
    doc: TextDocument,
    entity_text: str,
    coarse_label: str,
    start: int,
    end: int,
    typer: FineGrainedEntityTyper,
) -> None:
    mention = (entity_text or "").strip()
    if not mention:
        return
    if len(mention) < 2 or len(mention) > 50:
        return

    typing = typer.predict(
        entity_text=mention,
        page_text=doc.text,
        start=start,
        end=end,
        coarse_label=normalize_coarse_label(coarse_label, mention),
    )
    key = (doc.source_url, mention, start, end, typing.parent_label)
    if key in seen:
        return
    seen.add(key)
    entity_records.append(
        {
            "entity_text": mention,
            "entity_label": typing.fine_label,
            "entity_parent_label": typing.parent_label,
            "typing_score": round(float(typing.score), 3),
            "start": int(start),
            "end": int(end),
            "page_url": doc.source_url,
        }
    )


def apply_disambiguation(entity_records: list[dict], docs: list[TextDocument], disambiguator) -> list[dict]:
    text_by_url = {doc.source_url: doc.text for doc in docs}
    predicted = disambiguator.predict_parent_labels(entity_records, text_by_url)
    for row, parent in zip(entity_records, predicted):
        if not parent:
            continue
        row["entity_parent_label"] = parent
        row["entity_label"] = generic_fine_label_for_parent(parent)
    return entity_records


def save_entities_json(entity_records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entity_records, f, ensure_ascii=False, indent=2)


def save_entities_csv(entity_records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def run(
    urls: list[str],
    data_dir: Path,
    ner_backend: str,
    train_data: Path,
    crf_model: Path,
    bert_ner_model: Path,
    bert_model_name: str,
    bert_ner_epochs: int,
    bert_ner_batch_size: int,
    disambiguation_train_data: Path | None,
    disambiguation_backend: str,
    bert_disambiguation_model: Path,
    reuse_existing_model: bool,
    output_dir: Path,
    high_recall: bool,
    use_disambiguation: bool,
    bert_disambiguation_epochs: int,
) -> None:
    del high_recall

    output_dir.mkdir(parents=True, exist_ok=True)
    typer = FineGrainedEntityTyper()
    local_docs, web_docs = collect_documents(data_dir=data_dir, urls=urls)
    docs = local_docs + web_docs

    if ner_backend == "crf":
        recognizer = train_or_load_crf(train_data, crf_model, reuse_existing_model)
    elif ner_backend == "bert_bilstm_crf":
        recognizer = train_or_load_bert_ner(
            train_path=train_data,
            model_path=bert_ner_model,
            reuse_existing_model=reuse_existing_model,
            model_name=bert_model_name,
            epochs=bert_ner_epochs,
            batch_size=bert_ner_batch_size,
        )
    else:
        recognizer = CogIEEntityRecognizer(language="english", corpus="trex")

    entity_records: list[dict] = []
    seen: set[tuple] = set()

    for doc in docs:
        if ner_backend in {"crf", "bert_bilstm_crf"}:
            entities = recognizer.predict_entities(doc.text)
            for e in entities:
                append_typed_record(
                    entity_records=entity_records,
                    seen=seen,
                    doc=doc,
                    entity_text=e.text,
                    coarse_label=e.label,
                    start=e.start,
                    end=e.end,
                    typer=typer,
                )
        else:
            entities = recognizer.predict(doc.text)
            for e in entities:
                text = str(e.get("text", "")).strip()
                idx = doc.text.find(text)
                start = idx if idx >= 0 else -1
                end = idx + len(text) if idx >= 0 else -1
                append_typed_record(
                    entity_records=entity_records,
                    seen=seen,
                    doc=doc,
                    entity_text=text,
                    coarse_label=str(e.get("label", "UNKNOWN")),
                    start=start,
                    end=end,
                    typer=typer,
                )

    entity_records.sort(
        key=lambda r: (
            1 if str(r.get("page_url", "")) == "local://test.txt" else 0,
            str(r.get("page_url", "")),
            int(r.get("start", -1)),
            int(r.get("end", -1)),
            str(r.get("entity_text", "")),
        )
    )

    if use_disambiguation and entity_records:
        if disambiguation_backend == "bert_bilstm_crf":
            disambiguator, disamb_source = train_or_load_bert_disambiguator(
                disambiguation_train_path=disambiguation_train_data,
                ner_fallback_path=train_data,
                model_path=bert_disambiguation_model,
                reuse_existing_model=reuse_existing_model,
                model_name=bert_model_name,
                epochs=bert_disambiguation_epochs,
            )
            disamb_method = "bert_bilstm_crf_sequence"
        else:
            disambiguator, disamb_source = train_or_load_crf_disambiguator(
                disambiguation_train_path=disambiguation_train_data,
                ner_fallback_path=train_data,
                model_path=output_dir / "crf_disambiguator.bin",
                reuse_existing_model=reuse_existing_model,
            )
            disamb_method = "bow_cosine + crf_global_lbp"
        entity_records = apply_disambiguation(entity_records, docs, disambiguator)
    else:
        disamb_source = "disabled"
        disamb_method = "disabled"

    json_path = output_dir / "entities.json"
    csv_path = output_dir / "entities.csv"
    save_entities_json(entity_records, json_path)
    save_entities_csv(entity_records, csv_path)

    result = {
        "local_documents": len(local_docs),
        "web_documents": len(web_docs),
        "total_documents": len(docs),
        "ner_backend": ner_backend,
        "disambiguation_enabled": use_disambiguation,
        "disambiguation_backend": disambiguation_backend if use_disambiguation else "disabled",
        "disambiguation_method": disamb_method,
        "disambiguation_train_source": disamb_source,
        "entities_extracted": len(entity_records),
        "entities_json_path": str(json_path),
        "entities_csv_path": str(csv_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地文本+链接文本实体抽取与实体消歧")
    parser.add_argument("--urls", nargs="*", default=[], help="0到多个网页链接")
    parser.add_argument("--data-dir", default="datafile", help="本地 txt 目录")
    parser.add_argument(
        "--ner-backend",
        choices=["crf", "cogie", "bert_bilstm_crf"],
        default="bert_bilstm_crf",
        help="实体识别后端",
    )
    parser.add_argument("--train-data", default="data/sample_ner_train.jsonl", help="NER 训练数据")
    parser.add_argument("--crf-model", default="outputs/crf_model.bin", help="CRF 模型路径")
    parser.add_argument("--bert-ner-model", default="outputs/bert_bilstm_crf_ner.pt", help="BERT NER 模型路径")
    parser.add_argument("--bert-model-name", default="models/bert-base-chinese", help="BERT 预训练模型")
    parser.add_argument("--bert-ner-epochs", type=int, default=3, help="BERT NER 训练轮数")
    parser.add_argument("--bert-ner-batch-size", type=int, default=8, help="BERT NER batch size")
    parser.add_argument(
        "--disambiguation-train-data",
        default="data/sample_disambiguation_train.jsonl",
        help="实体消歧训练数据（jsonl，含 mentions）",
    )
    parser.add_argument(
        "--disambiguation-backend",
        choices=["crf", "bert_bilstm_crf"],
        default="bert_bilstm_crf",
        help="实体消歧后端",
    )
    parser.add_argument(
        "--bert-disambiguation-model",
        default="outputs/bert_bilstm_crf_disambiguator.pt",
        help="BERT 消歧模型路径",
    )
    parser.add_argument("--bert-disambiguation-epochs", type=int, default=4, help="BERT 消歧训练轮数")
    parser.add_argument("--reuse-crf-model", action="store_true", help="优先加载已有模型")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--high-recall", action="store_true", help="保留兼容参数")
    parser.add_argument("--disable-crf-disambiguation", action="store_true", help="兼容旧参数")
    parser.add_argument("--disable-disambiguation", action="store_true", help="关闭实体消歧")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_data_path = resolve_path(args.train_data)
    disambiguation_train_path = resolve_path(args.disambiguation_train_data)
    crf_model_path = resolve_path(args.crf_model)
    bert_ner_model_path = resolve_path(args.bert_ner_model)
    bert_disambiguation_model_path = resolve_path(args.bert_disambiguation_model)
    output_dir_path = resolve_path(args.output_dir)
    data_dir_path = resolve_path(args.data_dir)

    urls = parse_urls(" ".join(args.urls))

    disable_disambiguation = bool(args.disable_disambiguation or args.disable_crf_disambiguation)
    run(
        urls=urls,
        data_dir=data_dir_path,
        ner_backend=args.ner_backend,
        train_data=train_data_path,
        crf_model=crf_model_path,
        bert_ner_model=bert_ner_model_path,
        bert_model_name=args.bert_model_name,
        bert_ner_epochs=args.bert_ner_epochs,
        bert_ner_batch_size=args.bert_ner_batch_size,
        disambiguation_train_data=disambiguation_train_path,
        disambiguation_backend=args.disambiguation_backend,
        bert_disambiguation_model=bert_disambiguation_model_path,
        reuse_existing_model=args.reuse_crf_model,
        output_dir=output_dir_path,
        high_recall=args.high_recall,
        use_disambiguation=not disable_disambiguation,
        bert_disambiguation_epochs=args.bert_disambiguation_epochs,
    )
