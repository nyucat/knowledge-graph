from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlparse

import requests
import wikipedia
from bs4 import BeautifulSoup
from bs4.element import Tag


@dataclass
class TextDocument:
    title: str
    text: str
    source_url: str


class HtmlTextExtractor:
    def extract_from_file(self, html_path: Path) -> TextDocument | None:
        if not html_path.exists() or html_path.suffix.lower() not in {".html", ".htm"}:
            return None
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            return None
        soup = BeautifulSoup(raw, "html.parser")
        title = self._extract_title(soup, html_path.stem)
        text = self._clean_text(self._extract_main_text(soup, strip_references=False))
        if not text:
            return None
        return TextDocument(title=title, text=text, source_url=f"file://{html_path.name}")

    @staticmethod
    def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
        h1 = soup.select_one("h1")
        if h1:
            t = h1.get_text(strip=True)
            if t:
                return t
        title_tag = soup.select_one("title")
        if title_tag:
            t = title_tag.get_text(strip=True)
            if t:
                return t
        return fallback

    @staticmethod
    def _extract_main_text(soup: BeautifulSoup, strip_references: bool = True) -> str:
        candidates = [
            "#mw-content-text .mw-parser-output",
            "article",
            "main",
            "#content",
            ".content",
            "body",
        ]
        root = None
        for selector in candidates:
            root = soup.select_one(selector)
            if root:
                break
        if root is None:
            return ""

        blocks: List[str] = []
        keep_tags = {"p", "h1", "h2", "h3", "h4", "ul", "ol", "dl", "blockquote"}
        for child in root.descendants:
            if not isinstance(child, Tag):
                continue
            if child.name not in keep_tags:
                continue
            if child.get("class") and any(
                k in child.get("class", [])
                for k in ["reflist", "navbox", "metadata", "footer", "header"]
            ):
                continue
            text = child.get_text(" ", strip=True)
            norm = re.sub(r"\s+", " ", text).strip()
            if strip_references:
                norm = re.sub(r"\[\d+\]", "", norm).strip()
            if norm:
                blocks.append(norm)

        if not blocks:
            return ""
        # 去重并保持顺序
        deduped = list(dict.fromkeys(blocks))
        return "\n".join(deduped)

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = re.sub(r"\[\d+\]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

class UrlTextExtractor:
    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "knowledge-graph-course-project/1.0 (educational use)"}
        )

    @staticmethod
    def _parse_wikipedia_url(url: str) -> tuple[str, str] | None:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if "wikipedia.org" not in host:
            return None
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) < 2 or path_parts[0] != "wiki":
            return None
        title = unquote(path_parts[1]).replace("_", " ").strip()
        if not title:
            return None
        lang = host.split(".")[0] if "." in host else "en"
        if not lang:
            lang = "en"
        return lang, title

    def _fetch_with_wikipedia(self, url: str) -> TextDocument | None:
        parsed = self._parse_wikipedia_url(url)
        if parsed is None:
            return None
        lang, title = parsed
        try:
            wikipedia.set_lang(lang)
            page = wikipedia.page(title=title, auto_suggest=False, preload=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            return None
        except Exception:
            return None
        text = HtmlTextExtractor._clean_text(page.content)
        if not text:
            return None
        return TextDocument(title=page.title, text=text, source_url=url)

    def fetch(self, url: str) -> TextDocument | None:
        wiki_doc = self._fetch_with_wikipedia(url)
        if wiki_doc is not None:
            return wiki_doc
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        parsed = urlparse(url)
        fallback = parsed.netloc or url
        title = HtmlTextExtractor._extract_title(soup, fallback)
        text = HtmlTextExtractor._clean_text(
            HtmlTextExtractor._extract_main_text(soup, strip_references=False)
        )
        if not text:
            return None
        return TextDocument(title=title, text=text, source_url=url)


def load_local_txt_documents(data_dir: Path) -> List[TextDocument]:
    data_dir.mkdir(parents=True, exist_ok=True)
    docs: List[TextDocument] = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        if txt_file.name.lower() == "web.txt":
            continue
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(
            TextDocument(
                title=txt_file.stem,
                text=text,
                source_url=f"local://{txt_file.name}",
            )
        )
    return docs


def parse_urls(raw: str) -> List[str]:
    if not raw.strip():
        return []
    candidates = re.findall(r"https?://[^\s<>\"]+", raw, flags=re.IGNORECASE)
    unique: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.rstrip(").,，。;；!?！？]")
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)
    return unique


def build_web_txt_from_urls(urls: List[str], web_txt_path: Path) -> TextDocument | None:
    web_txt_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = UrlTextExtractor()
    sections: List[str] = []
    for url in urls:
        doc = extractor.fetch(url)
        if doc is None:
            continue
        sections.append(f"=== {doc.title} | {url} ===")
        sections.append(doc.text)

    content = "\n\n".join(sections).strip()
    web_txt_path.write_text(content, encoding="utf-8")
    if not content:
        return None

    return TextDocument(
        title="web",
        text=content,
        source_url=f"file://{web_txt_path.name}",
    )
