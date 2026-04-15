from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

WIKI_HOST = "https://{lang}.wikipedia.org"


@dataclass
class WikiPage:
    title: str
    url: str
    text: str
    sections: List[str]
    categories: List[str]
    links: List[str]


class WikipediaCrawler:
    def __init__(self, lang: str = "zh", timeout: int = 12) -> None:
        self.lang = lang
        self.timeout = timeout
        self.base_url = WIKI_HOST.format(lang=lang)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "knowledge-graph-course-project/1.0 (educational use)"
            }
        )

    def _build_url(self, title: str) -> str:
        return f"{self.base_url}/wiki/{title.replace(' ', '_')}"

    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        links: Set[str] = set()
        content = soup.select_one("#mw-content-text")
        if not content:
            return []
        for tag in content.select("a[href^='/wiki/']"):
            href = tag.get("href", "")
            if ":" in href:
                continue
            name = href.split("/wiki/")[-1].split("#")[0]
            if name:
                links.add(unquote(name))
        return sorted(links)

    def _extract_sections(self, soup: BeautifulSoup) -> List[str]:
        sections = []
        for h in soup.select("h2 .mw-headline, h3 .mw-headline"):
            text = h.get_text(strip=True)
            if text:
                sections.append(text)
        return sections

    def _extract_categories(self, soup: BeautifulSoup) -> List[str]:
        categories = []
        cat_box = soup.select_one("#mw-normal-catlinks")
        if not cat_box:
            return categories
        for tag in cat_box.select("a[href^='/wiki/Category:']"):
            c = tag.get_text(strip=True)
            if c:
                categories.append(c)
        return categories

    def _extract_plain_text(self, soup: BeautifulSoup) -> str:
        content_root = soup.select_one("#mw-content-text .mw-parser-output")
        if not content_root:
            return ""

        blocks: List[str] = []
        keep_tags = {"p", "h2", "h3", "h4", "ul", "ol", "dl", "blockquote"}

        for child in content_root.children:
            if not isinstance(child, Tag):
                continue
            if child.name not in keep_tags:
                continue
            if child.get("class") and any(
                cls in child.get("class", [])
                for cls in ["reflist", "navbox", "metadata", "mw-empty-elt"]
            ):
                continue
            for bad in child.select("sup.reference, span.mw-editsection"):
                bad.decompose()

            if child.name in {"ul", "ol"}:
                items = [li.get_text(" ", strip=True) for li in child.select(":scope > li")]
                cleaned_items = [self._normalize_text(i) for i in items if i.strip()]
                if cleaned_items:
                    blocks.append(" ; ".join(cleaned_items))
                continue

            text = child.get_text(" ", strip=True)
            normalized = self._normalize_text(text)
            if normalized:
                blocks.append(normalized)

        return "\n".join(blocks).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"\[\d+\]", "", normalized).strip()
        return normalized

    def fetch_page(self, title: str) -> WikiPage | None:
        url = self._build_url(title)
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.select_one("#firstHeading")
        final_title = title_tag.get_text(strip=True) if title_tag else title
        page = WikiPage(
            title=final_title,
            url=url,
            text=self._extract_plain_text(soup),
            sections=self._extract_sections(soup),
            categories=self._extract_categories(soup),
            links=self._extract_links(soup),
        )
        if not page.text:
            return None
        return page

    def crawl(self, seed_title: str, max_pages: int = 20) -> Dict[str, WikiPage]:
        queue = deque([seed_title])
        visited: Set[str] = set()
        result: Dict[str, WikiPage] = {}

        while queue and len(result) < max_pages:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            page = self.fetch_page(current)
            if page is None:
                continue
            result[page.title] = page

            for nxt in page.links[:100]:
                if nxt not in visited and len(queue) < max_pages * 5:
                    queue.append(nxt)
        return result


def save_pages_to_jsonl(pages: Dict[str, WikiPage], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for page in pages.values():
            f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
