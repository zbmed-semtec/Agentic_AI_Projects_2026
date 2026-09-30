"""Download each paper and save it as markdown.

Reads papers/*.md, fetches arXiv HTML, PMC XML, ACL Anthology PDFs, or an
author-posted PDF, and writes papers/fulltext/{stem}.md. Publisher pages
that require a subscription are not requested.
"""

from __future__ import annotations

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "papers" / "fulltext"
UA = "agentic-ai-deck/fulltext (local literature set)"

# Open copies only. arXiv ids are preprints of the same papers when the
# venue page is not openly downloadable.
SOURCES: dict[str, tuple[str, str]] = {
    "01_biobert": ("arxiv", "1901.08746"),
    "02_scibert": ("acl", "D19-1371"),
    "03_clinicalbert": ("arxiv", "1904.05342"),
    "04_pubmedbert": ("arxiv", "2007.15779"),
    "05_biomegatron": ("acl", "2020.emnlp-main.379"),
    "06_biom-transformers": ("acl", "2021.bionlp-1.24"),
    "07_sapbert": ("acl", "2021.naacl-main.334"),
    "08_linkbert": ("acl", "2022.acl-long.551"),
    "09_gatortron": ("pmc", "9792464"),
    "10_keblm": ("pdf", "https://blender.cs.illinois.edu/paper/biomedicallm2023.pdf"),
    "11_biogpt": ("arxiv", "2210.10341"),
    "12_scifive": ("arxiv", "2106.03598"),
    "13_pmc-llama": ("arxiv", "2304.14454"),
    "14_galactica": ("arxiv", "2211.09085"),
    "15_blue-benchmark": ("acl", "W19-5006"),
}

_SKIP_TAGS = {"script", "style", "svg", "noscript", "button", "nav", "footer", "annotation"}
_SKIP_CLASS = ("ltx_page_navbar", "ltx_pagination", "ltx_note_outer", "ltx_picture")
_HEADING = {
    "ltx_title_document": 1,
    "ltx_title_section": 2,
    "ltx_title_subsection": 3,
    "ltx_title_subsubsection": 4,
    "ltx_title_bibliography": 2,
}


class _Ar5iv(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._article = 0
        self._buf: list[str] = []
        self._blocks: list[str] = []
        self._heading = 0
        self._cell = False

    def _flush(self, prefix: str = "", suffix: str = "") -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self._blocks.append(f"{prefix}{text}{suffix}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = " ".join(v for k, v in attrs if k == "class" and v)
        if tag == "article":
            self._article += 1
        if self._article == 0:
            return
        if self._skip:
            self._skip += 1
            return
        if tag in _SKIP_TAGS or any(part in cls for part in _SKIP_CLASS):
            self._skip = 1
            return
        if tag == "br":
            self._buf.append(" ")
            return
        for name, level in _HEADING.items():
            if name in cls:
                self._flush()
                self._heading = level
                return
        if tag in {"h1", "h2", "h3", "h4"} and self._heading == 0:
            self._flush()
            self._heading = int(tag[1])
        if tag == "li":
            self._flush()
            self._buf.append("- ")
        if tag in {"td", "th"}:
            self._cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._article:
            self._flush()
            self._article -= 1
            return
        if self._article == 0:
            return
        if self._skip:
            self._skip -= 1
            return
        if self._heading and tag in {"h1", "h2", "h3", "h4", "span"}:
            # Titles are h1/h2/h3. Flush when the heading tag closes.
            if tag.startswith("h"):
                text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
                self._buf = []
                if text:
                    self._blocks.append(f"{'#' * self._heading} {text}")
                self._heading = 0
            return
        if tag == "p":
            self._flush()
        elif tag == "li":
            self._flush()
        elif tag == "tr":
            self._flush()
        elif tag in {"td", "th"}:
            self._buf.append(" | ")
            self._cell = False

    def handle_data(self, data: str) -> None:
        if self._article and not self._skip:
            self._buf.append(data)

    def markdown(self) -> str:
        self._flush()
        text = "\n\n".join(block for block in self._blocks if block.strip())
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip() + "\n"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def _strip_embedded(html: str) -> str:
    """Drop SVG and scripts so a figure cannot swallow the rest of the article."""
    html = re.sub(r"<svg\b[^>]*>.*?</svg>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.I | re.S)
    return html


def _arxiv_markdown(arxiv_id: str) -> tuple[str, str]:
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    parser = _Ar5iv()
    parser.feed(_strip_embedded(_get(url).decode("utf-8", "replace")))
    body = parser.markdown()
    if len(body) < 2000:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        body = _pdf_markdown(_get(pdf_url))
        return pdf_url, body
    return url, body


def _pdf_markdown(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n\n".join(pages)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _acl_markdown(anthology_id: str) -> tuple[str, str]:
    url = f"https://aclanthology.org/{anthology_id}.pdf"
    return url, _pdf_markdown(_get(url))


def _pdf_url_markdown(url: str) -> tuple[str, str]:
    return url, _pdf_markdown(_get(url))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text(node: ET.Element) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        if _local(child.tag) in {"xref", "graphic"}:
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_xml_text(child))
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _pmc_section(sec: ET.Element, level: int) -> list[str]:
    lines: list[str] = []
    for child in sec:
        name = _local(child.tag)
        if name == "title":
            title = _xml_text(child)
            if title:
                lines.append(f"{'#' * level} {title}")
                lines.append("")
        elif name == "p":
            paragraph = _xml_text(child)
            if paragraph:
                lines.append(paragraph)
                lines.append("")
        elif name == "sec":
            lines.extend(_pmc_section(child, min(level + 1, 4)))
        elif name in {"list", "fig", "table-wrap"}:
            paragraph = _xml_text(child)
            if paragraph:
                lines.append(paragraph)
                lines.append("")
    return lines


def _pmc_markdown(pmc_id: str) -> tuple[str, str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={pmc_id}&rettype=xml&retmode=xml&tool=agentic-ai-deck"
    )
    root = ET.fromstring(_get(url))
    lines: list[str] = []
    title = root.find(".//{*}article-title")
    if title is not None:
        lines.append(f"# {_xml_text(title)}")
        lines.append("")
    abstract = root.find(".//{*}abstract")
    if abstract is not None:
        lines.append("## Abstract")
        lines.append("")
        for paragraph in abstract.findall(".//{*}p"):
            text = _xml_text(paragraph)
            if text:
                lines.append(text)
                lines.append("")
    body = root.find(".//{*}body")
    if body is not None:
        for child in body:
            if _local(child.tag) == "sec":
                lines.extend(_pmc_section(child, 2))
            elif _local(child.tag) == "p":
                text = _xml_text(child)
                if text:
                    lines.append(text)
                    lines.append("")
    return url, "\n".join(lines).strip() + "\n"


def fetch_one(stem: str) -> str:
    kind, value = SOURCES[stem]
    if kind == "arxiv":
        url, body = _arxiv_markdown(value)
    elif kind == "acl":
        url, body = _acl_markdown(value)
    elif kind == "pmc":
        url, body = _pmc_markdown(value)
    elif kind == "pdf":
        url, body = _pdf_url_markdown(value)
    else:
        raise ValueError(kind)
    if len(body) < 1500:
        raise RuntimeError(f"{stem} text is too short ({len(body)} chars) from {url}")
    header = f"Fetched from: {url}\n\n"
    path = OUT_DIR / f"{stem}.md"
    path.write_text(header + body, encoding="utf-8")
    return f"{path.name}: {len(body):,} chars from {url}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [stem for stem in SOURCES if not (ROOT / "papers" / f"{stem}.md").exists()]
    if missing:
        raise SystemExit(f"Missing abstract files: {', '.join(missing)}")
    for index, stem in enumerate(SOURCES):
        if index:
            time.sleep(1)
        print(fetch_one(stem), flush=True)


if __name__ == "__main__":
    main()
