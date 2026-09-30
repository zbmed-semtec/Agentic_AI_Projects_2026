"""Tools for finding scholarly sources and extracting text from open PDFs."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from io import BytesIO
from typing import Type
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pypdf import PdfReader

from starter.tools.publication_search_state import mark_publication_search_completed


_USER_AGENT = "AgenticAIDeck/1.0 (scholarly research tool)"
_MAX_PDF_BYTES = 25 * 1024 * 1024
_MAX_REDIRECTS = 4


class SearchPublicationsInput(BaseModel):
    query: str = Field(description="Research topic or keywords to search for")
    max_results: int = Field(
        default=3, ge=1, le=5, description="Maximum results from each catalog (1-5)"
    )
    model_id: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "The exact modelId this search concerns. Include it to satisfy the MEDIUM-confidence "
            "publication-search requirement for the task or a specific model metadata property."
        ),
    )


class SearchPublicationsTool(BaseTool):
    name: str = "search_publications"
    description: str = (
        "Search Crossref and arXiv for real scholarly publications. Returns titles, "
        "authors, dates, abstracts when available, DOI/source links, and direct PDF "
        "links when available. For an annotation search, include the exact modelId and target "
        "property in the query so the write tool can require a completed, property-relevant "
        "search for MEDIUM confidence. Use the returned links as evidence; do not invent sources."
    )
    args_schema: Type[BaseModel] = SearchPublicationsInput

    def _run(self, query: str, max_results: int = 3, model_id: str | None = None) -> str:
        query = query.strip()
        if not query:
            return "Search failed: provide a non-empty query."
        model_id = model_id.strip() if model_id is not None else None
        if model_id == "":
            return "Search failed: modelId must be non-empty when provided."
        if model_id is not None and model_id.casefold() not in query.casefold():
            return "Search failed: annotation searches must include the exact modelId in the query."

        max_results = max(1, min(max_results, 5))
        sections: list[str] = []
        errors: list[str] = []

        try:
            response = requests.get(
                "https://api.crossref.org/works",
                params={
                    "query.title": query,
                    "rows": max_results,
                    "select": "DOI,title,author,published,URL,link,abstract,container-title",
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
            )
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
            records = [self._format_crossref(item) for item in items]
            if records:
                sections.append("CROSSREF RESULTS\n" + "\n\n".join(records))
            else:
                sections.append("CROSSREF RESULTS\nNo matching records found.")
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"Crossref search failed: {exc}")

        try:
            records = self._search_arxiv(f'ti:"{query}"', max_results)
            if not records:
                records = self._search_arxiv(f"all:{query}", max_results)
            if records:
                sections.append("\narXiv RESULTS\n" + "\n\n".join(records))
            else:
                sections.append("\narXiv RESULTS\nNo matching records found.")
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            errors.append(f"arXiv search failed: {exc}")

        if errors and not any("RESULTS\n" in section for section in sections):
            return "\n".join(errors)
        if errors:
            sections.append("Search notes: " + "; ".join(errors))
        if model_id is not None:
            mark_publication_search_completed(model_id, query)
            sections.append(f"Publication search completed for modelId: {model_id}")
        return "\n".join(sections)

    @classmethod
    def _search_arxiv(cls, search_query: str, max_results: int) -> list[str]:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": search_query, "start": 0, "max_results": max_results},
            headers={"User-Agent": _USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return [cls._format_arxiv(entry) for entry in root.findall("{*}entry")]

    @staticmethod
    def _format_crossref(item: dict) -> str:
        title = _clean_text((item.get("title") or ["Untitled"])[0])
        authors = ", ".join(
            " ".join(part for part in (author.get("given"), author.get("family")) if part)
            for author in item.get("author", [])[:6]
        ) or "Authors not listed"
        date_parts = item.get("published", {}).get("date-parts", [[]])[0]
        date = "-".join(str(part) for part in date_parts) if date_parts else "Date not listed"
        doi = item.get("DOI", "")
        doi_url = f"https://doi.org/{doi}" if doi else item.get("URL", "No DOI or URL")
        journal = _clean_text((item.get("container-title") or [""])[0])
        abstract = _clean_text(item.get("abstract", ""))
        pdf_links = [
            link.get("URL", "")
            for link in item.get("link", [])
            if "pdf" in link.get("content-type", "").lower() and link.get("URL")
        ]
        lines = [f"Title: {title}", f"Authors: {authors}", f"Date: {date}"]
        if journal:
            lines.append(f"Publication: {journal}")
        lines.append(f"DOI/source: {doi_url}")
        lines.append(f"PDF: {pdf_links[0] if pdf_links else 'No direct PDF link listed'}")
        if abstract:
            lines.append(f"Abstract: {abstract[:1200]}")
        return "\n".join(lines)

    @staticmethod
    def _format_arxiv(entry: ET.Element) -> str:
        atom = "{http://www.w3.org/2005/Atom}"
        title = _clean_text(entry.findtext(f"{atom}title", default="Untitled"))
        summary = _clean_text(entry.findtext(f"{atom}summary", default=""))
        authors = ", ".join(
            _clean_text(author.findtext(f"{atom}name", default=""))
            for author in entry.findall(f"{atom}author")[:6]
        ) or "Authors not listed"
        published = entry.findtext(f"{atom}published", default="")[:10] or "Date not listed"
        abs_url = entry.findtext(f"{atom}id", default="")
        pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf" if "/abs/" in abs_url else ""
        lines = [
            f"Title: {title}",
            f"Authors: {authors}",
            f"Date: {published}",
            f"Source: {abs_url or 'No arXiv URL listed'}",
            f"PDF: {pdf_url or 'No direct PDF link listed'}",
        ]
        if summary:
            lines.append(f"Abstract: {summary[:1200]}")
        return "\n".join(lines)


class ReadPublicationPDFInput(BaseModel):
    url: str = Field(description="Direct public HTTP(S) URL of a scholarly PDF")
    max_pages: int = Field(
        default=6, ge=1, le=20, description="Maximum number of leading pages to extract"
    )


class ReadPublicationPDFTool(BaseTool):
    name: str = "read_publication_pdf"
    description: str = (
        "Download and read a public scholarly PDF at a direct PDF URL. Returns extracted "
        "text from up to 20 pages (6 by default), along with document metadata. Use only "
        "PDF links returned by search_publications or another source you have verified."
    )
    args_schema: Type[BaseModel] = ReadPublicationPDFInput

    def _run(self, url: str, max_pages: int = 6) -> str:
        max_pages = max(1, min(max_pages, 20))
        current_url = url.strip()
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}

        try:
            for redirect_count in range(_MAX_REDIRECTS + 1):
                _validate_public_url(current_url)
                response = requests.get(
                    current_url,
                    headers=headers,
                    timeout=(10, 30),
                    stream=True,
                    allow_redirects=False,
                )
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    response.close()
                    if not location or redirect_count >= _MAX_REDIRECTS:
                        return "PDF read failed: too many or invalid redirects."
                    current_url = requests.compat.urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > _MAX_PDF_BYTES:
                    response.close()
                    return "PDF read failed: file exceeds the 25 MB size limit."
                data = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    data.extend(chunk)
                    if len(data) > _MAX_PDF_BYTES:
                        response.close()
                        return "PDF read failed: file exceeds the 25 MB size limit."
                content_type = response.headers.get("Content-Type", "").lower()
                response.close()
                break
            else:
                return "PDF read failed: too many redirects."
        except (requests.RequestException, ValueError, OSError) as exc:
            return f"PDF download failed: {exc}"

        if not data.startswith(b"%PDF-"):
            return f"The URL did not return a PDF (content type: {content_type or 'unknown'})."

        try:
            reader = PdfReader(BytesIO(data), strict=False)
            if reader.is_encrypted:
                return "PDF read failed: this PDF is encrypted and cannot be extracted."
            page_count = len(reader.pages)
            extracted_pages = min(page_count, max_pages)
            text_parts = []
            char_count = 0
            for page_number, page in enumerate(reader.pages[:extracted_pages], start=1):
                page_text = page.extract_text() or ""
                remaining = 12_000 - char_count
                if remaining <= 0:
                    break
                text_parts.append(f"[Page {page_number}]\n{page_text[:remaining]}")
                char_count += min(len(page_text), remaining)
            metadata = reader.metadata
            title = _clean_text(metadata.title or "") if metadata else ""
            header = [f"PDF source: {current_url}", f"Pages: {page_count}; extracted: {extracted_pages}"]
            if title:
                header.append(f"PDF title: {title}")
            text = "\n\n".join(text_parts).strip()
            if not text:
                return "\n".join(header) + "\nNo selectable text found; the PDF may be scanned."
            return "\n".join(header) + "\n\nEXTRACTED TEXT\n" + text
        except Exception as exc:
            return f"PDF extraction failed: {exc}"


def _clean_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return re.sub(r"\s+", " ", value).strip()


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise ValueError(f"Could not resolve PDF host: {exc}") from exc
    if not addresses or any(
        not ipaddress.ip_address(address.split("%", 1)[0]).is_global for address in addresses
    ):
        raise ValueError("Private or local network addresses are not allowed.")