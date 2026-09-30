"""Tools the agents can call. Fetched sources are cached in .cache/."""

import json
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Callable

import requests
from crewai.tools import tool
from huggingface_hub import hf_hub_download, model_info
from pypdf import PdfReader

CACHE = Path(__file__).parent / ".cache"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*|eprint\s*=\s*[{\"])(\d{4}\.\d{4,5})", re.I)
BIBTEX_TITLE = re.compile(r"title\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", re.I)


def _cached(path: Path, fetch: Callable[[], str]) -> str:
    """Return the cached text at path, or fetch it, save it and return it."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _filename(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    return [t.removeprefix(prefix) for t in tags if t.startswith(prefix)]


def _card_text(repo_id: str) -> str:
    def fetch() -> str:
        with open(hf_hub_download(repo_id.strip(), "README.md"), encoding="utf-8") as f:
            return f.read()

    return _cached(CACHE / "cards" / f"{_filename(repo_id)}.md", fetch)


def _metadata(repo_id: str) -> str:
    def fetch() -> str:
        info = model_info(repo_id.strip())
        card = info.card_data.to_dict() if info.card_data else {}
        config = info.config or {}
        tags = info.tags or []
        return json.dumps(
            {
                "url": f"https://huggingface.co/{repo_id.strip()}",
                "pipeline_tag": info.pipeline_tag,
                "architectures": config.get("architectures"),
                "model_type": config.get("model_type"),
                "safetensors_total_params": info.safetensors.total if info.safetensors else None,
                "base_model": card.get("base_model"),
                "base_model_relation": [t for t in _tag_values(tags, "base_model:") if ":" in t],
                "datasets": card.get("datasets"),
                "arxiv_papers": _tag_values(tags, "arxiv:"),
            }
        )

    return _cached(CACHE / "metadata" / f"{_filename(repo_id)}.json", fetch)


def _arxiv_query(**params) -> list[dict]:
    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()
    return [
        {
            "arxiv_id": e.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1].split("v")[0],
            "title": " ".join(e.find("a:title", ATOM).text.split()),
            "abstract": " ".join(e.find("a:summary", ATOM).text.split()),
        }
        for e in ET.fromstring(r.text).findall("a:entry", ATOM)
    ]


@tool("hf_metadata")
def hf_metadata(repo_id: str) -> str:
    """Hugging Face metadata of a model repo: pipeline task, architecture,
    exact parameter count, base model, training datasets and linked arXiv
    papers. Input is the repo id, e.g. 'microsoft/phi-1_5'."""
    return _metadata(repo_id)


@tool("hf_modelcard")
def hf_modelcard(repo_id: str) -> str:
    """Full text of the model card (README.md) of a Hugging Face model repo.
    Input is the repo id, e.g. 'microsoft/phi-1_5'."""
    return _card_text(repo_id)


@tool("fetch_paper_local")
def fetch_paper_local(repo_id: str) -> str:
    """Candidate papers the repo itself mentions: arXiv tags in the metadata,
    arXiv links and citations in the model card. Returns papers as
    arxiv_id + title, and citation titles that can be passed to
    fetch_paper_api. Input is the repo id, e.g. 'microsoft/phi-1_5'."""

    def fetch() -> str:
        card = _card_text(repo_id)
        ids = json.loads(_metadata(repo_id))["arxiv_papers"]
        ids += [i for i in ARXIV_ID.findall(card) if i not in ids]
        papers = _arxiv_query(id_list=",".join(ids), max_results=len(ids)) if ids else []
        titles = [" ".join(re.sub(r"\\\w+|[{}]", "", t).split()) for t in BIBTEX_TITLE.findall(card)]
        return json.dumps(
            {
                "papers": [{"arxiv_id": p["arxiv_id"], "title": p["title"]} for p in papers],
                "citation_titles": titles,
            },
            ensure_ascii=False,
        )

    return _cached(CACHE / "paper_candidates" / f"{_filename(repo_id)}.json", fetch)


@tool("fetch_paper_api")
def fetch_paper_api(query: str) -> str:
    """Search arXiv when the repo does not mention the paper. Returns the top
    hits as arxiv_id + title + abstract. Input is free text, e.g. a model
    name or paper title."""
    return json.dumps(_arxiv_query(search_query=f"all:{query}", max_results=5), ensure_ascii=False)


@tool("retrieve_paper_from_arxiv")
def retrieve_paper_from_arxiv(arxiv_id: str) -> str:
    """Full text of an arXiv paper. Input is an arXiv id, e.g. '2309.05463'."""

    def fetch() -> str:
        r = requests.get(f"https://arxiv.org/pdf/{arxiv_id.strip()}", timeout=60)
        r.raise_for_status()
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(r.content)).pages)

    return _cached(CACHE / "papers" / f"{arxiv_id.strip()}.txt", fetch)
