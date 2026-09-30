"""Agents and tasks."""

import json
import os
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Task
from pydantic import BaseModel

from tools import (
    fetch_paper_api,
    fetch_paper_local,
    hf_metadata,
    hf_modelcard,
    retrieve_paper_from_arxiv,
)

HERE = Path(__file__).parent
AGENTS = yaml.safe_load((HERE / "config" / "agents.yaml").read_text(encoding="utf-8"))
TASKS = yaml.safe_load((HERE / "config" / "tasks.yaml").read_text(encoding="utf-8"))
# property -> allowed sources, in the order they should be checked
PROPERTIES = json.loads((HERE / "data" / "properties.json").read_text(encoding="utf-8"))

SOURCE_TOOLS = {
    "metadata": [hf_metadata],
    "card": [hf_modelcard],
    "paper": [fetch_paper_local, fetch_paper_api, retrieve_paper_from_arxiv],
}


class Evidence(BaseModel):
    source: str
    url: str | None = None
    quote: str


class Finding(BaseModel):
    value: str | list[str] | None = None
    evidence: list[Evidence] = []
    confident: bool = False


class Reference(BaseModel):
    item: str
    model: str


class References(BaseModel):
    references: list[Reference] = []


def local_llm() -> LLM:
    return LLM(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY", "local"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


def build_crew(sources: list[str]) -> Crew:
    """Two tasks: extract the property from the given sources, then list the items
    of the answer that only point to another model (e.g. "phi-1's training data")."""
    agent = Agent(**AGENTS["metadata_agent"], llm=local_llm())
    extract_task = Task(
        **TASKS["extract_task"],
        agent=agent,
        tools=[t for source in sources for t in SOURCE_TOOLS[source]],
        output_pydantic=Finding,
    )
    reference_task = Task(
        **TASKS["reference_task"], agent=agent, context=[extract_task], output_pydantic=References
    )
    return Crew(agents=[agent], tasks=[extract_task, reference_task], verbose=True)


def merge(finding: Finding, item: str, resolved: Finding) -> Finding:
    """Replace a pointer item in finding.value with the value found for the model it points to."""
    as_list = lambda v: v if isinstance(v, list) else [v]
    if not resolved.value:
        return finding.model_copy(update={"confident": False})
    value = [new for old in as_list(finding.value) for new in (as_list(resolved.value) if old == item else [old])]
    return Finding(
        value=value if isinstance(finding.value, list) or len(value) > 1 else value[0],
        evidence=finding.evidence + resolved.evidence,
        confident=resolved.confident,
    )


def extract(model_id: str, prop: str, sources: list[str] | None = None, depth: int = 1) -> Finding:
    """Extract prop for model_id. Items that point to another model are resolved by
    extracting prop for that model from its paper, up to `depth` levels deep."""
    sources = sources or PROPERTIES[prop]
    result = build_crew(sources).kickoff(
        inputs={
            "model_id": model_id,
            "property": prop,
            "hint": TASKS["hints"].get(prop, "").strip(),
            "sources": " ".join(
                f"{i}. {s}: {TASKS['sources'][s].strip()}" for i, s in enumerate(sources, start=1)
            ),
        }
    )
    finding = result.tasks_output[0].pydantic or Finding()
    references = result.tasks_output[1].pydantic or References()

    for ref in references.references if depth > 0 else []:
        print(f"\nResolving '{ref.item}' via {ref.model}")
        finding = merge(finding, ref.item, extract(ref.model, prop, sources=["paper"], depth=depth - 1))
    return finding
