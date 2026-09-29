"""Agents and tasks."""

import os
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Task

from tools import (
    fetch_paper_api,
    fetch_paper_local,
    hf_metadata,
    hf_modelcard,
    retrieve_paper_from_arxiv,
)

TOOLS = [hf_metadata, hf_modelcard, fetch_paper_local, fetch_paper_api, retrieve_paper_from_arxiv]

CONFIG = Path(__file__).parent / "config"
AGENTS = yaml.safe_load((CONFIG / "agents.yaml").read_text(encoding="utf-8"))
TASKS = yaml.safe_load((CONFIG / "tasks.yaml").read_text(encoding="utf-8"))


def local_llm() -> LLM:
    return LLM(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY", "local"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


def build_crew() -> Crew:
    agent = Agent(**AGENTS["metadata_agent"], llm=local_llm(), tools=TOOLS)
    task = Task(**TASKS["metadata_task"], agent=agent)
    return Crew(agents=[agent], tasks=[task], verbose=True)
