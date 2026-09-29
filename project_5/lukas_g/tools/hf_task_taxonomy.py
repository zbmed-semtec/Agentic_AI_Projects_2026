"""Fetch and validate task IDs from the Hugging Face Hub taxonomy."""

from __future__ import annotations

from functools import lru_cache
from typing import Type

import requests
from crewai.tools import BaseTool

_HF_TASKS_URL = "https://huggingface.co/api/tasks"
_USER_AGENT = "AgenticAIDeck/1.0 (Hugging Face task taxonomy)"


class HuggingFaceTaskCatalogError(RuntimeError):
    """Raised when the official Hugging Face task catalog is unavailable or invalid."""


@lru_cache(maxsize=1)
def huggingface_tasks() -> dict[str, str]:
    """Return current Hugging Face task IDs mapped to their display labels."""
    try:
        response = requests.get(
            _HF_TASKS_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HuggingFaceTaskCatalogError(f"Could not load Hugging Face tasks: {exc}") from exc

    if not isinstance(payload, dict) or not payload:
        raise HuggingFaceTaskCatalogError("Hugging Face returned an empty or invalid task catalog.")

    tasks = {
        task_id: details.get("label", task_id)
        for task_id, details in payload.items()
        if isinstance(task_id, str) and task_id and isinstance(details, dict)
    }
    if len(tasks) != len(payload):
        raise HuggingFaceTaskCatalogError("Hugging Face returned malformed task entries.")
    return tasks


class ListHuggingFaceTasksTool(BaseTool):
    name: str = "list_huggingface_tasks"
    description: str = (
        "List the current official Hugging Face task IDs and display labels. Use exactly one "
        "listed task ID as the mlTask value; do not write a description or multiple tasks."
    )

    def _run(self) -> str:
        try:
            tasks = huggingface_tasks()
        except HuggingFaceTaskCatalogError as exc:
            return str(exc)
        return "Valid mlTask IDs (use the ID, not the label):\n" + "\n".join(
            f"{task_id}: {label}" for task_id, label in sorted(tasks.items())
        )
