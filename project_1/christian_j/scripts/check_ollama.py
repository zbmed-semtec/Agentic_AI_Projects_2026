"""Check that Ollama is running and the model from .env is installed.

Usage:  uv run python -m scripts.check_ollama
Exit code 0 = all good, 1 = server or model missing.
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv


def installed_models(base_url: str) -> list[str] | None:
    """Return model names, or None when the server is not answering."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return [item.get("name", "") for item in payload.get("models", [])]


def has_model(names: list[str], wanted: str) -> bool:
    """Return True if the model is installed.

    Ollama stores "llama3" as "llama3:latest", so both names are checked.
    """
    return wanted in names or f"{wanted}:latest" in names


def check() -> None:
    """Exit with a clear message if Ollama or the model is missing."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        print(f"LLM_PROVIDER={provider}, skipping Ollama check.")
        return

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()

    names = installed_models(base_url)
    if names is None:
        sys.exit(f"Ollama is not answering at {base_url}.\nStart it with: ollama serve")
    if not has_model(names, model):
        available = ", ".join(names) or "none"
        sys.exit(
            f"Ollama is running, but '{model}' is not installed.\n"
            f"Installed models: {available}\n"
            f"Download it with: ollama pull {model}"
        )
    print(f"Using Ollama at {base_url}  model: {model}")


if __name__ == "__main__":
    check()
