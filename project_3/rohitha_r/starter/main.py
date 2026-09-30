"""Run the literature triage crew against a local LLM."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from starter.crew import StarterCrew
from starter.ollama_server import find_server, has_model, installed_models, remember_base_url
from starter.router import route_question, steps_for

ROOT = Path(__file__).resolve().parent.parent
ABSTRACTS_DIR = ROOT / "papers"
FULLTEXT_DIR = ROOT / "fulltext"
DEFAULT_QUESTION = "Which of these papers used a transformer architecture?"


def _check_ollama() -> None:
    found = find_server()
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
    if found is None:
        raise SystemExit(
            "Ollama is not running.\n"
            "From the project folder, with the virtual environment active, run:\n"
            "  python -m starter.setup\n"
            "That starts the server and downloads the model. "
            "It uses the server that is actually running, so a stale "
            "OLLAMA_HOST in your shell does not send the command to the wrong port."
        )
    remember_base_url(found)
    names = installed_models(found) or []
    if names and not has_model(names, model):
        available = ", ".join(names)
        raise SystemExit(
            f"Ollama is running at {found}, but '{model}' is not installed.\n"
            f"Installed models: {available}.\n"
            "Run: python -m starter.setup"
        )
    print(f"Using Ollama at {found}  model: {model}")


def load_papers(folder: Path) -> str:
    """Read one Markdown file per paper and join them for the crew."""
    files = sorted(folder.glob("*.md"))
    if not files:
        raise SystemExit(
            f"No markdown files in {folder}.\n"
            "Add one Markdown file per paper, then run this command again."
        )
    chunks = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        chunks.append(f"### {path.stem}\n{text}")
    return "\n\n".join(chunks)


def main() -> None:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    question = sys.argv[1] if len(sys.argv) > 1 else os.getenv("QUESTION", DEFAULT_QUESTION)
    abstracts = load_papers(ABSTRACTS_DIR)
    fulltext = load_papers(FULLTEXT_DIR)

    if provider == "ollama":
        _check_ollama()
    else:
        print(f"Using OpenAI-compatible model: {os.getenv('OPENAI_MODEL_NAME', 'local-model')}")

    route = route_question(question)
    print(f"Question: {question}")
    print(f"Route: {route} ({', '.join(('relevance', *steps_for(route), 'synthesis'))})")
    print(f"Abstracts: {len(list(ABSTRACTS_DIR.glob('*.md')))} files in {ABSTRACTS_DIR.name}/")
    print(f"Full text: {len(list(FULLTEXT_DIR.glob('*.md')))} files in {FULLTEXT_DIR.name}/\n")
    crew = StarterCrew()
    crew.route = route
    result = crew.crew().kickoff(
        inputs={
            "question": question,
            "abstracts": abstracts,
            "fulltext": fulltext,
            "route": route,
        }
    )
    print("\n----- Answer -----\n")
    print(result)


if __name__ == "__main__":
    main()
