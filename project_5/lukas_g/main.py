"""Run the starter crew against a local LLM."""

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from starter.crew import StarterCrew
from starter.ollama_server import find_server, has_model, installed_models, remember_base_url


class _TeeStream:
    """Write captured output to both its original stream and an in-memory log."""

    def __init__(self, original, capture: io.StringIO) -> None:
        self.original = original
        self.capture = capture

    def write(self, text: str) -> int:
        self.capture.write(text)
        return self.original.write(text)

    def flush(self) -> None:
        self.capture.flush()
        self.original.flush()

    def isatty(self) -> bool:
        return self.original.isatty()

    @property
    def encoding(self):
        return self.original.encoding


def save_agent_log(log_text: str, log_directory: Path | None = None) -> Path:
    """Save one agent run log to a uniquely timestamped UTF-8 file."""
    if log_directory is None:
        log_directory = Path(__file__).resolve().parent / "data" / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log_path = log_directory / f"agent_run_{timestamp}.log"
    log_path.write_text(log_text, encoding="utf-8")
    return log_path


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


def main() -> None:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if sys.argv[1:2] == ["--annotate-mltasks"]:
        if provider == "ollama":
            _check_ollama()
        else:
            print(f"Using OpenAI-compatible model: {os.getenv('OPENAI_MODEL_NAME', 'local-model')}")
        from starter.tools.model_data_tools import initialize_output_dataset

        try:
            output_path = initialize_output_dataset()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Could not initialize annotation output from input data: {exc}") from exc
        print(f"Annotation output will be written to {output_path}")
        from starter.model_task_annotator import ModelTaskAnnotatorCrew

        run_log = io.StringIO()
        try:
            with redirect_stdout(_TeeStream(sys.stdout, run_log)):
                with redirect_stderr(_TeeStream(sys.stderr, run_log)):
                    result = ModelTaskAnnotatorCrew().crew().kickoff()
                    print("\n----- Model-task annotation -----\n")
                    print(result)
        finally:
            log_path = save_agent_log(run_log.getvalue())
        print(f"Agent log saved to {log_path}")
        return

    topic = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TOPIC", "local LLM agents with CrewAI")

    if provider == "ollama":
        _check_ollama()
    else:
        print(f"Using OpenAI-compatible model: {os.getenv('OPENAI_MODEL_NAME', 'local-model')}")

    print(f"Topic: {topic}\n")
    result = StarterCrew().crew().kickoff(inputs={"topic": topic})
    print("\n----- Brief -----\n")
    print(result)


if __name__ == "__main__":
    main()
