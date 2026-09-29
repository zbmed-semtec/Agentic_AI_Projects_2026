"""Run the starter crew against a local LLM."""

import json
import os
import sys

from dotenv import load_dotenv

from starter.crew import StarterCrew
from starter.ollama_server import find_server, has_model, installed_models, remember_base_url


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

        result = ModelTaskAnnotatorCrew().crew().kickoff()
        print("\n----- Model-task annotation -----\n")
        print(result)
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
