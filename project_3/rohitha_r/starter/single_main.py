"""Run the one-agent literature triage crew against a local LLM."""

import os
import sys

from dotenv import load_dotenv

from starter.main import (
    ABSTRACTS_DIR,
    DEFAULT_QUESTION,
    FULLTEXT_DIR,
    _check_ollama,
    load_papers,
)
from starter.single_crew import SingleAgentCrew


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

    print(f"Question: {question}")
    print(f"Abstracts: {len(list(ABSTRACTS_DIR.glob('*.md')))} files in {ABSTRACTS_DIR.name}/")
    print(f"Full text: {len(list(FULLTEXT_DIR.glob('*.md')))} files in {FULLTEXT_DIR.name}/")
    print("Crew: one agent, four tasks\n")
    result = SingleAgentCrew().crew().kickoff(
        inputs={"question": question, "abstracts": abstracts, "fulltext": fulltext}
    )
    print("\n----- Answer -----\n")
    print(result)


if __name__ == "__main__":
    main()
