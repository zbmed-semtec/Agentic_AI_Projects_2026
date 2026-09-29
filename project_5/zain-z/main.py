"""Run the crew on one Hugging Face model.

    python main.py microsoft/phi-1_5
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from crew import build_crew


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    model_id = sys.argv[1] if len(sys.argv) > 1 else "microsoft/phi-1_5"
    result = build_crew().kickoff(inputs={"model_id": model_id})
    print(result)


if __name__ == "__main__":
    main()
