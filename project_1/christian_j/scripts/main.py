"""Terminal chat loop: ask a question, get an answer and follow-up questions.

Usage:  uv run python -m scripts.main   (or: make run)
"""

from scripts.check_ollama import check
from scripts.crew import StarterCrew


def main():
    """Check Ollama, build the crew once, then answer questions until empty input.

    Prints the result of each task separately: tasks_output[0] is the answer,
    tasks_output[1] are the suggested follow-up questions.
    """
    check()  # before building the crew: is Ollama running, is the model there?

    crew = StarterCrew().crew()  # build once, before the loop

    print("Card Guide – ask a question about the model cards (Enter = quit).")

    while True:
        question = input("\nQuestion: ").strip()
        if not question:  # empty input → leave the loop
            break
        result = crew.kickoff(inputs={"question": question})
        print(f"\nAnswer: {result.tasks_output[0].raw}")
        print(f"\nSuggestions: \n{result.tasks_output[1].raw}")


if __name__ == "__main__":
    main()
