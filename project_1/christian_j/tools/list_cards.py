"""Tool list_cards: a one-line overview of every model card."""

from pathlib import Path

from crewai.tools import BaseTool

CARDS_DIR = Path(__file__).resolve().parent.parent / "data" / "model_cards"


class ListCardsTool(BaseTool):
    """List all model cards with their key facts.

    `description` is written for the LLM; this docstring is for humans.
    """

    name: str = "list_cards"
    description: str = (
        "Lists all available model cards with filename, model name, "
        "license and tasks. Use this first to find relevant cards."
    )

    def _run(self) -> str:
        """Read the header of every card and return one line per card.

        Format: "filename | model name | license | tasks". License and tasks
        come from the "- License:" and "- Tasks:" lines at the top of a card.
        """
        lines = []
        for path in sorted(CARDS_DIR.glob("*.md")):
            head = path.read_text(encoding="utf-8").splitlines()[:6]

            title = head[0].removeprefix("# ").strip() if head else path.stem
            lic = "unknown"
            tasks = "unknown"
            for line in head:
                if line.startswith("- License:"):
                    lic = line.removeprefix("- License:").strip()
                elif line.startswith("- Tasks:"):
                    tasks = line.removeprefix("- Tasks:").strip()

            lines.append(f"{path.name} | {title} | {lic} | {tasks}")

        if not lines:
            return f"No model cards found in {CARDS_DIR}."
        return "\n".join(lines)
