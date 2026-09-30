"""Tool read_card: returns the full text of one model card."""

from pathlib import Path

from crewai.tools import BaseTool

from tools.list_cards import CARDS_DIR


class ReadCardTool(BaseTool):
    """Open one model card by its filename."""

    name: str = "read_card"
    description: str = (
        "Opens one model card and returns its full text. Pass only the "
        "filename exactly as list_cards or search_cards show it "
        "(e.g. 'Microsoft__phi.md'). Argument: filename."
    )

    def _run(self, filename: str) -> str:
        """Return the card text, or a hint if the file does not exist.

        Only the filename is used, so "data/model_cards/x.md" works as well.
        A missing file returns a message instead of raising an error, so the
        model can try again with a correct name.
        """
        # If the model passes a path, keep only the filename.
        name = Path(filename.strip()).name
        path = CARDS_DIR / name

        if not path.is_file():
            return f"No card named '{name}'. Use list_cards to see the filenames."
        return path.read_text(encoding="utf-8")
