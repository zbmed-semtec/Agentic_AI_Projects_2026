"""Tool search_cards: keyword search in the full text of all model cards."""

import re

from crewai.tools import BaseTool

from tools.list_cards import CARDS_DIR

MAX_HITS_PER_CARD = 2
MAX_LINE_LEN = 150
CHARS_BEFORE_HIT = 60


def snippet(line: str, hit_pos: int) -> str:
    """Cut a long line so that the search word is always visible."""
    if len(line) <= MAX_LINE_LEN:
        return line
    start = max(0, hit_pos - CHARS_BEFORE_HIT)
    end = start + MAX_LINE_LEN
    text = line[start:end]
    if start > 0:
        text = "..." + text
    if end < len(line):
        text = text + "..."
    return text


class SearchCardsTool(BaseTool):
    """Search all cards for one keyword."""

    name: str = "search_cards"
    description: str = (
        "Searches the full text of all model cards for ONE keyword "
        "(e.g. 'bias', 'translation', 'bert-base-uncased'). Returns the "
        "filename and the matching lines, and a last line 'Found in N cards: ...' "
        "listing every matching card. Use a single word, not a sentence. "
        "Argument: query (the keyword)."
    )

    def _run(self, query: str) -> str:
        """Return matching lines per card and a summary of all matching cards.

        - Matches at word start and ignores case ("limitation" also finds
          "Limitations").
        - At most MAX_HITS_PER_CARD lines per card, each shortened around
          the match (see snippet).
        - Last line: "Found in N cards: ..." with each matching card once.
        """
        needle = query.strip()
        if not needle:
            return "Please give a keyword to search for."

        # Match at word start only: "mit" finds "MIT" but not "limitations".
        pattern = re.compile(r"\b" + re.escape(needle), re.IGNORECASE)

        results = []
        cards_with_hits = []
        for path in sorted(CARDS_DIR.glob("*.md")):
            hits = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                match = pattern.search(line)
                if match:
                    hits.append(snippet(line, match.start()))
                    if len(hits) == MAX_HITS_PER_CARD:
                        break

            if hits:
                cards_with_hits.append(path.name)
            for hit in hits:
                results.append(f"{path.name} | {hit}")

        if not results:
            return f"No card mentions '{query}'."

        names = ", ".join(cards_with_hits)
        summary = f"Found in {len(cards_with_hits)} cards: {names}"
        return "\n".join(results) + "\n\n" + summary
