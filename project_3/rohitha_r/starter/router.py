"""Choose which agents run for a question. The choice is fixed before the crew starts."""

from __future__ import annotations

Route = str

# Steps after the screener. Synthesis always runs last.
_STEPS: dict[str, tuple[str, ...]] = {
    "lookup": ("extraction",),
    "survey": ("extraction",),
    "evidence": ("extraction", "contradiction"),
    "disagreement": ("extraction", "contradiction"),
    "coverage": ("coverage",),
}


def route_question(question: str) -> Route:
    """Map a question to one route.

    The labels match the question types in questions.txt. An unmatched
    question takes the evidence route, which keeps the contradiction step.
    """
    text = question.lower()
    if any(word in text for word in ("disagree", "agree", "contradict")):
        return "disagreement"
    if any(
        phrase in text
        for phrase in ("beginner", "broad coverage", "which 5", "which five", "read first")
    ):
        return "coverage"
    if any(phrase in text for phrase in ("summarize", "main approaches")):
        return "survey"
    if "provide evidence" in text:
        return "evidence"
    if any(
        phrase in text
        for phrase in ("transformer", "architecture", "which papers used", "which of the papers used")
    ):
        return "lookup"
    return "evidence"


def steps_for(route: Route) -> tuple[str, ...]:
    """Return the step names between screening and the final answer."""
    return _STEPS.get(route, _STEPS["evidence"])
