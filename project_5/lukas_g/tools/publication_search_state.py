"""Track successful publication searches for confidence-gated task annotations."""

from __future__ import annotations

import threading


_COMPLETED_SEARCHES: dict[str, str] = {}
_SEARCH_LOCK = threading.Lock()


def mark_publication_search_completed(model_id: str, query: str) -> None:
    """Record the query for a completed publication search on a model ID."""
    normalized_id = model_id.strip()
    if normalized_id:
        with _SEARCH_LOCK:
            _COMPLETED_SEARCHES[normalized_id] = query.strip()


def completed_publication_search_query(model_id: str) -> str | None:
    """Return the completed search query for a model ID, if one exists."""
    with _SEARCH_LOCK:
        return _COMPLETED_SEARCHES.get(model_id.strip())


def clear_publication_search_completed(model_id: str) -> None:
    """Consume a completed search after its annotation has been saved."""
    with _SEARCH_LOCK:
        _COMPLETED_SEARCHES.pop(model_id.strip(), None)
