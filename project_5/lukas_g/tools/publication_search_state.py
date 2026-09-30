"""Track successful publication searches for confidence-gated task annotations."""

from __future__ import annotations

import threading


_COMPLETED_SEARCHES: dict[str, list[str]] = {}
_SEARCH_LOCK = threading.Lock()


def mark_publication_search_completed(model_id: str, query: str) -> None:
    """Record the query for a completed publication search on a model ID."""
    normalized_id = model_id.strip()
    if normalized_id:
        with _SEARCH_LOCK:
            queries = _COMPLETED_SEARCHES.setdefault(normalized_id, [])
            normalized_query = query.strip()
            if normalized_query not in queries:
                queries.append(normalized_query)


def completed_publication_search_query(model_id: str) -> str | None:
    """Return the completed search query for a model ID, if one exists."""
    with _SEARCH_LOCK:
        queries = _COMPLETED_SEARCHES.get(model_id.strip(), [])
        return queries[-1] if queries else None


def completed_publication_search_queries(model_id: str) -> tuple[str, ...]:
    """Return all completed publication queries for a model ID."""
    with _SEARCH_LOCK:
        return tuple(_COMPLETED_SEARCHES.get(model_id.strip(), ()))


def clear_publication_search_completed(model_id: str, query: str | None = None) -> None:
    """Consume one completed query, or clear all searches for a model ID."""
    normalized_id = model_id.strip()
    with _SEARCH_LOCK:
        if query is None:
            _COMPLETED_SEARCHES.pop(normalized_id, None)
            return
        queries = _COMPLETED_SEARCHES.get(normalized_id, [])
        try:
            queries.remove(query)
        except ValueError:
            return
        if not queries:
            _COMPLETED_SEARCHES.pop(normalized_id, None)
