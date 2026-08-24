"""The query-embedding cache: one forward pass per distinct query, not per search.

Embedding a query is the most expensive single step in a search and a pure function of
(text, embedding version). It is also the step most often repeated — the same question
from different people, a retry, the AI path searching twice in one turn.
"""
from __future__ import annotations

import asyncio

from src.core.config import settings
from src.domain import search_service


def _run(coro):
    return asyncio.run(coro)


def test_a_repeated_query_is_embedded_once(monkeypatch):
    calls: list[str] = []

    def fake_embed(text: str) -> list[float]:
        calls.append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(search_service, "get_bge_embedding", fake_embed)
    search_service.reset_query_embedding_cache()

    first = _run(search_service.get_text_embedding("chinh sach nghi phep"))
    second = _run(search_service.get_text_embedding("chinh sach nghi phep"))

    assert first == second == [0.1, 0.2, 0.3]
    assert calls == ["chinh sach nghi phep"], "the second search must not re-embed"


def test_a_different_query_is_not_served_from_another_entry(monkeypatch):
    def fake_embed(text: str) -> list[float]:
        return [float(len(text))]

    monkeypatch.setattr(search_service, "get_bge_embedding", fake_embed)
    search_service.reset_query_embedding_cache()

    assert _run(search_service.get_text_embedding("abc")) == [3.0]
    assert _run(search_service.get_text_embedding("abcd")) == [4.0]


def test_a_caller_cannot_corrupt_the_entry_for_the_next_one(monkeypatch):
    """The vector is handed to callers that pass it into SQL; one list, shared, is a trap."""
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0, 2.0])
    search_service.reset_query_embedding_cache()

    first = _run(search_service.get_text_embedding("q"))
    first.append(999.0)

    assert _run(search_service.get_text_embedding("q")) == [1.0, 2.0]


def test_changing_the_embedding_version_invalidates_the_entry(monkeypatch):
    """A vector from another model is a point in an unrelated space, never a cache hit."""
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0])
    search_service.reset_query_embedding_cache()
    _run(search_service.get_text_embedding("q"))

    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [2.0])
    monkeypatch.setattr(settings, "EMBEDDING_VERSION", "some-other-version-v9")

    assert _run(search_service.get_text_embedding("q")) == [2.0]


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0])
    search_service.reset_query_embedding_cache()

    for index in range(search_service._QUERY_EMBEDDING_CACHE_MAX + 25):
        _run(search_service.get_text_embedding(f"query-{index}"))

    assert (
        len(search_service._QUERY_EMBEDDING_CACHE)
        == search_service._QUERY_EMBEDDING_CACHE_MAX
    )


def test_a_failed_embedding_is_not_cached(monkeypatch):
    """Keyword search continues without a vector; the next attempt must retry."""
    attempts: list[int] = []

    def flaky(text: str) -> list[float]:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return [7.0]

    monkeypatch.setattr(search_service, "get_bge_embedding", flaky)
    search_service.reset_query_embedding_cache()

    assert _run(search_service.get_text_embedding("q")) is None
    assert _run(search_service.get_text_embedding("q")) == [7.0]
