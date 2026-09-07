"""The query-embedding cache: one forward pass per distinct query, not per search.

Embedding a query is the most expensive single step in a search and a pure function of
(text, embedding version). It is also the step most often repeated — the same question
from different people, a retry, the AI path searching twice in one turn.
"""
from __future__ import annotations

import asyncio

import pytest

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

    first = _run(search_service.embed_query("chinh sach nghi phep"))
    second = _run(search_service.embed_query("chinh sach nghi phep"))

    assert first == second == [0.1, 0.2, 0.3]
    assert calls == ["chinh sach nghi phep"], "the second search must not re-embed"


def test_a_different_query_is_not_served_from_another_entry(monkeypatch):
    def fake_embed(text: str) -> list[float]:
        return [float(len(text))]

    monkeypatch.setattr(search_service, "get_bge_embedding", fake_embed)
    search_service.reset_query_embedding_cache()

    assert _run(search_service.embed_query("abc")) == [3.0]
    assert _run(search_service.embed_query("abcd")) == [4.0]


def test_a_caller_cannot_corrupt_the_entry_for_the_next_one(monkeypatch):
    """The vector is handed to callers that pass it into SQL; one list, shared, is a trap."""
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0, 2.0])
    search_service.reset_query_embedding_cache()

    first = _run(search_service.embed_query("q"))
    first.append(999.0)

    assert _run(search_service.embed_query("q")) == [1.0, 2.0]


def test_changing_the_embedding_version_invalidates_the_entry(monkeypatch):
    """A vector from another model is a point in an unrelated space, never a cache hit."""
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0])
    search_service.reset_query_embedding_cache()
    _run(search_service.embed_query("q"))

    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [2.0])
    monkeypatch.setattr(settings, "EMBEDDING_VERSION", "some-other-version-v9")

    assert _run(search_service.embed_query("q")) == [2.0]


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [1.0])
    search_service.reset_query_embedding_cache()

    for index in range(search_service._QUERY_EMBEDDING_CACHE_MAX + 25):
        _run(search_service.embed_query(f"query-{index}"))

    assert (
        len(search_service._QUERY_EMBEDDING_CACHE)
        == search_service._QUERY_EMBEDDING_CACHE_MAX
    )


def test_a_failed_embedding_is_not_cached(monkeypatch):
    """A failure must not poison the entry: the next attempt re-embeds and succeeds.

    The failure is now raised rather than returned as None. Returning None made an
    unusable embedding model look like an ordinary search to every caller, so the RAG
    answer path grounded answers in a keyword-only pool and said nothing about it. The
    caching property this test exists for is unchanged, and is what the second half
    asserts: one failed call, then a successful one, and only two forward passes.
    """
    attempts: list[int] = []

    def flaky(text: str) -> list[float]:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return [7.0]

    monkeypatch.setattr(search_service, "get_bge_embedding", flaky)
    search_service.reset_query_embedding_cache()

    with pytest.raises(search_service.VectorSearchUnavailable):
        _run(search_service.embed_query("q"))
    assert not search_service._QUERY_EMBEDDING_CACHE, "a failure must not be cached"

    assert _run(search_service.embed_query("q")) == [7.0]
    # Cached now, so a third call must not reach the model at all.
    assert _run(search_service.embed_query("q")) == [7.0]
    assert len(attempts) == 2


def test_an_empty_vector_is_a_failure_not_a_result(monkeypatch):
    """pgvector cannot use an empty vector, so silently passing one on hides the fault."""
    monkeypatch.setattr(search_service, "get_bge_embedding", lambda text: [])
    search_service.reset_query_embedding_cache()

    with pytest.raises(search_service.VectorSearchUnavailable):
        _run(search_service.embed_query("q"))
    assert not search_service._QUERY_EMBEDDING_CACHE
