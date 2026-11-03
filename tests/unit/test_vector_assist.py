"""A chunk the embedding ranked near the top is not dropped for sharing few words with the question.

The reranker's score is lexical overlap and the context gate reads it. "How is the watchdog counter clocked and how
long is one tick?" against "WDOG_COUNT increments every cycle of i_clk_wdt (50 ns)" scored 0.167 with the chunk at
rank 2 of the vector leg, so the model was told the Knowledge Base did not say.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from src.core.config import settings
from src.models.user import Department, User
from src.rag.reranker import (
    normalize_query,
    prepare_query_for_chunks,
    rerank_chunks_with_scores,
    retrieval_score,
)
from src.repositories.chunk import ChunkRepository

QUESTION = "How is the watchdog counter clocked and how long is one tick?"
TEXT = "WDOG_COUNT increments every cycle of i_clk_wdt (50 ns)."


def _chunk(text: str = TEXT):
    return SimpleNamespace(
        id=uuid.uuid4(),
        chunk_text=text,
        article=SimpleNamespace(title="QNSC_WDT_MAS"),
        parent_chunk=None,
    )


def _score(chunk, ranks):
    prepared = prepare_query_for_chunks(
        normalize_query(QUESTION), [chunk], vector_ranks=ranks
    )
    return retrieval_score(normalize_query(QUESTION), chunk, prepared)


def test_the_passage_alone_shares_too_few_words_to_pass_the_context_gate():
    assert _score(_chunk(), {}) < settings.RAG_MIN_CONTEXT_SCORE


def test_a_chunk_near_the_top_of_the_vector_search_reaches_the_gate():
    chunk = _chunk()
    assert _score(chunk, {chunk.id: 2}) >= settings.RAG_MIN_CONTEXT_SCORE


def test_a_chunk_further_down_is_scored_as_before():
    chunk = _chunk()
    rank = settings.RAG_VECTOR_ASSIST_TOP_N + 1
    assert _score(chunk, {chunk.id: rank}) == _score(chunk, {})


def test_it_never_lowers_a_score():
    text = "The watchdog counter is clocked every tick, one long tick each."
    chunk = _chunk(text)
    plain = _score(chunk, {})
    assert plain > settings.RAG_VECTOR_ASSIST_SCORE
    assert _score(chunk, {chunk.id: 1}) == plain


def test_zero_turns_it_off(monkeypatch):
    monkeypatch.setattr(settings, "RAG_VECTOR_ASSIST_TOP_N", 0)
    chunk = _chunk()
    assert _score(chunk, {chunk.id: 1}) == _score(chunk, {})


def test_it_reorders_the_pool_so_the_semantic_match_beats_lexical_noise():
    match = _chunk()
    noise = _chunk("Watchdog counter long tick clocked.")
    ranked = rerank_chunks_with_scores(
        normalize_query(QUESTION),
        [noise, match],
        limit=2,
        vector_ranks={match.id: 1},
    )
    assert {chunk.id for chunk, _ in ranked} == {match.id, noise.id}
    assert (
        dict((c.id, s) for c, s in ranked)[match.id] >= settings.RAG_VECTOR_ASSIST_SCORE
    )


# --- the repository records the ranks ---------------------------------------------------------------------------


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self


class _FakeDB:
    """Answers the vector statement (the one that orders by cosine distance) with the rows registered for it."""

    def __init__(self, vector_rows):
        self.vector_rows = vector_rows

    async def execute(self, statement, params=None, *args, **kwargs):
        sql = str(statement.compile(dialect=postgresql.dialect()))
        return _Rows(self.vector_rows if "<=>" in sql else [])

    async def scalar(self, statement):
        return 0


def _user() -> User:
    user = User(
        id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering"
    )
    user.departments = [Department(id=uuid.uuid4(), name="Engineering", active=True)]
    return user


def _search(repo: ChunkRepository):
    return asyncio.run(
        repo.hybrid_search(
            "watchdog tick", [0.1] * int(settings.EMBEDDING_DIMENSION or 384), _user()
        )
    )


def test_the_repository_records_each_vector_hit_with_its_rank():
    first, second = _chunk("a"), _chunk("b")
    repo = ChunkRepository(_FakeDB([first, second]))
    _search(repo)
    assert repo.vector_ranks == {first.id: 1, second.id: 2}


def test_several_queries_of_one_search_keep_the_best_rank():
    first, second = _chunk("a"), _chunk("b")
    repo = ChunkRepository(_FakeDB([first, second]))
    _search(repo)
    repo.db = _FakeDB([second])
    _search(repo)
    assert repo.vector_ranks[second.id] == 1


def test_a_search_without_a_vector_leg_records_nothing():
    repo = ChunkRepository(_FakeDB([_chunk()]))
    asyncio.run(repo.hybrid_search("watchdog tick", None, _user()))
    assert repo.vector_ranks == {}
