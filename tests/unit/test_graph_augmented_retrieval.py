"""Graph-augmented retrieval: a third RRF leg, off by default, never allowed to break
retrieval if it fails or the graph is unpopulated.

Mirrors test_hnsw_candidate_pool.py's approach: a fake DB records/answers compiled SQL
rather than needing a live Postgres, since hybrid_search's SQL (to_tsvector,
immutable_unaccent, hnsw GUCs) is Postgres-specific and cannot run against sqlite.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from src.core.config import settings
from src.models.user import Department, User
from src.repositories.chunk import ChunkRepository


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self


class _CannedDB:
    """Answers a query with whichever canned rows are registered for a distinguishing
    substring of its compiled SQL (e.g. "FROM graph_entities"), empty otherwise. Also
    records every call so a test can assert what was (or was not) queried.
    """

    def __init__(self, canned: dict[str, list] | None = None):
        self._canned = canned or {}
        self.calls: list[str] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(dialect=postgresql.dialect()))
        except Exception:
            rendered = str(statement)
        self.calls.append(rendered)
        for needle, rows in self._canned.items():
            if needle in rendered:
                return _Rows(rows)
        return _Rows([])

    async def scalar(self, statement):
        return 0


def _user() -> User:
    user = User(id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering")
    user.departments = [Department(id=uuid.uuid4(), name="Engineering", active=True)]
    return user


def _vector_embedding() -> list[float]:
    return [0.1] * int(settings.EMBEDDING_DIMENSION or 384)


# --- _matched_graph_entity_ids: substring match + one-hop expansion -------------------


def test_matches_entities_named_in_the_query():
    entity_id = uuid.uuid4()
    other_id = uuid.uuid4()
    db = _CannedDB({
        "FROM graph_entities": [(entity_id, "clock tree synthesis"), (other_id, "unrelated topic")],
        "FROM graph_relationships": [],
    })
    repo = ChunkRepository(db)

    result = asyncio.run(repo._matched_graph_entity_ids("acme.test", "how does clock tree synthesis work"))

    assert result == [entity_id]


def test_no_matching_entity_returns_empty_without_querying_relationships():
    db = _CannedDB({"FROM graph_entities": [(uuid.uuid4(), "unrelated topic")]})
    repo = ChunkRepository(db)

    result = asyncio.run(repo._matched_graph_entity_ids("acme.test", "clock tree synthesis"))

    assert result == []
    assert not any("FROM graph_relationships" in call for call in db.calls)


def test_expands_one_hop_via_relationships():
    matched_id = uuid.uuid4()
    neighbor_id = uuid.uuid4()
    db = _CannedDB({
        "FROM graph_entities": [(matched_id, "clock tree synthesis")],
        "FROM graph_relationships": [(matched_id, neighbor_id)],
    })
    repo = ChunkRepository(db)

    result = asyncio.run(repo._matched_graph_entity_ids("acme.test", "clock tree synthesis"))

    assert set(result) == {matched_id, neighbor_id}


# --- _graph_augmented_candidates: article-level leg, coarse but safe ------------------


def test_returns_empty_without_a_company_domain():
    db = _CannedDB()
    repo = ChunkRepository(db)

    result = asyncio.run(repo._graph_augmented_candidates(None, "clock tree synthesis", [], 5))

    assert result == []
    assert not db.calls, "must not query anything without a tenant to scope by"


def test_returns_empty_when_nothing_matches():
    db = _CannedDB({"FROM graph_entities": []})
    repo = ChunkRepository(db)

    result = asyncio.run(repo._graph_augmented_candidates("acme.test", "clock tree synthesis", [], 5))

    assert result == []


# --- hybrid_search wiring: off by default, never breaks the search on failure ---------


def _search_with_graph(monkeypatch, *, enabled: bool, side_effect=None) -> _CannedDB:
    monkeypatch.setattr(settings, "GRAPH_RETRIEVAL_ENABLED", enabled)
    db = _CannedDB()
    repo = ChunkRepository(db)
    if side_effect is not None:
        async def _raise(*_args, **_kwargs):
            raise side_effect
        monkeypatch.setattr(repo, "_graph_augmented_candidates", _raise)
    asyncio.run(
        repo.hybrid_search(
            user=_user(),
            query="clock tree synthesis",
            query_embedding=_vector_embedding(),
            limit=5,
            filters={"company_domain": "acme.test"},
        )
    )
    return db


def test_graph_leg_is_untouched_when_the_feature_is_disabled(monkeypatch):
    db = _search_with_graph(monkeypatch, enabled=False)
    assert not any("graph_entities" in call for call in db.calls)


def test_a_graph_leg_failure_does_not_break_the_search(monkeypatch):
    # Must not raise: a third leg that breaks must degrade to "found nothing", not take
    # the whole search down with it.
    _search_with_graph(monkeypatch, enabled=True, side_effect=RuntimeError("graph is down"))
