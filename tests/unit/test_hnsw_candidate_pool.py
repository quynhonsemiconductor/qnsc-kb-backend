"""The HNSW candidate list must be widened before the vector query runs.

`hnsw.ef_search` caps how many candidates a single index pass yields and defaults to 40 —
BELOW the 48 this application asks for. pgvector also applies filtering *after* the index
scan, so the department-audience, published-status and embedding_version predicates consume
candidates the index already committed to:

    "If a condition matches 10% of rows, with HNSW and the default hnsw.ef_search of 40,
     only 4 rows will match on average."
    -- https://github.com/pgvector/pgvector#filtering

MEASURED on pgvector 0.8.6 (20k rows, 10% passing the filter, partial cosine HNSW index
matching ix_article_chunks_embedding_hnsw, LIMIT 48):

    default ef_search=40            -> Index Scan returned 23 rows (asked for 48)
    ef_search=200 + iterative_scan  -> 48 rows

Every permission-filtered search was short of candidates, silently: no error, no log line,
just fewer passages reaching the reranker. These tests pin the four properties that fix
relies on, because none of them are visible in the query text itself.
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy.dialects import postgresql

from src.core.config import settings
from src.models.user import Department, User
from src.repositories.chunk import ChunkRepository


class _Result:
    def scalars(self):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _RecordingDB:
    """Records each statement with its bind parameters, in issue order."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(dialect=postgresql.dialect()))
        except Exception:
            rendered = str(statement)
        self.calls.append((rendered, params or {}))
        return _Result()

    async def scalar(self, statement):
        return 0


def _user() -> User:
    user = User(
        id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering"
    )
    user.departments = [Department(id=uuid.uuid4(), name="Engineering", active=True)]
    return user


def _search(embedding: list[float] | None) -> list[tuple[str, dict]]:
    db = _RecordingDB()
    repo = ChunkRepository(db)
    asyncio.run(
        repo.hybrid_search(
            user=_user(),
            query="clock tree synthesis",
            query_embedding=embedding,
            limit=5,
            filters={"company_domain": "acme.test"},
        )
    )
    return db.calls


def _vector_embedding() -> list[float]:
    return [0.1] * int(settings.EMBEDDING_DIMENSION or 384)


def _matching(calls: list[tuple[str, dict]], needle: str) -> list[tuple[str, dict]]:
    return [call for call in calls if needle in call[0]]


def test_ef_search_is_widened_before_the_vector_query_is_issued():
    """Order is the whole point: setting it after the scan would change nothing."""
    calls = _search(_vector_embedding())

    ef_at = [i for i, (sql, _) in enumerate(calls) if "hnsw.ef_search" in sql]
    vector_at = [i for i, (sql, _) in enumerate(calls) if "<=>" in sql]

    assert ef_at, "hnsw.ef_search was never set, so the candidate pool cannot arrive"
    assert vector_at, "no vector query was issued"
    assert min(ef_at) < min(vector_at)


def test_ef_search_is_never_below_the_candidate_pool_we_ask_for():
    """A single HNSW pass cannot return more candidates than ef_search."""
    assert settings.HNSW_EF_SEARCH >= settings.RAG_CANDIDATE_POOL_SIZE

    _sql, params = _matching(_search(_vector_embedding()), "hnsw.ef_search")[0]

    assert int(params["ef_search"]) == settings.HNSW_EF_SEARCH


def test_ef_search_is_transaction_local_so_it_cannot_leak_between_requests():
    """Session scope would ride a pooled connection into an unrelated request.

    `true` is the third argument to set_config, exactly as the tenant context in
    api/deps.py does it and for the same reason.
    """
    sql, _params = _matching(_search(_vector_embedding()), "hnsw.ef_search")[0]

    assert "set_config" in sql
    assert "true" in sql, "without the third argument this would outlive the request"


def test_the_ef_search_value_is_bound_not_interpolated_into_sql():
    """String-formatting a setting into SQL is how an injection lands in a query later."""
    sql, params = _matching(_search(_vector_embedding()), "hnsw.ef_search")[0]

    assert str(settings.HNSW_EF_SEARCH) not in sql
    assert params.get("ef_search") == str(settings.HNSW_EF_SEARCH)


def test_iterative_scan_is_requested_so_post_scan_filters_cannot_exhaust_the_pool():
    """Without it the index stops at the first ef_search candidates, filtered or not."""
    matches = _matching(_search(_vector_embedding()), "iterative_scan")

    assert matches, "iterative_scan was not requested"
    sql, params = matches[0]
    assert "set_config" in sql and "true" in sql
    assert params.get("mode") == "relaxed_order"


def test_no_hnsw_tuning_is_issued_when_there_is_no_vector_query():
    """Keyword-only search touches no HNSW index, so the GUCs would be pure overhead."""
    calls = _search(None)

    assert not _matching(calls, "hnsw.ef_search")
    assert not _matching(calls, "iterative_scan")
