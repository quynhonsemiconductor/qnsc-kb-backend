"""Recording that a search found nothing must never be why nothing is returned.

A RAG answer stream returned a 500. The cause was 26 characters:

    asyncpg.exceptions.StringDataRightTruncationError:
        value too long for type character varying(255)
    [SQL: INSERT INTO gaps (company_domain, query, ...)]

`Gap.query` is VARCHAR(255) and `log_gap` wrote the query whole. The query in question
was 281 characters, so the INSERT raised -- and because the gap is recorded from inside
`SearchService.search`, the exception propagated out of search, out of `AiService.ask`,
and reached the reader as a crashed stream instead of "no results".

Two things were wrong, and both are fixed here. The value was not bounded to the column
it goes into, and bookkeeping about an empty result was allowed to destroy the request
that produced it.

The rollback matters as much as the swallow: `log_gap` commits, so a failed write leaves
the session in a failed transaction. Catching the error without resetting the session
just moves the 500 to the next statement in the same request.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from src.models.governance import Gap
from src.repositories.governance import GovernanceRepository

COLUMN_CHARS = Gap.__table__.c.query.type.length

#: The shape that actually broke it: an assistant-composed retrieval query.
REPORTED_QUERY = (
    'I am reviewing the pending document "Day011_QuachHuynhHuuTai" in department '
    '"public2". I need help identifying who is allowed to change the source article '
    "and how to request the modification. Explain which role or person owns this "
    "responsibility and help me prepare a concise request. Draft ID: "
    "8700aa21-3b84-4426-9e52-06fd0e3b6ec2."
)


class _Result:
    def __init__(self, gap_id):
        self._gap_id = gap_id

    def scalar_one(self):
        return self._gap_id


class _DB:
    """Records the single upsert statement and the row it would write.

    `log_gap` is now one INSERT ... ON CONFLICT DO UPDATE rather than a select followed by
    an increment or an insert, so what these tests inspect is the values bound to that
    statement -- there is no ORM object handed to `add` any more.
    """

    def __init__(self):
        self.statements: list = []
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0
        self._gap_id = uuid.uuid4()

    async def execute(self, statement):
        self.statements.append(statement)
        # `compile` resolves the bound values the upsert would insert. Reading them from
        # the statement is the only way left to assert on the row.
        compiled = statement.compile(dialect=postgresql.dialect())
        self.added.append(SimpleNamespace(**compiled.params))
        return _Result(self._gap_id)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def get(self, _model, gap_id):
        return Gap(id=gap_id)

    async def refresh(self, _obj):
        return None

    async def rollback(self):
        self.rollbacks += 1


class _User:
    company_domain = "qnsc.vn"
    dept = None


def test_the_reported_query_is_longer_than_the_column():
    """Guards the premise: if this were false the test below would prove nothing."""
    assert len(REPORTED_QUERY) > COLUMN_CHARS


def test_a_long_query_is_cut_to_fit_the_column():
    db = _DB()
    asyncio.run(GovernanceRepository(db).log_gap(REPORTED_QUERY, "qnsc.vn"))

    assert len(db.added) == 1
    assert len(db.added[0].query) == COLUMN_CHARS
    assert db.added[0].query == REPORTED_QUERY[:COLUMN_CHARS]


def test_a_short_query_is_stored_whole():
    db = _DB()
    asyncio.run(GovernanceRepository(db).log_gap("who owns payroll?", "qnsc.vn"))
    assert db.added[0].query == "who owns payroll?"


def test_the_limit_comes_from_the_column_not_a_literal():
    """A widened column must not leave a 255 hard-coded somewhere still cutting."""
    assert GovernanceRepository._GAP_QUERY_CHARS == COLUMN_CHARS


@pytest.mark.parametrize("query", ["", None])
def test_an_empty_query_does_not_crash_the_truncation(query):
    db = _DB()
    asyncio.run(GovernanceRepository(db).log_gap(query, "qnsc.vn"))
    assert db.added[0].query == ""


# --- the guard at the call site --------------------------------------------


class _ExplodingRepo:
    """A gap writer that fails the way the real one did."""

    def __init__(self):
        self.db = _DB()
        self.calls = 0

    async def log_gap(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("value too long for type character varying(255)")


def _search_service(gov_repo):
    from src.domain.search_service import SearchService

    return SearchService(chunk_repo=None, gov_repo=gov_repo)


def test_a_failed_gap_write_does_not_reach_the_caller():
    """The whole point. Search returns its (empty) result rather than raising."""
    repo = _ExplodingRepo()
    asyncio.run(_search_service(repo)._record_gap(_User(), "anything"))
    assert repo.calls == 1


def test_a_failed_gap_write_resets_the_session():
    """log_gap commits, so a failure leaves a failed transaction behind. Without the
    rollback the next statement in the same request raises and the 500 simply moves."""
    repo = _ExplodingRepo()
    asyncio.run(_search_service(repo)._record_gap(_User(), "anything"))
    assert repo.db.rollbacks == 1


def test_a_rollback_that_also_fails_is_still_survivable():
    """Nothing in this path may raise; it runs while returning an empty result."""

    class _WorseRepo(_ExplodingRepo):
        def __init__(self):
            super().__init__()

            class _NoRollback:
                async def rollback(self_inner):
                    raise RuntimeError("connection is gone")

            self.db = _NoRollback()

    asyncio.run(_search_service(_WorseRepo())._record_gap(_User(), "anything"))


def test_a_working_gap_write_is_not_rolled_back():
    class _QuietRepo:
        def __init__(self):
            self.db = _DB()
            self.calls = 0

        async def log_gap(self, **_kwargs):
            self.calls += 1

    repo = _QuietRepo()
    asyncio.run(_search_service(repo)._record_gap(_User(), "anything"))
    assert repo.calls == 1 and repo.db.rollbacks == 0
