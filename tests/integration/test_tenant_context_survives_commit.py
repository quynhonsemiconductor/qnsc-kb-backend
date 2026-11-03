"""The tenant context must survive a commit, on whatever connection follows it.

WHY THIS TEST LOOKS ODD. It warms the connection pool first, and that is the entire
point: with a single session and a fresh pool the bug is INVISIBLE — the session gets its
own connection back after committing and the context is still on it. Every simpler test
passes against the broken code, which is why this shipped.

The failure needs two things that only happen in a real process:

  1. an AsyncSession releases its connection at commit and takes one again for the next
     statement (measured: a different backend on 11 of 12 runs), and
  2. the pool contains connections that carry no `app.*` settings.

Before the fix, get_db() produced (2) itself by RESETting app.* on release — correct in
isolation, since no request may inherit another's tenant, but combined with (1) it meant
any statement after a commit ran with NO context. Under FORCEd RLS that returns zero rows:
/ai/ask surfaced it as "Could not refresh instance", while a plain read would have
returned nothing and reported success.

The fix makes the context transaction-local and re-applies it on every `after_begin`, so
neither (1) nor (2) can strip it.
"""
from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs a real PostgreSQL connection pool; the behaviour under test is the pool's",
)


@pytest_asyncio.fixture
async def sessions():
    """A pool of this test's own, disposed with it.

    Deliberately not src.api.deps.SessionLocal: that engine is created at import and
    binds to whichever event loop imported it, while pytest-asyncio gives each test a new
    one — the mismatch shows up as "Event loop is closed" and "another operation is in
    progress", neither of which has anything to do with what is being tested.

    The listener under test is registered on the Session CLASS, so any sessionmaker
    exercises it.
    """
    engine = create_async_engine(os.environ["DATABASE_URL"], pool_size=6, max_overflow=0)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield factory
    finally:
        await engine.dispose()

READ_CONTEXT = text(
    "SELECT current_setting('app.user_id', true), current_setting('app.company_domain', true)"
)

USER_ID = "98b75957-497b-44ec-b7a8-e7c11e44e769"
DOMAIN = "qnsc.vn"


async def _strip_pooled_connections(factory, count: int = 6) -> None:
    """Fill the pool with connections carrying no app.* settings.

    Reproduces what get_db's RESET block used to leave behind, and what any connection
    that has never served a tenant-scoped request looks like anyway.
    """
    async def one() -> None:
        async with factory() as session:
            for name in (
                "app.company_domain",
                "app.global_admin",
                "app.user_id",
                "app.global_article_access",
                "app.global_identity_access",
                "app.global_connector_access",
                "app.global_governance_access",
            ):
                await session.execute(text(f"RESET {name}"))
            await session.commit()

    await asyncio.gather(*(one() for _ in range(count)))


@pytest.mark.asyncio
async def test_context_survives_a_commit(sessions):
    from src.api.deps import set_database_context

    await _strip_pooled_connections(sessions)

    async with sessions() as session:
        await set_database_context(session, DOMAIN, False, USER_ID)
        before = list((await session.execute(READ_CONTEXT)).first())
        assert before == [USER_ID, DOMAIN]

        await session.commit()

        after = list((await session.execute(READ_CONTEXT)).first())
        assert after == [USER_ID, DOMAIN], (
            "tenant context was lost after commit — the session took a pooled connection "
            "that never received it, so every following statement runs unscoped and RLS "
            "returns nothing"
        )


@pytest.mark.asyncio
async def test_context_survives_several_commits(sessions):
    """Repeated, because one commit may coincidentally reuse the same connection."""
    from src.api.deps import set_database_context

    await _strip_pooled_connections(sessions)

    async with sessions() as session:
        await set_database_context(session, DOMAIN, False, USER_ID)
        for attempt in range(5):
            await session.commit()
            observed = list((await session.execute(READ_CONTEXT)).first())
            assert observed == [USER_ID, DOMAIN], f"context lost on commit {attempt + 1}"


@pytest.mark.asyncio
async def test_context_does_not_leak_to_a_session_that_never_set_one(sessions):
    """The property get_db's RESET block used to provide, now structural.

    A session with no context must see none — not the previous request's. Transaction
    local settings cannot outlive their transaction, so this holds without any cleanup
    step that has to be remembered.
    """
    from src.api.deps import set_database_context

    async with sessions() as session:
        await set_database_context(session, DOMAIN, False, USER_ID)
        await session.commit()

    async with sessions() as session:
        observed = list((await session.execute(READ_CONTEXT)).first())
        assert observed == ["", ""], f"leaked tenant context into a fresh session: {observed}"


@pytest.mark.asyncio
async def test_a_session_without_context_is_not_given_one(sessions):
    """Workers, the migrator and startup open sessions with no tenant deliberately.

    They must keep failing closed under RLS rather than inheriting a context, so the
    listener has to skip them rather than apply some default.
    """
    
    await _strip_pooled_connections(sessions)

    async with sessions() as session:
        observed = list((await session.execute(READ_CONTEXT)).first())
        assert observed == ["", ""]
