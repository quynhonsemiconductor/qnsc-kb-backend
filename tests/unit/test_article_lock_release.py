"""`article_lock` must not be able to outlive the work it guards, or collide across ids.

This used to take a SESSION-level advisory lock, which outlives its transaction and is
released only by an explicit unlock or by the connection closing. That is unrecoverable
in-band when the body fails: the transaction is aborted, Postgres rejects the unlock too,
and the pool's reset-on-return is a ROLLBACK, which does not touch advisory locks. It
happened -- one failed index left an article stuck in `index_status='processing'` with the
lock held by an idle backend, and reindexing timed out until that backend was terminated.

The session lock existed because the chunk repository committed several times inside the
lock, and a transaction-scoped lock would have ended at the first of those commits. The
rebuild is now ONE transaction, so `pg_advisory_xact_lock` is both correct and self-
releasing: it ends when that transaction ends, on commit or rollback, with nothing left to
leak and no unlock to reject.

The width matters separately. `hashtext` is int4, so unrelated article ids collide at a
birthday bound around 2^16 ids -- and a collision serialises two unrelated articles against
each other for a full index run. `hashtextextended` is int8.
"""
from __future__ import annotations

import asyncio

import pytest

from src.lib.locking import article_lock

ARTICLE_ID = "69808cf8-f04d-412b-8250-ae29653511db"


class FakeSession:
    """Rejects statements once the transaction is aborted, as Postgres does."""

    def __init__(self) -> None:
        self.aborted = False
        self.statements: list[str] = []
        self.params: list[dict] = []
        self.rollbacks = 0

    async def execute(self, statement, params=None):
        if self.aborted:
            raise RuntimeError("current transaction is aborted, commands ignored")
        self.statements.append(str(statement))
        self.params.append(params or {})
        return None

    async def rollback(self) -> None:
        self.rollbacks += 1
        self.aborted = False


def _run(session: FakeSession, *, fail: bool) -> None:
    async def body() -> None:
        async with article_lock(session, ARTICLE_ID):
            if fail:
                session.aborted = True  # what a failed statement leaves behind
                raise ValueError("indexing failed")

    asyncio.run(body())


def test_the_lock_is_transaction_scoped_so_it_cannot_ride_a_pooled_connection():
    session = FakeSession()
    _run(session, fail=False)

    assert len(session.statements) == 1
    assert "pg_advisory_xact_lock" in session.statements[0]


def test_the_lock_key_is_64_bit():
    """int4 collisions block two unrelated articles against each other for a whole run."""
    session = FakeSession()
    _run(session, fail=False)

    assert "hashtextextended" in session.statements[0]
    assert "hashtext(" not in session.statements[0]


def test_the_key_is_bound_not_interpolated():
    session = FakeSession()
    _run(session, fail=False)

    assert session.params[0] == {"lock_key": f"qnsc:article:{ARTICLE_ID}"}


def test_a_failed_body_needs_no_unlock_and_no_recovery_rollback():
    """The case that leaked before. An aborted transaction rejects every statement, so the
    old explicit unlock could not run -- and now there is nothing to run: the rollback the
    caller performs releases the lock as a side effect of ending the transaction."""
    session = FakeSession()
    with pytest.raises(ValueError):
        _run(session, fail=True)

    assert not [sql for sql in session.statements if "unlock" in sql.lower()]
    assert session.rollbacks == 0, "the lock needs no rollback of its own to be released"
