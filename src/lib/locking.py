"""Database-backed locks for article indexing and permission reconciliation."""
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# 64-bit and transaction-scoped, the same pair sync_queue.claim_sync_request uses.
#
# `hashtext` returns int4, so two different article ids collide roughly once per 2^16 of
# them (birthday bound) — and a collision here is not a near miss, it is two unrelated
# articles serialised against each other for a full index run, one blocking on a lock it
# has no reason to want. `hashtextextended` is int8 and takes the same single-argument
# advisory lock form.
#
# Transaction scope is what the indexing rewrite bought: a session-level lock had to be
# unlocked explicitly, and the unlock is rejected by an aborted transaction — so a failed
# index returned a connection to the pool still holding the lock, and because the pool's
# reset-on-return is a ROLLBACK, which does not touch advisory locks, nothing ever released
# it. One failed index left an article permanently stuck in `processing`. Now that the
# whole rebuild is one transaction, the lock ends exactly when that transaction does, on
# commit OR rollback, with nothing to leak.
_ARTICLE_LOCK = text("SELECT pg_advisory_xact_lock(hashtextextended(:lock_key, 0))")


@asynccontextmanager
async def article_lock(db: AsyncSession, article_id: str) -> AsyncIterator[None]:
    """Serialize lifecycle work for one article across API processes."""
    await db.execute(_ARTICLE_LOCK, {"lock_key": f"qnsc:article:{article_id}"})
    yield
