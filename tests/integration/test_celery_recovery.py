"""The stuck-article sweeper and the idempotent outbox handler, against a real, migrated PostgreSQL.

The unit tests prove the decisions; these prove the SQL: the `FOR UPDATE SKIP LOCKED` claim, the
`updated_at` cutoff, that only published, active articles are touched, and that a marker written by one
sweep is still there when the next one reads it.

Everything runs on the worker's own event loop (`sync_run`), because the shared engine binds its pool to the
loop that first used it. Rows are tagged and removed afterwards.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete, select

from src.workers.loop import sync_run

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs a real PostgreSQL; the point is the claim query and the row lock",
)

TAG = "it-celery-"
NOW = datetime(2030, 1, 1, 12, 0, 0)
LONG_AGO = NOW - timedelta(hours=5)


def _run(coro):
    return sync_run(coro)


def _article(title: str, **overrides):
    from src.models.article import Article

    values = dict(
        title=f"{TAG}{title}",
        body_md="# body",
        dept="Kỹ thuật",
        domain="ops",
        company_domain="it-celery.test",
        type="SOP",
        sensitivity="internal",
        visibility="department",
        status="published",
        lifecycle_status="active",
        version=1,
        needs_update=False,
        index_status="processing",
        index_error=None,
        created_at=LONG_AGO,
        updated_at=LONG_AGO,
    )
    values.update(overrides)
    return Article(**values)


@pytest.fixture
def db(monkeypatch):
    """A session factory of this test's own, installed in every module under test.

    Not `src.api.deps.SessionLocal`: that engine pools connections, and in a full run the unit tests
    before this file have already used it on other event loops, so its pool hands back a connection that is
    still tied to one ("another operation is in progress"). A pool created here lives and dies with this test,
    on the one loop it runs on.
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from src.api.deps import set_database_context
    from src.domain import index_recovery, indexing
    from src.models.article import Article
    from src.models.ops import OutboxEvent
    from src.workers import tasks

    engine = create_async_engine(os.environ["DATABASE_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    for module in (index_recovery, indexing, tasks):
        monkeypatch.setattr(module, "SessionLocal", factory)

    async def clean():
        async with factory() as session:
            await set_database_context(session, None, True)
            await session.execute(delete(Article).where(Article.title.like(f"{TAG}%")))
            await session.execute(
                delete(OutboxEvent).where(OutboxEvent.event_type.like(f"{TAG}%"))
            )
            await session.commit()

    _run(clean())
    yield factory
    _run(clean())
    _run(engine.dispose())


async def _seed(factory, *articles):
    from src.api.deps import set_database_context

    async with factory() as session:
        await set_database_context(session, None, True)
        session.add_all(articles)
        await session.commit()
        return [article.id for article in articles]


async def _state(factory, article_id):
    from src.api.deps import set_database_context
    from src.models.article import Article

    async with factory() as session:
        await set_database_context(session, None, True)
        row = (
            await session.execute(
                select(
                    Article.index_status, Article.index_error, Article.updated_at
                ).where(Article.id == article_id)
            )
        ).one()
        return row


# ── the sweeper ───────────────────────────────────────────────────────────────────────────


def test_a_stuck_article_is_requeued_then_failed_after_the_cap(db):
    from src.domain.index_recovery import (
        INDEX_RECOVERY_MAX_ATTEMPTS,
        INDEX_STUCK_MINUTES,
        INTERRUPTED_PREFIX,
        recover_stuck_indexing,
    )

    window = timedelta(minutes=INDEX_STUCK_MINUTES + 1)
    [stuck] = _run(_seed(db, _article("stuck")))
    moment = NOW

    # Each interruption within the cap queues it again and records the count.
    for attempt in range(1, INDEX_RECOVERY_MAX_ATTEMPTS + 1):
        requeue, failed = _run(recover_stuck_indexing(now=moment))
        assert stuck in requeue and stuck not in failed
        status, error, touched = _run(_state(db, stuck))
        assert (
            status == "processing"
        ), "a requeued article stays sweepable if the new run dies too"
        assert error.startswith(INTERRUPTED_PREFIX) and f"attempt {attempt} of" in error
        assert touched == moment, "the stuck clock restarts"
        # Not found again straight away: it has to be stuck for the whole window once more.
        soon = moment + timedelta(minutes=5)
        assert stuck not in sum(_run(recover_stuck_indexing(now=soon)), [])
        moment += window

    # One interruption past the cap, and it is reported instead of retried.
    requeue, failed = _run(recover_stuck_indexing(now=moment))
    assert stuck in failed and stuck not in requeue
    status, error, _ = _run(_state(db, stuck))
    assert status == "failed" and "reprocess" in error.lower()

    # A failed article is out of the sweeper's reach for good.
    assert stuck not in sum(_run(recover_stuck_indexing(now=moment + window * 4)), [])


def test_only_articles_stuck_longer_than_the_window_are_touched(db):
    from src.domain.index_recovery import INDEX_STUCK_MINUTES, recover_stuck_indexing

    just_inside = NOW - timedelta(minutes=INDEX_STUCK_MINUTES - 5)
    just_outside = NOW - timedelta(minutes=INDEX_STUCK_MINUTES + 5)
    running, stale = _run(
        _seed(
            db,
            _article("running", updated_at=just_inside),
            _article("stale", updated_at=just_outside),
        )
    )
    requeue, failed = _run(recover_stuck_indexing(now=NOW))
    assert stale in requeue
    assert running not in requeue and running not in failed
    assert _run(_state(db, running))[0] == "processing"
    assert (
        _run(_state(db, running))[1] is None
    ), "a run still inside its window is left exactly as it was"


def test_only_live_published_processing_articles_are_considered(db):
    from src.domain.index_recovery import recover_stuck_indexing

    ids = _run(
        _seed(
            db,
            _article("draft", status="draft"),
            _article("archived", lifecycle_status="archived"),
            _article("ready", index_status="ready"),
            _article("failed", index_status="failed", index_error="boom"),
            _article("pending", index_status="pending"),
        )
    )
    requeue, failed = _run(recover_stuck_indexing(now=NOW))
    assert not set(ids) & (set(requeue) | set(failed))
    assert (
        _run(_state(db, ids[3]))[1] == "boom"
    ), "someone else's error text is never overwritten"


def test_a_pass_is_bounded_so_one_sweep_stays_short(db):
    from src.domain.index_recovery import BATCH_LIMIT, recover_stuck_indexing

    ids = _run(_seed(db, *[_article(f"bulk-{n}") for n in range(BATCH_LIMIT + 5)]))
    requeue, failed = _run(recover_stuck_indexing(now=NOW))
    mine = [item for item in requeue + failed if item in ids]
    assert len(mine) == BATCH_LIMIT
    # The remainder is picked up by the next pass, not lost.
    again, _ = _run(recover_stuck_indexing(now=NOW))
    assert len([item for item in again if item in ids]) == 5


def test_two_sweeps_at_once_claim_different_articles(db):
    """`SKIP LOCKED`: a second pass that overlaps the first takes other rows instead of waiting or repeating."""
    import asyncio

    from src.domain.index_recovery import BATCH_LIMIT, recover_stuck_indexing

    ids = _run(_seed(db, *[_article(f"race-{n}") for n in range(BATCH_LIMIT * 2)]))

    async def both():
        return await asyncio.gather(
            recover_stuck_indexing(now=NOW), recover_stuck_indexing(now=NOW)
        )

    (first, _), (second, _) = _run(both())
    mine_first = {item for item in first if item in ids}
    mine_second = {item for item in second if item in ids}
    assert not mine_first & mine_second, "the same article was claimed by two sweeps"
    assert len(mine_first | mine_second) == BATCH_LIMIT * 2


def test_the_marker_survives_the_retry_and_a_success_clears_it(db):
    from src.domain.index_recovery import INTERRUPTED_PREFIX, recover_stuck_indexing
    from src.domain.indexing import set_index_status

    [article] = _run(_seed(db, _article("marker")))
    _run(recover_stuck_indexing(now=NOW))
    marker = _run(_state(db, article))[1]
    assert marker.startswith(INTERRUPTED_PREFIX)

    # The retried run begins by setting "processing": the count must still be there afterwards.
    _run(set_index_status(article, "processing"))
    assert _run(_state(db, article))[1] == marker
    # Success clears it.
    _run(set_index_status(article, "ready"))
    status, error, _ = _run(_state(db, article))
    assert status == "ready" and error is None


def test_an_ordinary_old_error_is_still_cleared_when_indexing_starts_again(db):
    from src.domain.indexing import set_index_status

    [article] = _run(
        _seed(
            db,
            _article(
                "old-error", index_status="failed", index_error="DataError: wrong width"
            ),
        )
    )
    _run(set_index_status(article, "processing"))
    status, error, _ = _run(_state(db, article))
    assert (
        status == "processing" and error is None
    ), "a stale failure message must not sit under a running job"


# ── the outbox handler ────────────────────────────────────────────────────────────────────


def _outbox(factory, status: str):
    from src.api.deps import set_database_context
    from src.models.ops import OutboxEvent

    async def create():
        async with factory() as session:
            await set_database_context(session, None, True)
            event = OutboxEvent(
                event_type=f"{TAG}ArticlePublished",
                payload={},
                status=status,
                attempts=1,
            )
            session.add(event)
            await session.commit()
            return event.id

    return _run(create())


def _outbox_status(factory, event_id):
    from src.api.deps import set_database_context
    from src.models.ops import OutboxEvent

    async def read():
        async with factory() as session:
            await set_database_context(session, None, True)
            return (
                await session.execute(
                    select(OutboxEvent.status).where(OutboxEvent.id == event_id)
                )
            ).scalar_one()

    return _run(read())


class _Delay:
    def __init__(self):
        self.calls: list[str] = []
        self.fail = False

    def delay(self, article_id):
        if self.fail:
            raise ConnectionError("broker down")
        self.calls.append(article_id)


def test_a_duplicate_event_message_queues_the_work_once_against_a_real_row(
    db, monkeypatch
):
    from src.workers import tasks

    recorder = _Delay()
    monkeypatch.setattr(tasks, "generate_embeddings_task", recorder)
    event_id = _outbox(db, "processing")
    article = str(uuid.uuid4())
    payload = {"article_id": article, "_outbox_id": str(event_id)}

    tasks.handle_domain_event_task("ArticlePublished", dict(payload))
    tasks.handle_domain_event_task(
        "ArticlePublished", dict(payload)
    )  # the copy replay sent while the first waited

    assert recorder.calls == [article]
    assert _outbox_status(db, event_id) == "dispatched"


def test_a_broker_failure_leaves_the_row_claimable_and_the_retry_succeeds(
    db, monkeypatch
):
    from src.workers import tasks

    recorder = _Delay()
    monkeypatch.setattr(tasks, "generate_embeddings_task", recorder)
    event_id = _outbox(db, "processing")
    payload = {"article_id": str(uuid.uuid4()), "_outbox_id": str(event_id)}

    recorder.fail = True
    with pytest.raises(ConnectionError):
        tasks.handle_domain_event_task("ArticlePublished", dict(payload))
    assert (
        _outbox_status(db, event_id) == "processing"
    ), "a failed fan-out must not be recorded as dispatched"

    recorder.fail = False
    tasks.handle_domain_event_task("ArticlePublished", dict(payload))
    assert recorder.calls == [payload["article_id"]]
    assert _outbox_status(db, event_id) == "dispatched"
