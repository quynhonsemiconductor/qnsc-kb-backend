"""Celery: stale ticks, duplicate events and articles that are interrupted without a trace.

The worker is ONE process (`--concurrency=1`, sized for one embedding session) that also runs hour-long
connector syncs and OCR. Three consequences are pinned here, each of which was reproduced first:

* beat enqueues on the clock whether or not anything consumes. With no `expires`, 100 s without a consumer left
  8 messages in Redis and the restarted worker ran all of them. Every schedule entry now expires with its tick.
* `replay_outbox_task` re-sends an event whose backoff has passed even while the first message is still
  queued behind a long task, so the same event can wait twice and used to fan out twice.
* a prefork child killed by the OOM killer raises nothing, so its article stayed `processing` for good and
  dropped out of search.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from celery.schedules import crontab

from src.domain import index_recovery as recovery
from src.workers import tasks
from src.workers.celery_app import (
    TASK_HARD_TIME_LIMIT,
    TASK_SOFT_TIME_LIMIT,
    VISIBILITY_TIMEOUT,
    celery_app,
    periodic,
)

SCHEDULE = celery_app.conf.beat_schedule


# ── beat: nothing outlives its own tick ───────────────────────────────────────────────────


def test_every_scheduled_task_expires_instead_of_queueing_up():
    missing = [
        name
        for name, entry in SCHEDULE.items()
        if "expires" not in entry.get("options", {})
    ]
    assert (
        not missing
    ), f"beat entries with no expiry would pile up behind a busy worker: {missing}"


def test_an_entry_expires_when_its_next_tick_would_have_run_anyway():
    for name, entry in SCHEDULE.items():
        expires = entry["options"]["expires"]
        interval = entry["schedule"]
        if isinstance(interval, crontab):
            # A wall-clock entry runs once a day; yesterday's missed run is not worth running late.
            assert expires == 86400.0, name
        else:
            assert (
                expires >= 30.0
            ), f"{name}: below the floor, a worker a few seconds behind would skip it"
            assert expires <= max(
                float(interval), 30.0
            ), f"{name}: outlives its own next tick"


def test_periodic_builds_an_expiring_entry():
    entry = periodic("some_task", 600)
    assert entry == {
        "task": "some_task",
        "schedule": 600.0,
        "options": {"expires": 600.0},
    }
    # Never below the floor: a 5 s interval must not discard a tick a busy worker reaches at 6 s.
    assert periodic("fast", 5)["options"]["expires"] == 30.0
    wall = crontab(hour=1, minute=0)
    assert periodic("daily", 86400, schedule=wall)["schedule"] is wall


def test_every_scheduled_task_is_registered_and_routed_somewhere_that_is_consumed():
    consumed = {
        "document-preparation",
        "celery",
        "ingestion",
        "connectors",
        "permissions",
    }
    routes = celery_app.conf.task_routes
    for name, entry in SCHEDULE.items():
        assert (
            entry["task"] in celery_app.tasks
        ), f"{name} schedules a task nobody registered"
        queue = routes.get(entry["task"], {}).get("queue", "celery")
        assert queue in consumed, f"{name} goes to {queue!r}, which no worker consumes"


def test_the_sweeper_is_scheduled_and_runs_beside_the_embedding_work():
    assert any(entry["task"] == "recover_stuck_indexing" for entry in SCHEDULE.values())
    assert celery_app.conf.task_routes["recover_stuck_indexing"] == {
        "queue": "ingestion"
    }


def test_time_limits_stay_ordered_so_redelivery_only_follows_a_killed_worker():
    assert TASK_SOFT_TIME_LIMIT < TASK_HARD_TIME_LIMIT < VISIBILITY_TIMEOUT
    assert (
        celery_app.conf.broker_transport_options["visibility_timeout"]
        == VISIBILITY_TIMEOUT
    )


def test_results_are_not_stored_because_nothing_reads_them():
    assert celery_app.conf.task_ignore_result is True
    assert celery_app.conf.broker_connection_retry_on_startup is True


# ── the sweeper's decisions ───────────────────────────────────────────────────────────────


def test_an_article_is_only_swept_after_every_legitimate_run_and_redelivery_is_over():
    assert recovery.INDEX_STUCK_MINUTES * 60 > VISIBILITY_TIMEOUT > TASK_HARD_TIME_LIMIT


@pytest.mark.parametrize(
    ("index_error", "expected"),
    [
        (None, 0),
        ("", 0),
        (
            "Embedding unavailable for 3 chunk(s)",
            0,
        ),  # an ordinary failure is not a recovery count
        (
            f"{recovery.INTERRUPTED_PREFIX} (attempt 1 of 2); queued again automatically.",
            1,
        ),
        (
            f"{recovery.INTERRUPTED_PREFIX} (attempt 2 of 2); queued again automatically.",
            2,
        ),
        (f"{recovery.INTERRUPTED_PREFIX} but no number", 0),
        ("Something else (attempt 2 of 2)", 0),  # only our own marker counts
    ],
)
def test_the_recorded_attempts_are_read_only_from_our_marker(index_error, expected):
    assert recovery.interrupted_attempts(index_error) == expected


def test_an_interrupted_article_is_requeued_until_the_cap_and_then_reported_failed():
    action, first = recovery.next_step(None)
    assert (
        action == "requeue"
        and "attempt 1 of" in first
        and first.startswith(recovery.INTERRUPTED_PREFIX)
    )
    action, second = recovery.next_step(first)
    assert action == "requeue" and "attempt 2 of" in second
    action, message = recovery.next_step(second)
    assert action == "fail" and "reprocess" in message.lower()
    # The cap holds however the sequence is entered: an old failure message starts from the beginning.
    assert recovery.next_step("DataError: expected 1024 dimensions")[0] == "requeue"


def test_the_failure_message_says_what_happened_and_what_to_do():
    message = recovery.next_step(
        f"{recovery.INTERRUPTED_PREFIX} (attempt {recovery.INDEX_RECOVERY_MAX_ATTEMPTS} of "
        f"{recovery.INDEX_RECOVERY_MAX_ATTEMPTS}); queued again automatically."
    )[1]
    assert str(recovery.INDEX_RECOVERY_MAX_ATTEMPTS + 1) in message
    assert "worker" in message


# ── the outbox handler is idempotent ──────────────────────────────────────────────────────


class _Event:
    def __init__(self, status: str):
        self.status = status
        self.last_error = "old error"


class _Session:
    """Stands in for the AsyncSession: serves one outbox row and records the commit."""

    def __init__(self, event):
        self.event = event
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, _statement):
        return SimpleNamespace(scalar_one_or_none=lambda: self.event)

    async def commit(self):
        self.committed = True


class _Recorder:
    def __init__(self, raises: Exception | None = None):
        self.calls: list[str] = []
        self.raises = raises

    def delay(self, article_id: str):
        if self.raises:
            raise self.raises
        self.calls.append(article_id)


@pytest.fixture
def fan_out(monkeypatch):
    recorders = {
        "embed": _Recorder(),
        "permissions": _Recorder(),
        "delete": _Recorder(),
    }
    monkeypatch.setattr(tasks, "generate_embeddings_task", recorders["embed"])
    monkeypatch.setattr(tasks, "recompute_permissions_task", recorders["permissions"])
    monkeypatch.setattr(tasks, "delete_article_chunks_task", recorders["delete"])

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tasks, "set_database_context", no_context)
    return recorders


def _handle(
    monkeypatch, event, event_type="ArticlePublished", article_id=None, outbox=True
):
    session = _Session(event)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    payload = {"article_id": article_id or str(uuid.uuid4())}
    if outbox:
        payload["_outbox_id"] = str(uuid.uuid4())
    tasks.handle_domain_event_task(event_type, payload)
    return session, payload


def test_a_first_copy_fans_out_once_and_marks_the_event_dispatched(
    monkeypatch, fan_out
):
    event = _Event("processing")
    session, payload = _handle(monkeypatch, event)
    assert fan_out["embed"].calls == [payload["article_id"]]
    assert event.status == "dispatched" and event.last_error is None
    assert session.committed


@pytest.mark.parametrize("settled", ["dispatched", "completed", "dead"])
def test_a_second_copy_of_the_same_event_does_nothing(monkeypatch, fan_out, settled):
    event = _Event(settled)
    session, _ = _handle(monkeypatch, event)
    assert (
        fan_out["embed"].calls == []
        and fan_out["permissions"].calls == []
        and fan_out["delete"].calls == []
    )
    assert event.status == settled
    assert not session.committed


def test_two_copies_in_a_row_queue_the_work_once(monkeypatch, fan_out):
    event = _Event("processing")
    article = str(uuid.uuid4())
    _handle(monkeypatch, event, article_id=article)
    _handle(monkeypatch, event, article_id=article)
    assert fan_out["embed"].calls == [article]


@pytest.mark.parametrize(
    ("event_type", "target"),
    [
        ("ArticleUpdated", "embed"),
        ("PermissionChanged", "permissions"),
        ("ArticleDeleted", "delete"),
    ],
)
def test_each_event_type_still_reaches_its_task(
    monkeypatch, fan_out, event_type, target
):
    _, payload = _handle(monkeypatch, _Event("processing"), event_type=event_type)
    assert fan_out[target].calls == [payload["article_id"]]


def test_an_unknown_event_type_or_a_missing_article_queues_nothing(
    monkeypatch, fan_out
):
    _handle(monkeypatch, _Event("processing"), event_type="SomethingNew")
    tasks.handle_domain_event_task(
        "ArticlePublished", {"_outbox_id": str(uuid.uuid4())}
    )
    assert all(not recorder.calls for recorder in fan_out.values())


def test_a_failed_fan_out_leaves_the_event_for_the_next_replay(monkeypatch, fan_out):
    fan_out["embed"].raises = ConnectionError("broker down")
    event = _Event("processing")
    session = _Session(event)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    with pytest.raises(ConnectionError):
        tasks.handle_domain_event_task(
            "ArticlePublished",
            {"article_id": str(uuid.uuid4()), "_outbox_id": str(uuid.uuid4())},
        )
    # Not committed, so the row is still "processing" and replay_outbox_task will try again.
    assert event.status == "processing" and not session.committed


def test_an_event_whose_outbox_row_is_gone_is_still_handled(monkeypatch, fan_out):
    session, payload = _handle(monkeypatch, None)
    assert fan_out["embed"].calls == [payload["article_id"]]
    assert session.committed


def test_an_event_that_was_never_recorded_fans_out_without_touching_the_database(
    monkeypatch, fan_out
):
    def explode():
        raise AssertionError("no outbox row exists, so there is nothing to claim")

    monkeypatch.setattr(tasks, "SessionLocal", explode)
    article = str(uuid.uuid4())
    tasks.handle_domain_event_task("ArticlePublished", {"article_id": article})
    assert fan_out["embed"].calls == [article]


def test_the_handler_runs_on_the_workers_own_loop_not_a_new_one_per_call(
    monkeypatch, fan_out
):
    # sync_run keeps ONE loop per worker process (asyncpg pools bind to a loop); the handler must use it.
    seen = []
    real = tasks.sync_run

    def spy(coro):
        seen.append(asyncio.iscoroutine(coro))
        return real(coro)

    monkeypatch.setattr(tasks, "sync_run", spy)
    _handle(monkeypatch, _Event("processing"))
    assert seen == [True]
