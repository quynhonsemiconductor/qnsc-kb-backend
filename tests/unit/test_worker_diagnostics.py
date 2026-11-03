"""The verdict must separate "slow" from "broken", because they need opposite responses.

An upload sat at "preparing" indefinitely and every retry did nothing, while the UI and
`/health/ready` both reported healthy — the API and the worker are separate services, so
a ready API says nothing about whether a worker exists to drain a queue. With
`min_count = 0` on develop, no worker at all is a reachable steady state.

The distinction this file pins:

    backlog WITH a consumer     -> wait
    backlog WITHOUT a consumer  -> scale the worker up

Getting that backwards is the whole failure: a deep queue reported as broken sends an
operator chasing a fault that is only a queue, and a starved queue reported as healthy
reproduces exactly the silence this endpoint exists to break.
"""

from __future__ import annotations

import inspect

import pytest

from src.domain.worker_diagnostics import (
    QueueSnapshot,
    VERDICT_BROKER_UNREACHABLE,
    VERDICT_BUSY,
    VERDICT_HEALTHY,
    VERDICT_INLINE_MODE,
    VERDICT_NO_CONSUMER,
    VERDICT_QUEUE_UNCONSUMED,
    WorkerSnapshot,
    attach_consumers,
    summarise,
)

QUEUES = ["celery", "connectors", "document-preparation", "ingestion", "permissions"]


def _worker(name: str = "celery@host", queues=QUEUES, active=()) -> WorkerSnapshot:
    return WorkerSnapshot(
        name=name, queues=tuple(queues), active_task_names=tuple(active)
    )


def _queues(depths: dict[str, int] | None = None, workers=None) -> list[QueueSnapshot]:
    filled = {name: 0 for name in QUEUES}
    filled.update(depths or {})
    return attach_consumers(QUEUES, filled, workers if workers is not None else [])


# --- a live consumer, nothing waiting ------------------------------------------


def test_a_consumed_and_empty_queue_set_is_healthy():
    worker = _worker()
    result = summarise(
        job_mode="celery", workers=[worker], queues=_queues(workers=[worker])
    )

    assert result["verdict"] == VERDICT_HEALTHY
    assert result["healthy"] is True
    assert result["worker_count"] == 1
    assert result["total_queued"] == 0
    assert result["queues_without_consumer"] == []


# --- the distinction that matters ---------------------------------------------


def test_a_backlog_with_a_live_consumer_is_busy_not_broken():
    """Work is moving. Reporting this as a fault sends an operator after nothing."""
    worker = _worker()
    result = summarise(
        job_mode="celery",
        workers=[worker],
        queues=_queues({"document-preparation": 12}, workers=[worker]),
    )

    assert result["verdict"] == VERDICT_BUSY
    assert result["healthy"] is True
    assert result["total_queued"] == 12
    assert result["queues_with_backlog_and_no_consumer"] == []


def test_a_backlog_with_no_consumer_is_the_actionable_failure():
    """The reported incident: queued work, nobody draining it, retries piling on."""
    result = summarise(
        job_mode="celery",
        workers=[],
        queues=_queues({"document-preparation": 3}),
    )

    assert result["verdict"] == VERDICT_NO_CONSUMER
    assert result["healthy"] is False
    assert "retry" in result["detail"]


def test_a_worker_that_skips_one_queue_leaves_that_queue_starved():
    """A worker CAN answer while consuming a subset -- a `-Q` list missing a queue.

    Everything routed there never runs, and a plain "a worker is alive" check calls that
    healthy. This is why the verdict is about consumption per queue, not liveness.
    """
    worker = _worker(queues=[q for q in QUEUES if q != "document-preparation"])
    result = summarise(
        job_mode="celery",
        workers=[worker],
        queues=_queues({"document-preparation": 5}, workers=[worker]),
    )

    assert result["verdict"] == VERDICT_QUEUE_UNCONSUMED
    assert result["healthy"] is False
    assert result["queues_with_backlog_and_no_consumer"] == ["document-preparation"]


def test_an_unconsumed_queue_is_reported_before_anything_lands_on_it():
    """Latent version of the same fault. An operator wants it before it bites."""
    worker = _worker(queues=["celery"])
    result = summarise(
        job_mode="celery", workers=[worker], queues=_queues(workers=[worker])
    )

    assert result["queues_without_consumer"] == [
        "connectors",
        "document-preparation",
        "ingestion",
        "permissions",
    ]
    # No backlog yet, so not the urgent verdict -- but visible.
    assert result["verdict"] == VERDICT_HEALTHY


def test_no_worker_is_a_fault_even_with_every_queue_empty():
    """Nothing runs, including the beat schedule. Empty queues are not reassurance."""
    result = summarise(job_mode="celery", workers=[], queues=_queues())

    assert result["verdict"] == VERDICT_NO_CONSUMER
    assert result["healthy"] is False


# --- modes and failures that must not read as broken --------------------------


@pytest.mark.parametrize("mode", ["inline", "INLINE", " inline "])
def test_inline_mode_expects_no_worker(mode):
    """The API runs jobs in-process here, so an empty worker list is correct."""
    result = summarise(job_mode=mode, workers=[], queues=_queues())

    assert result["verdict"] == VERDICT_INLINE_MODE
    assert result["healthy"] is True


def test_inline_mode_still_reports_a_worker_that_does_exist():
    """A worker left running against an inline API is worth seeing, not hiding."""
    worker = _worker()
    result = summarise(
        job_mode="inline", workers=[worker], queues=_queues(workers=[worker])
    )

    assert result["verdict"] == VERDICT_HEALTHY
    assert result["worker_count"] == 1


def test_a_broker_error_outranks_every_other_verdict():
    """Depth is unknown, so "no backlog" would be a guess presented as a fact."""
    result = summarise(
        job_mode="celery",
        workers=[],
        queues=attach_consumers(QUEUES, {name: None for name in QUEUES}, []),
        broker_error="ConnectionError: timed out",
    )

    assert result["verdict"] == VERDICT_BROKER_UNREACHABLE
    assert result["healthy"] is False
    assert result["total_queued"] is None
    assert all(queue["depth"] is None for queue in result["queues"])


def test_an_unknown_depth_is_not_counted_as_empty():
    queue = QueueSnapshot(name="ingestion", depth=None, consumers=("celery@host",))

    assert queue.has_backlog is False
    assert queue.is_consumed is True


def test_active_tasks_make_a_busy_worker_distinguishable_from_a_dead_one():
    """Queues can be empty because a worker already picked the work up."""
    worker = _worker(active=["restructure_pending_draft_task"])
    result = summarise(
        job_mode="celery", workers=[worker], queues=_queues(workers=[worker])
    )

    assert result["verdict"] == VERDICT_BUSY
    assert result["workers"][0]["active_task_count"] == 1
    assert result["workers"][0]["active_task_names"] == [
        "restructure_pending_draft_task"
    ]


# --- what must never appear in the response -----------------------------------


def test_the_response_carries_no_credentials_or_task_arguments():
    """Task arguments carry draft ids, user ids and extracted document text.

    "Is work moving" is answered by names and counts, so nothing else is collected.
    """
    worker = _worker(active=["restructure_pending_draft_task"])
    result = summarise(
        job_mode="celery",
        workers=[worker],
        queues=_queues({"ingestion": 2}, workers=[worker]),
        broker_error=None,
    )

    rendered = repr(result)
    # A bare "@" is NOT in this list: a Celery node is named `celery@hostname`, and
    # which worker answered is exactly what an operator needs. What must never appear
    # is a credential, which in a broker URL reaches us as `user:password@host`.
    for forbidden in ("redis://", "rediss://", "password", "devredis", ":@"):
        assert forbidden not in rendered
    assert "args" not in rendered and "kwargs" not in rendered


def test_the_broker_error_is_a_message_and_never_the_url():
    """The broker URL carries the Redis password; the exception text does not."""
    result = summarise(
        job_mode="celery",
        workers=[],
        queues=_queues(),
        broker_error="ConnectionError: Error 111 connecting to redis:6379",
    )

    assert "redis://" not in repr(result)
    assert result["broker_error"] == (
        "ConnectionError: Error 111 connecting to redis:6379"
    )


def test_the_endpoint_collects_names_only_and_bounds_every_call():
    """Source-level, because the collection happens inside the route handler.

    Two properties worth pinning there: no task argument is ever read off a reply, and
    both the control API and the broker get an explicit timeout -- this endpoint is
    reached while something is already wrong.
    """
    from src.api.routers import governance

    source = inspect.getsource(governance.get_worker_diagnostics)

    assert '"name"' in source
    assert ".get(\"args\")" not in source and ".get(\"kwargs\")" not in source
    assert "control_timeout" in source
    assert "socket_connect_timeout" in source and "socket_timeout" in source
    # Synchronous kombu control calls must not block the event loop for the whole
    # timeout while every other request waits.
    assert "asyncio.to_thread" in source


def test_the_endpoint_requires_global_governance_read():
    """Queue and worker topology is operator-only."""
    from src.api.routers import governance

    source = inspect.getsource(governance.get_worker_diagnostics)

    assert 'require_permission("governance.read", scope="global")' in source
