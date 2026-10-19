"""Is anything actually CONSUMING the Celery queues?

An uploaded document can sit at "preparing" forever while the UI and `/health/ready` both
look healthy, because the API and the Celery worker are separate services: the API
answering a readiness probe says nothing about whether a worker exists to drain a queue.
On develop the worker runs `min_count = 0` with `desired_count` under `ignore_changes`
(infra/live/develop/main.tf), so "no worker running" is a reachable steady state — and
until this module there was no way for an operator to see it. The queue simply grew, and
every retry enqueued into a queue nobody was draining.

The verdict is deliberately about CONSUMPTION rather than liveness, because the two
failures look identical from outside and need opposite responses:

    a deep queue WITH a live consumer          -> slow, wait
    a queue with a backlog and NO consumer     -> broken, scale the worker up

Nothing here reports task arguments. They carry draft ids, user ids and extracted
document text, and this endpoint exists to answer "is work moving", which names and
counts answer completely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: No worker is expected in this mode -- the API runs jobs in-process -- so an empty
#: worker list is correct rather than broken.
INLINE_JOB_MODE = "inline"

# Verdicts, ordered by how much they demand of an operator.
VERDICT_BROKER_UNREACHABLE = "broker_unreachable"
VERDICT_NO_CONSUMER = "no_consumer"
VERDICT_QUEUE_UNCONSUMED = "queue_unconsumed"
VERDICT_INLINE_MODE = "inline_mode"
VERDICT_BUSY = "busy"
VERDICT_HEALTHY = "healthy"

_DETAIL = {
    VERDICT_BROKER_UNREACHABLE: (
        "The broker could not be reached, so queue depth is unknown and no task can be "
        "enqueued or consumed."
    ),
    VERDICT_NO_CONSUMER: (
        "No Celery worker answered. Nothing is consuming any queue, so queued work will "
        "not run and a retry only enqueues again. Check that the worker service is "
        "running and scaled above zero."
    ),
    VERDICT_QUEUE_UNCONSUMED: (
        "A queue has queued work but no worker consuming it. That work will never run "
        "until a worker subscribes to the queue."
    ),
    VERDICT_INLINE_MODE: (
        "JOB_MODE is inline: the API runs jobs in-process and no Celery worker is "
        "expected."
    ),
    VERDICT_BUSY: (
        "Work is queued and a worker is consuming it. This is a backlog, not a fault."
    ),
    VERDICT_HEALTHY: "A worker is consuming every queue and nothing is backed up.",
}


@dataclass(frozen=True)
class WorkerSnapshot:
    """One worker that answered a ping.

    `active_task_names` is names only, deliberately: see the module docstring.
    """

    name: str
    queues: tuple[str, ...] = ()
    active_task_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "queues": sorted(self.queues),
            "active_task_count": len(self.active_task_names),
            # Sorted rather than ordered-by-start: this is a census of what is running,
            # and a stable order makes two consecutive calls comparable by eye.
            "active_task_names": sorted(self.active_task_names),
        }


@dataclass(frozen=True)
class QueueSnapshot:
    """One routed queue and how much is waiting on it.

    `depth` is None when the broker could not answer for this queue; that is reported as
    unknown rather than as zero, because zero is the answer an operator acts on.
    """

    name: str
    depth: int | None = None
    consumers: tuple[str, ...] = field(default=())

    @property
    def has_backlog(self) -> bool:
        return bool(self.depth)

    @property
    def is_consumed(self) -> bool:
        return bool(self.consumers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "consumers": sorted(self.consumers),
            "consumed": self.is_consumed,
        }


def attach_consumers(
    queue_names: list[str],
    depths: dict[str, int | None],
    workers: list[WorkerSnapshot],
) -> list[QueueSnapshot]:
    """Pair every routed queue with its depth and the workers subscribed to it."""
    return [
        QueueSnapshot(
            name=name,
            depth=depths.get(name),
            consumers=tuple(worker.name for worker in workers if name in worker.queues),
        )
        for name in queue_names
    ]


def summarise(
    *,
    job_mode: str,
    workers: list[WorkerSnapshot],
    queues: list[QueueSnapshot],
    broker_error: str | None = None,
) -> dict[str, Any]:
    """Reduce a worker/queue census to one verdict an operator can act on."""
    inline = (job_mode or "").strip().lower() == INLINE_JOB_MODE
    starved = [queue for queue in queues if queue.has_backlog and not queue.is_consumed]
    # Reported whatever the depth: a queue nobody consumes is a latent version of the
    # same fault, and an operator wants to know before something lands on it.
    unconsumed = [queue.name for queue in queues if not queue.is_consumed]
    backlog = sum(queue.depth or 0 for queue in queues)

    if broker_error:
        verdict = VERDICT_BROKER_UNREACHABLE
    elif inline and not workers:
        verdict = VERDICT_INLINE_MODE
    elif not workers:
        # Failing even with every queue empty. Nothing will run -- including the beat
        # schedule -- so this is a fault that has not yet become visible, not an idle
        # system.
        verdict = VERDICT_NO_CONSUMER
    elif starved:
        verdict = VERDICT_QUEUE_UNCONSUMED
    elif backlog or any(worker.active_task_names for worker in workers):
        verdict = VERDICT_BUSY
    else:
        verdict = VERDICT_HEALTHY

    return {
        "verdict": verdict,
        "healthy": verdict in {VERDICT_INLINE_MODE, VERDICT_BUSY, VERDICT_HEALTHY},
        "detail": _DETAIL[verdict],
        "job_mode": job_mode,
        "worker_count": len(workers),
        "workers": [worker.as_dict() for worker in workers],
        "queues": [queue.as_dict() for queue in queues],
        "total_queued": backlog if broker_error is None else None,
        "queues_without_consumer": sorted(unconsumed),
        "queues_with_backlog_and_no_consumer": sorted(queue.name for queue in starved),
        # The message, never the URL: the broker URL carries the Redis password.
        "broker_error": broker_error,
    }
