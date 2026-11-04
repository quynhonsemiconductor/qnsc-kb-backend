import ssl
import time

from celery import Celery
from celery.schedules import crontab
from celery.signals import (
    task_postrun,
    task_prerun,
    worker_process_init,
    worker_process_shutdown,
)
from src.core import otel_metrics
from src.core.config import settings
from src.lib import log_redaction

# The worker never calls setup_logging(), and it is the process that downloads from SharePoint/OneDrive:
# a pre-authenticated download URL (`tempauth=...`) is a bearer credential and was being logged in full by
# httpx's INFO line. Applies to the worker, beat and anything that imports this module.
log_redaction.install()

celery_app = Celery(
    "qnsc_kb_workers", broker=settings.REDIS_URL, backend=settings.REDIS_URL
)

# Managed caches (ElastiCache/Valkey with encryption in transit) are reached over
# `rediss://`. Celery refuses such a URL outright — "A rediss:// URL must have
# parameter ssl_cert_reqs" — unless the requirement is stated, and it does NOT infer
# a default the way redis-py does. Verifying against the system CA bundle is correct
# for AWS-issued certificates; never relax this to CERT_NONE, which would accept any
# certificate and make the encryption decorative.
#
# Applied only for rediss://, so a plain redis:// (local Compose, CI) is untouched.
_ssl_options = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
_use_ssl = settings.REDIS_URL.startswith("rediss://")

# The longest task in this application is a connector walk (up to MAX_CONNECTOR_FILES
# documents, each extracted and possibly OCR'd) followed by AI restructuring, whose own
# per-call budget is RESTRUCTURE_TIMEOUT_SECONDS of 300 s. Redis has no broker-side ack
# deadline: Celery emulates one with `visibility_timeout`, and its default of one hour
# is a SILENT correctness bug rather than a delay — a task still running when the timer
# expires is redelivered, so a sync walks the same drive twice and an OCR job rasterizes
# the same document twice, concurrently.
#
# So the ordering below is the invariant, not a set of independent knobs:
#   task_soft_time_limit < task_time_limit < visibility_timeout
# The hard limit bounds how long any task can hold its message, and the visibility
# timeout sits above that bound, so redelivery can only ever happen after the worker has
# been killed — which is exactly when it is wanted.
TASK_SOFT_TIME_LIMIT = 3300
TASK_HARD_TIME_LIMIT = 3600
#: How long Redis waits for an ack before it hands a message to another consumer.
VISIBILITY_TIMEOUT = TASK_HARD_TIME_LIMIT + 300

# A periodic task is only worth running while its own next tick has not arrived. beat enqueues on the clock
# whether or not anything is consuming, and the worker is ONE process that also runs hour-long syncs and OCR:
# every 30 s tick that lands behind such a task waits in Redis, and with no expiry the worker replays all of
# them back to back the moment it is free. Measured on the local stack: 100 s without a consumer left 8
# messages queued, none with an expiry, and the restarted worker ran every one. `expires` makes the broker
# side discard a tick that is already stale, so a recovered worker does one pass, not a backlog of them.
#
# Never below this floor, so a worker that is merely a few seconds behind does not skip a tick it could run.
_EXPIRY_FLOOR_SECONDS = 30.0


def periodic(task: str, every: float, schedule=None) -> dict:
    """A beat entry that expires when its next tick would have run anyway.

    `every` is the interval in seconds (for a crontab, the gap between runs), and `schedule` overrides the
    plain interval when the entry fires at a wall-clock time.
    """
    return {
        "task": task,
        "schedule": float(every) if schedule is None else schedule,
        "options": {"expires": max(float(every), _EXPIRY_FLOOR_SECONDS)},
    }


celery_app.conf.update(
    broker_use_ssl=_ssl_options if _use_ssl else None,
    redis_backend_use_ssl=_ssl_options if _use_ssl else None,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=["src.workers.tasks"],
    broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT},
    # Nothing reads a task's return value (dispatch is fire-and-forget; progress lives in the database), so
    # storing one per run only fills Redis: a 30 s beat alone writes thousands of result keys a day.
    task_ignore_result=True,
    # Say so explicitly. Left unset, Celery 5.x warns and retries a broker that is not up yet, and the next
    # major release stops retrying, which turns "Redis started a second after the worker" into a crash loop.
    broker_connection_retry_on_startup=True,
    # Acknowledge on completion, not on receipt. With early acks a worker killed mid-sync
    # loses the message entirely and the durable queue row stays "running" until the stale
    # sweep notices; with late acks the broker replays it.
    task_acks_late=True,
    # Late acks make prefetching dangerous: a prefetched message is unacked while it waits
    # behind a long-running task, so with the default multiplier of four it can exceed the
    # visibility timeout and be redelivered before it has even started. One at a time.
    worker_prefetch_multiplier=1,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_time_limit=TASK_HARD_TIME_LIMIT,
    # The worker consumes every named queue listed in the image command. Document
    # preparation has its own queue so one retry does not sit behind an arbitrarily long
    # FIFO backlog of embedding/OCR work on ``ingestion``. Concurrency stays one because
    # the task is sized for one embedding session; separate queues provide fair pickup,
    # not unsafe parallelism.
    task_routes={
        "generate_embeddings_task": {"queue": "ingestion"},
        "reprocess_index_job_task": {"queue": "ingestion"},
        "restructure_pending_draft_task": {"queue": "document-preparation"},
        "delete_article_chunks_task": {"queue": "ingestion"},
        "recover_stale_index_reprocess_jobs": {"queue": "ingestion"},
        "sync_cloud_connector_task": {"queue": "connectors"},
        "schedule_cloud_connector_syncs": {"queue": "connectors"},
        "reconcile_cloud_connectors": {"queue": "connectors"},
        "dispatch_pending_sync_requests": {"queue": "connectors"},
        "renew_webhook_subscriptions": {"queue": "connectors"},
        "recompute_permissions_task": {"queue": "permissions"},
        "recover_stuck_indexing": {"queue": "ingestion"},
    },
    beat_schedule={
        "replay-domain-outbox": periodic("replay_outbox_task", 30.0),
        "poll-cloud-connectors": periodic("schedule_cloud_connector_syncs", 600.0),
        "reconcile-cloud-connectors": periodic(
            "reconcile_cloud_connectors",
            settings.CONNECTOR_RECONCILE_INTERVAL_MINUTES * 60.0,
        ),
        # The interval is a setting because it is the floor on webhook-to-index latency
        # for anything the request-time dispatch missed. Hard-coding 30 here left
        # CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS declared, documented and dead.
        "dispatch-pending-sync-requests": periodic(
            "dispatch_pending_sync_requests",
            float(settings.CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS),
        ),
        # Graph drive subscriptions can expire quickly and a lapsed one fails silently.
        # Renew frequently against a horizon so several attempts land before expiry.
        "renew-webhook-subscriptions": periodic("renew_webhook_subscriptions", 600.0),
        # A certificate that lapses silently stops all application mail. A wall-clock time, not a 24 h interval:
        # an interval counts from beat's own start (on ephemeral storage), so an environment redeployed more than
        # daily would never reach it. The worker also runs the check each time it starts (tasks.py).
        "check-mail-certificate": periodic(
            "check_mail_certificate", 86400.0, schedule=crontab(hour=1, minute=0)
        ),
        "prune-operational-metrics": periodic("prune_operational_metrics", 86400.0),
        "cleanup-orphaned-source-objects": periodic(
            "cleanup_orphaned_source_objects", 86400.0
        ),
        "deliver-notification-queue": periodic("deliver_notification_queue", 30.0),
        "verify-review-deadlines": periodic("verify_review_deadlines", 86400.0),
        "run-approval-agent": periodic(
            "run_approval_agent", float(settings.APPROVAL_AGENT_RUN_INTERVAL_SECONDS)
        ),
        "escalate-overdue-drafts": periodic("escalate_overdue_drafts", 21600.0),
        # A reprocess job is a long loop of independent article re-indexes with no
        # heartbeat, so a worker that dies mid-run leaves the row "running" forever and
        # the operator sees a progress bar that never moves again.
        "recover-stale-index-reprocess-jobs": periodic(
            "recover_stale_index_reprocess_jobs", 900.0
        ),
        # The same failure for a single article: a child killed by the OOM killer raises nothing in Python, so
        # `index_status` stays "processing" for good and the article drops out of search. See
        # src/domain/index_recovery.py.
        "recover-stuck-indexing": periodic("recover_stuck_indexing", 900.0),
        # Nightly, same cadence as the other corpus-wide maintenance sweeps above
        # (prune-operational-metrics, cleanup-orphaned-source-objects): topical matches
        # shift slowly as the corpus grows, so there is no value in running this more
        # often, only cost.
        "link-related-articles": periodic("link_related_articles", 86400.0),
    },
)


# ── Worker metrics (OTLP) ────────────────────────────────────────────────────
# The worker process had no metrics at all: a queue that stopped draining or a
# task that failed every run was invisible until a user reported it. Record one
# run/failure/duration triple per finished task from Celery's own signals — no
# broker polling, no per-task network call.
#
# CONFIGURE IN THE CHILD, not at worker_ready: the worker uses the prefork pool
# (Dockerfile --concurrency=1), so task signals fire in a forked child. A
# MeterProvider created in the parent has an exporter THREAD, and threads do
# not survive fork — the child would inherit a dead provider and silently drop
# every point. worker_process_init runs once per child, after the fork.
#
# `--pool=solo` has NO child process and does not emit worker_process_init, so
# task_prerun also calls configure_metrics (idempotent once a provider exists).
# kb's deployments use the Dockerfile's prefork `--concurrency=1`; the fallback
# means a solo-pool dev/local run still reports instead of silently no-op'ing.
_task_started_at: dict[str, float] = {}


@worker_process_init.connect
def _init_worker_metrics(**_kwargs: object) -> None:
    otel_metrics.configure_metrics(process_instance_id=True)


@worker_process_shutdown.connect
def _flush_worker_metrics(**_kwargs: object) -> None:
    otel_metrics.shutdown_metrics()


@task_prerun.connect
def _task_started(task_id: str | None = None, **_kwargs: object) -> None:
    # Idempotent: covers --pool=solo (and any pool where the init signal did
    # not fire for this process) without creating a second provider.
    otel_metrics.configure_metrics(process_instance_id=True)
    if task_id:
        _task_started_at[task_id] = time.monotonic()


@task_postrun.connect
def _task_finished(
    sender: object = None,
    task_id: str | None = None,
    state: str | None = None,
    **_kwargs: object
) -> None:
    started = _task_started_at.pop(task_id, None) if task_id else None
    if started is None:
        return
    otel_metrics.record_task_finished(
        getattr(sender, "name", None),
        state,
        (time.monotonic() - started) * 1000.0,
    )
