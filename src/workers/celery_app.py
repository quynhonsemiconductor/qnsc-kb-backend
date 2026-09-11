import ssl

from celery import Celery
from src.core.config import settings

celery_app = Celery(
    "qnsc_kb_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
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
_TASK_SOFT_TIME_LIMIT = 3300
_TASK_HARD_TIME_LIMIT = 3600

celery_app.conf.update(
    broker_use_ssl=_ssl_options if _use_ssl else None,
    redis_backend_use_ssl=_ssl_options if _use_ssl else None,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=["src.workers.tasks"],
    broker_transport_options={"visibility_timeout": _TASK_HARD_TIME_LIMIT + 300},
    # Acknowledge on completion, not on receipt. With early acks a worker killed mid-sync
    # loses the message entirely and the durable queue row stays "running" until the stale
    # sweep notices; with late acks the broker replays it.
    task_acks_late=True,
    # Late acks make prefetching dangerous: a prefetched message is unacked while it waits
    # behind a long-running task, so with the default multiplier of four it can exceed the
    # visibility timeout and be redelivered before it has even started. One at a time.
    worker_prefetch_multiplier=1,
    task_soft_time_limit=_TASK_SOFT_TIME_LIMIT,
    task_time_limit=_TASK_HARD_TIME_LIMIT,
    # The worker consumes celery,ingestion,connectors,permissions (Dockerfile), but a task
    # goes to the default queue unless it is routed. Without this map three of those four
    # queues were permanently empty and every job — a 300 s restructure next to a 30 s
    # outbox replay — contended for the same slots on `celery`.
    task_routes={
        "generate_embeddings_task": {"queue": "ingestion"},
        "reprocess_index_job_task": {"queue": "ingestion"},
        "restructure_pending_draft_task": {"queue": "ingestion"},
        "delete_article_chunks_task": {"queue": "ingestion"},
        "recover_stale_index_reprocess_jobs": {"queue": "ingestion"},
        "sync_cloud_connector_task": {"queue": "connectors"},
        "schedule_cloud_connector_syncs": {"queue": "connectors"},
        "reconcile_cloud_connectors": {"queue": "connectors"},
        "dispatch_pending_sync_requests": {"queue": "connectors"},
        "renew_webhook_subscriptions": {"queue": "connectors"},
        "recompute_permissions_task": {"queue": "permissions"},
    },
    beat_schedule={
        "replay-domain-outbox": {
            "task": "replay_outbox_task",
            "schedule": 30.0,
        },
        "poll-cloud-connectors": {
            "task": "schedule_cloud_connector_syncs",
            "schedule": 600.0,
        },
        "reconcile-cloud-connectors": {
            "task": "reconcile_cloud_connectors",
            "schedule": settings.CONNECTOR_RECONCILE_INTERVAL_MINUTES * 60.0,
        },
        # The interval is a setting because it is the floor on webhook-to-index latency
        # for anything the request-time dispatch missed. Hard-coding 30 here left
        # CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS declared, documented and dead.
        "dispatch-pending-sync-requests": {
            "task": "dispatch_pending_sync_requests",
            "schedule": float(settings.CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS),
        },
        # Graph drive subscriptions can expire quickly and a lapsed one fails silently.
        # Renew frequently against a horizon so several attempts land before expiry.
        "renew-webhook-subscriptions": {
            "task": "renew_webhook_subscriptions",
            "schedule": 600.0,
        },
        "prune-operational-metrics": {
            "task": "prune_operational_metrics",
            "schedule": 86400.0,
        },
        "cleanup-orphaned-source-objects": {
            "task": "cleanup_orphaned_source_objects",
            "schedule": 86400.0,
        },
        "deliver-notification-queue": {
            "task": "deliver_notification_queue",
            "schedule": 30.0,
        },
        "verify-review-deadlines": {
            "task": "verify_review_deadlines",
            "schedule": 86400.0,
        },
        "run-approval-agent": {
            "task": "run_approval_agent",
            "schedule": float(settings.APPROVAL_AGENT_RUN_INTERVAL_SECONDS),
        },
        "escalate-overdue-drafts": {
            "task": "escalate_overdue_drafts",
            "schedule": 21600.0,
        },
        # A reprocess job is a long loop of independent article re-indexes with no
        # heartbeat, so a worker that dies mid-run leaves the row "running" forever and
        # the operator sees a progress bar that never moves again.
        "recover-stale-index-reprocess-jobs": {
            "task": "recover_stale_index_reprocess_jobs",
            "schedule": 900.0,
        },
        # Nightly, same cadence as the other corpus-wide maintenance sweeps above
        # (prune-operational-metrics, cleanup-orphaned-source-objects): topical matches
        # shift slowly as the corpus grows, so there is no value in running this more
        # often, only cost.
        "link-related-articles": {
            "task": "link_related_articles",
            "schedule": 86400.0,
        },
    },
)
