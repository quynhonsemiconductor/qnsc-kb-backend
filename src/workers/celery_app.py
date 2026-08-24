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

celery_app.conf.update(
    broker_use_ssl=_ssl_options if _use_ssl else None,
    redis_backend_use_ssl=_ssl_options if _use_ssl else None,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=["src.workers.tasks"],
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
        "escalate-overdue-drafts": {
            "task": "escalate_overdue_drafts",
            "schedule": 21600.0,
        },
    },
)
