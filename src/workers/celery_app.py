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
        # Graph drive subscriptions expire an hour after they are created, and a lapsed
        # one fails silently — the provider just stops calling. Every 10 minutes against
        # a 20-minute horizon, so several attempts land before any subscription expires.
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
