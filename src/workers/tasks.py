import asyncio
import uuid
import structlog
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from sqlalchemy import delete, select
from celery.signals import worker_ready, worker_process_init, task_prerun, task_failure
from src.workers.celery_app import celery_app
from src.api.deps import SessionLocal, engine, set_database_context
from src.repositories.article import ArticleRepository
from src.repositories.chunk import ChunkRepository
from src.domain.permissions import PermissionService
from src.core.config import settings
from src.models.ops import ApiRequestMetric, OutboxEvent, IndexReprocessJob, NotificationQueue
from src.models.governance import PendingDraft
from src.models.user import User
from src.services.email import get_email_sender
from src.models.ai import AiCache
from src.workers.loop import reset_worker_loop, sync_run

logger = structlog.get_logger()


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Drop the parent process's asyncpg pool after Celery forks."""
    reset_worker_loop()
    engine.sync_engine.dispose(close=False)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    logger.info("Celery worker ready", worker=str(sender))


@task_prerun.connect
def task_prerun_handler(task_id=None, task=None, **kwargs):
    logger.info(
        "Celery task started", task_name=getattr(task, "name", None), task_id=task_id
    )


@task_failure.connect
def task_failure_handler(task_id=None, exception=None, sender=None, **kwargs):
    logger.error(
        "Celery task failed",
        task_name=getattr(sender, "name", None),
        task_id=task_id,
        error=str(exception),
    )


@celery_app.task(name="handle_domain_event_task")
def handle_domain_event_task(event_type: str, payload: dict):
    logger.info(
        "Celery event worker received event", event_type=event_type, payload=payload
    )

    if event_type in ["ArticlePublished", "ArticleUpdated"]:
        article_id = payload.get("article_id")
        if article_id:
            generate_embeddings_task.delay(article_id)

    elif event_type == "PermissionChanged":
        article_id = payload.get("article_id")
        if article_id:
            recompute_permissions_task.delay(article_id)

    elif event_type == "ArticleDeleted":
        article_id = payload.get("article_id")
        if article_id:
            delete_article_chunks_task.delay(article_id)

    outbox_id = payload.get("_outbox_id")
    if outbox_id:

        async def mark_dispatched():
            async with SessionLocal() as db:
                await set_database_context(db, None, True)
                event = await db.get(OutboxEvent, uuid.UUID(outbox_id))
                if event:
                    event.status = "dispatched"
                    event.last_error = None
                    await db.commit()

        sync_run(mark_dispatched())


@celery_app.task(name="replay_outbox_task")
def replay_outbox_task():
    async def replay():
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            result = await db.execute(
                select(OutboxEvent)
                .where(
                    OutboxEvent.status.in_(["pending", "failed", "processing"]),
                    OutboxEvent.next_attempt_at <= datetime.utcnow(),
                )
                .order_by(OutboxEvent.created_at)
                .limit(100)
            )
            events = result.scalars().all()
            for event in events:
                event.status = "processing"
                event.attempts += 1
                event.next_attempt_at = datetime.utcnow() + timedelta(
                    minutes=min(30, 2 ** min(event.attempts, 5))
                )
                await db.commit()
                payload = dict(event.payload)
                payload["_outbox_id"] = str(event.id)
                handle_domain_event_task.delay(event.event_type, payload)

    sync_run(replay())


@celery_app.task(name="prune_operational_metrics")
def prune_operational_metrics() -> None:
    """Bound telemetry growth and remove physically expired answer caches."""

    async def prune() -> None:
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            cutoff = datetime.utcnow() - timedelta(days=settings.METRICS_RETENTION_DAYS)
            await db.execute(
                ApiRequestMetric.__table__.delete().where(
                    ApiRequestMetric.created_at < cutoff
                )
            )
            # Cache answers can contain authorized document passages. Their
            # six-hour expiry must remove storage as well as disable reads.
            await db.execute(
                AiCache.__table__.delete().where(AiCache.expires_at < datetime.utcnow())
            )
            await db.commit()

    sync_run(prune())


@celery_app.task(name="deliver_notification_queue")
def deliver_notification_queue() -> None:
    """Deliver queued email notifications and persist an auditable outcome."""
    async def deliver() -> None:
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            rows = (await db.execute(
                select(NotificationQueue)
                .where(NotificationQueue.type == "email", NotificationQueue.status.in_(["pending", "failed"]))
                .order_by(NotificationQueue.created_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )).scalars().all()
            sender = get_email_sender()
            for row in rows:
                row.attempts += 1
                payload = row.payload or {}
                try:
                    await sender.send(
                        to=str(payload["to"]),
                        subject=str(payload.get("subject") or "QNSC notification"),
                        text=str(payload.get("text") or ""),
                        html=str(payload["html"]) if payload.get("html") else None,
                    )
                    row.status = "sent"
                    row.sent_at = datetime.utcnow()
                    row.last_error = None
                except Exception as exc:
                    row.status = "failed"
                    row.last_error = str(exc)[:1000]
            await db.commit()

    sync_run(deliver())


@celery_app.task(name="verify_review_deadlines")
def verify_review_deadlines() -> None:
    async def verify() -> None:
        from src.domain.review import ReviewService
        from src.models.article import Article
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            domains = (await db.execute(select(Article.company_domain).distinct())).scalars().all()
            for domain in domains:
                await ReviewService(ArticleRepository(db)).verify_review_deadlines(str(domain))
    sync_run(verify())


@celery_app.task(name="escalate_overdue_drafts")
def escalate_overdue_drafts() -> None:
    """Escalate drafts past the approval SLA to their submitter and approver."""
    async def escalate() -> None:
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            cutoff = datetime.utcnow() - timedelta(days=settings.REVIEW_SLA_DAYS)
            drafts = (await db.execute(select(PendingDraft).where(
                PendingDraft.status == "pending",
                PendingDraft.assigned_at.is_not(None),
                PendingDraft.assigned_at < cutoff,
            ).limit(500))).scalars().all()
            users = {item.id: item for item in (await db.execute(select(User).where(User.id.in_({uid for draft in drafts for uid in (draft.created_by, draft.assigned_approver_id) if uid})))).scalars().all()} if drafts else {}
            for draft in drafts:
                for user_id in {draft.created_by, draft.assigned_approver_id} - {None}:
                    recipient = users.get(user_id)
                    if not recipient:
                        continue
                    recent = await db.scalar(select(NotificationQueue.id).where(
                        NotificationQueue.recipient_user_id == recipient.id,
                        NotificationQueue.type == "email",
                        NotificationQueue.created_at >= datetime.utcnow() - timedelta(hours=24),
                        NotificationQueue.payload["event"].as_string() == "draft_overdue",
                        NotificationQueue.payload["draft_id"].as_string() == str(draft.id),
                    ).limit(1))
                    if recent:
                        continue
                    db.add(NotificationQueue(
                        recipient_user_id=recipient.id,
                        type="email",
                        payload={"event": "draft_overdue", "draft_id": str(draft.id), "to": recipient.email,
                                 "subject": f"Approval overdue: {draft.title}",
                                 "text": f"The draft '{draft.title}' has been awaiting approval beyond the {settings.REVIEW_SLA_DAYS}-day SLA."},
                    ))
            await db.commit()
    sync_run(escalate())


async def _run_orphan_source_cleanup() -> int:
    """Delete old private R2 objects that have no database reference."""
    from src.domain.source_storage import delete_source, list_source_objects
    from src.models.article import DocumentSource
    from src.models.connectors import DocumentVersion
    from src.models.governance import PendingDraft

    if not settings.SOURCE_STORAGE_BUCKET:
        logger.warning("Skipping R2 orphan sweep because no bucket is configured")
        return 0

    objects = await asyncio.to_thread(list_source_objects)
    cutoff = datetime.now(timezone.utc) - timedelta(
        hours=max(1, settings.SOURCE_ORPHAN_GRACE_HOURS)
    )
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        referenced_keys: set[str] = set()
        for model in (PendingDraft, DocumentSource, DocumentVersion):
            result = await db.execute(
                select(model.storage_key).where(model.storage_key.is_not(None))
            )
            referenced_keys.update(
                str(storage_key)
                for storage_key in result.scalars().all()
                if storage_key
            )

    deleted_count = 0
    for item in objects:
        storage_key = item.get("storage_key")
        last_modified = item.get("last_modified")
        if not storage_key or not isinstance(last_modified, datetime):
            continue
        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)
        if last_modified >= cutoff or storage_key in referenced_keys:
            continue
        try:
            await asyncio.to_thread(delete_source, storage_key)
            deleted_count += 1
        except Exception:
            logger.exception(
                "R2 orphan source deletion failed", storage_key=storage_key
            )
    logger.info(
        "R2 orphan source sweep completed",
        scanned=len(objects),
        deleted=deleted_count,
        referenced=len(referenced_keys),
    )
    return deleted_count


@celery_app.task(
    name="cleanup_orphaned_source_objects",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def cleanup_orphaned_source_objects() -> int:
    """Periodic safety net for objects left by failed ingestion transactions."""
    return sync_run(_run_orphan_source_cleanup())


async def run_restructure_pending_draft(
    draft_id_str: str, company_domain: str, user_id_str: str
) -> None:
    """Format a stored upload, on the caller's async event loop.

    Shared by the Celery task and inline-mode dispatch so the feature works
    identically in both deployment job modes.
    """
    from src.domain.content_restructure import restructure_document
    from src.models import User
    from src.models.governance import AuditLog, DraftCandidate, PendingDraft
    from src.models.user import Department
    from src.domain.department_routing import route_document_candidates
    from src.domain.llm_config import load_runtime_config
    from src.repositories.feature_flags import FeatureFlagRepository

    async with SessionLocal() as db:
        # This is an internal task for a draft that was already authorized
        # and persisted by the request. Keep the tenant context explicit.
        await set_database_context(
            db,
            company_domain,
            True,
            user_id=user_id_str,
            global_governance_access=True,
        )
        # Celery has its own Python process and does not run API startup;
        # load the administrator's saved provider before calling the LLM.
        await load_runtime_config(db)
        draft = await db.get(PendingDraft, uuid.UUID(draft_id_str))
        if not draft or draft.status != "pending":
            return
        user = await db.get(User, uuid.UUID(user_id_str))
        enabled = bool(
            settings.RESTRUCTURE_ENABLED
            and user
            and await FeatureFlagRepository(db).is_enabled(
                "ai.document_restructure", user
            )
        )
        source_text = draft.summary or "\n\n".join(
            str(page.get("text", ""))
            for page in (draft.page_texts or [])
            if page.get("text")
        )
        draft.restructure_status = "processing"
        draft.restructure_error = None
        draft.restructure_candidate_md = None
        draft.restructure_decision = "not_reviewed"
        await db.commit()
        try:
            departments = list(
                (
                    await db.execute(
                        select(Department).where(
                            Department.company_domain == draft.company_domain,
                            Department.active.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )
            result = await restructure_document(
                draft.title,
                source_text,
                enabled=enabled,
                department_descriptions=[
                    (department.name, department.description or "")
                    for department in departments
                ],
            )
            draft.restructured_body_md = result.body_md
            draft.restructure_candidate_md = result.candidate_body_md
            draft.restructure_decision = (
                "pending_review"
                if result.candidate_body_md
                else ("ai_ready" if result.status == "llm" else "lossless_ready")
            )
            draft.restructure_status = result.status
            draft.restructure_model = result.model
            draft.restructure_error = result.error
            # Batch review operates on the formatted reading view, not raw extraction.
            # Recreate candidates only after formatting has completed, then use the
            # active department descriptions to choose an editable default route.
            await db.execute(
                delete(DraftCandidate).where(DraftCandidate.draft_id == draft.id)
            )
            for item in route_document_candidates(
                draft.title, result.body_md, departments
            ):
                db.add(
                    DraftCandidate(
                        draft_id=draft.id,
                        **item,
                    )
                )
            db.add(
                AuditLog(
                    user_id=user.id if user else None,
                    action="restructure",
                    target_type="draft",
                    target_id=str(draft.id),
                )
            )
            await db.commit()
            logger.info(
                "Pending draft AI formatting completed",
                draft_id=str(draft.id),
                restructure_status=result.status,
                restructure_model=result.model,
            )
        except Exception as exc:
            draft.restructured_body_md = source_text
            draft.restructure_candidate_md = None
            draft.restructure_decision = "lossless_ready"
            draft.restructure_status = "fallback_formatting"
            draft.restructure_model = "lossless-markdown"
            draft.restructure_error = f"AI formatting failed ({str(exc) or 'unknown error'}); retry from Pending Drafts."
            await db.commit()
            logger.exception(
                "Pending draft AI formatting failed", draft_id=str(draft.id)
            )


@celery_app.task(
    name="restructure_pending_draft_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def restructure_pending_draft_task(
    draft_id_str: str, company_domain: str, user_id_str: str
):
    """Format a stored upload after the upload request has completed."""
    sync_run(run_restructure_pending_draft(draft_id_str, company_domain, user_id_str))


def dispatch_restructure_pending_draft(
    draft_id_str: str, company_domain: str, user_id_str: str
) -> bool:
    """Dispatch AI draft formatting according to the deployment job mode.

    Returns False only when the Celery dispatch itself failed (e.g. missing
    broker), so callers can mark the draft as fallback-formatted.
    """
    if settings.JOB_MODE == "celery":
        try:
            restructure_pending_draft_task.delay(
                draft_id_str, company_domain, user_id_str
            )
            return True
        except Exception:
            logger.exception(
                "Could not queue source restructuring", draft_id=draft_id_str
            )
            return False

    async def _run() -> None:
        try:
            await run_restructure_pending_draft(
                draft_id_str, company_domain, user_id_str
            )
        except Exception:
            logger.exception(
                "Inline source restructuring failed", draft_id=draft_id_str
            )

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        # No running loop (script context): run to completion inline.
        sync_run(_run())
    return True


@celery_app.task(name="generate_embeddings_task")
def generate_embeddings_task(article_id_str: str):
    article_id = uuid.UUID(article_id_str)
    logger.info("Generating embeddings for article", article_id=article_id)

    async def process():
        from src.domain.indexing import index_article

        await index_article(article_id)

    try:
        sync_run(process())
    except Exception:
        logger.exception("Embedding generation task failed", article_id=article_id)
        raise


async def run_reprocess_index_job(job_id_str: str) -> None:
    """Re-index a durable article set on the caller's async event loop."""
    from src.models.article import Article
    from src.domain.indexing import index_article

    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        job = await db.get(IndexReprocessJob, uuid.UUID(job_id_str))
        if not job:
            return
        ids = [uuid.UUID(str(item)) for item in (job.target_article_ids or [])]
        stmt = select(Article.id).where(
            Article.company_domain == job.company_domain,
            Article.status == "published",
            Article.lifecycle_status == "active",
        )
        if ids:
            stmt = stmt.where(Article.id.in_(ids))
        article_ids = [item for item in (await db.execute(stmt)).scalars().all()]
        job.status = "running"
        job.total = len(article_ids)
        job.completed = 0
        job.failed = 0
        job.last_error = None
        job.started_at = datetime.utcnow()
        await db.commit()
    for article_id in article_ids:
        try:
            await index_article(article_id)
            async with SessionLocal() as progress_db:
                await set_database_context(progress_db, None, True)
                progress = await progress_db.get(
                    IndexReprocessJob, uuid.UUID(job_id_str)
                )
                if progress:
                    progress.completed += 1
                    await progress_db.commit()
        except Exception as exc:
            async with SessionLocal() as progress_db:
                await set_database_context(progress_db, None, True)
                progress = await progress_db.get(
                    IndexReprocessJob, uuid.UUID(job_id_str)
                )
                if progress:
                    progress.failed += 1
                    progress.last_error = str(exc)[:2000]
                    await progress_db.commit()
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        job = await db.get(IndexReprocessJob, uuid.UUID(job_id_str))
        if job:
            job.status = "failed" if job.failed else "completed"
            job.completed_at = datetime.utcnow()
            await db.commit()


@celery_app.task(name="reprocess_index_job_task")
def reprocess_index_job_task(job_id_str: str):
    """Celery adapter for the shared async reprocess implementation."""
    sync_run(run_reprocess_index_job(job_id_str))


@celery_app.task(name="recompute_permissions_task")
def recompute_permissions_task(article_id_str: str):
    article_id = uuid.UUID(article_id_str)
    logger.info(
        "Recomputing permission bitmask snapshot on chunks", article_id=article_id
    )

    async def process():
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            article_repo = ArticleRepository(db)
            chunk_repo = ChunkRepository(db)

            article = await article_repo.get_by_id(article_id)
            if not article:
                logger.warn(
                    "Article not found, skipping permission recomputation",
                    article_id=article_id,
                )
                return

            bitmap = PermissionService.calculate_article_bitmask(article)
            await chunk_repo.update_permissions(
                article_id=article_id,
                bitmap=bitmap,
                sensitivity=article.sensitivity,
                visibility=article.visibility,
                dept=article.dept,
            )
            logger.info(
                "Permission bitmap updated successfully",
                article_id=article_id,
                bitmap=bitmap,
            )

    sync_run(process())


@celery_app.task(name="delete_article_chunks_task")
def delete_article_chunks_task(article_id_str: str):
    article_id = uuid.UUID(article_id_str)
    logger.info("Deleting chunks for removed article", article_id=article_id)

    async def process():
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            chunk_repo = ChunkRepository(db)
            await chunk_repo.delete_by_article_id(article_id)
            logger.info("Article chunks deleted successfully", article_id=article_id)

    sync_run(process())


@celery_app.task(
    name="sync_cloud_connector_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def sync_cloud_connector_task(connector_id_str: str, job_id_str: str):
    """Run an idempotent provider delta sync from a durable cursor."""

    async def process():
        from src.domain.cloud_sync import sync_cloud_connector
        from src.models.ops import Connector, ConnectorJob

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            connector = await db.get(Connector, uuid.UUID(connector_id_str))
            job = await db.get(ConnectorJob, uuid.UUID(job_id_str))
            if not connector or not job:
                return
            await sync_cloud_connector(db, connector, job)

    sync_run(process())


@celery_app.task(name="schedule_cloud_connector_syncs")
def schedule_cloud_connector_syncs():
    """Polling fallback: wake every active cloud connector periodically."""

    async def process():
        from datetime import timedelta
        from src.models.ops import Connector, ConnectorJob
        from src.models.connectors import SourceScope

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            connectors = (
                (
                    await db.execute(
                        select(Connector).where(
                            Connector.system.in_(["sharepoint", "google_drive"]),
                            Connector.status.in_(["active", "error"]),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for connector in connectors:
                sync_mode = (connector.config_json or {}).get("sync_mode", "daily")
                if sync_mode == "manual":
                    continue
                interval = (
                    timedelta(days=1) if sync_mode == "daily" else timedelta(minutes=10)
                )
                if (
                    connector.last_sync
                    and connector.last_sync > datetime.utcnow() - interval
                ):
                    continue
                selected = (
                    await db.execute(
                        select(SourceScope.id)
                        .where(
                            SourceScope.connector_id == connector.id,
                            SourceScope.selected.is_(True),
                        )
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if not selected:
                    continue
                recent = (
                    await db.execute(
                        select(ConnectorJob.id)
                        .where(
                            ConnectorJob.connector_id == connector.id,
                            ConnectorJob.status.in_(["queued", "running"]),
                        )
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if recent:
                    continue
                job = ConnectorJob(
                    connector_id=connector.id,
                    requested_by=connector.created_by,
                    status="queued",
                    attempts=0,
                )
                db.add(job)
                await db.commit()
                sync_cloud_connector_task.delay(str(connector.id), str(job.id))

    sync_run(process())


@celery_app.task(name="renew_webhook_subscriptions")
def renew_webhook_subscriptions():
    """Extend provider push subscriptions before they lapse.

    Microsoft Graph subscriptions on a drive are short-lived — create_webhook asks for one
    hour — and a lapsed subscription is not an error anyone sees: the provider simply
    stops calling, the connector keeps reporting `on_update`, and content silently goes
    stale until someone notices the KB is out of date. Nothing renewed them, so
    "real-time sync" lasted exactly one hour from the moment it was switched on.

    Runs every 10 minutes against a 20-minute horizon, so a subscription gets several
    attempts before it expires and one failed run is not enough to drop it.
    """

    async def process():
        from datetime import timedelta

        from src.domain.connector_adapters import ConnectorProviderError, adapter_for
        from src.models.connectors import WebhookSubscription
        from src.models.ops import Connector

        horizon = datetime.utcnow() + timedelta(minutes=20)
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            due = (
                (
                    await db.execute(
                        select(WebhookSubscription).where(
                            WebhookSubscription.active.is_(True),
                            WebhookSubscription.expires_at.isnot(None),
                            WebhookSubscription.expires_at <= horizon,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for subscription in due:
                connector = await db.get(Connector, subscription.connector_id)
                if connector is None or connector.status not in ("active", "error"):
                    continue
                try:
                    renewed = await adapter_for(connector).renew_webhook(
                        subscription.provider_subscription_id
                    )
                except ConnectorProviderError as exc:
                    # The subscription is gone at the provider, or the token no longer
                    # grants it. Deactivate rather than retry forever: the polling loop in
                    # schedule_cloud_connector_syncs still covers this connector, so the
                    # cost is latency, not lost content.
                    logger.warning(
                        "Webhook subscription renewal failed",
                        connector_id=str(connector.id),
                        subscription_id=subscription.provider_subscription_id,
                        error=str(exc),
                    )
                    subscription.active = False
                    continue
                if renewed is None:
                    # Provider cannot extend in place (Google Drive channels). Retire it so
                    # the row does not claim a liveness it no longer has.
                    subscription.active = False
                    continue
                subscription.expires_at = renewed
            await db.commit()

    sync_run(process())
