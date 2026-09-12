import asyncio
import uuid
import structlog
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from sqlalchemy import JSON, DateTime, Uuid, delete, exists, literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from celery.signals import worker_ready, worker_process_init, task_prerun, task_failure
from src.workers.celery_app import celery_app
from src.api.deps import SessionLocal, engine, set_database_context
from src.repositories.article import ArticleRepository
from src.repositories.chunk import ChunkRepository
from src.core.config import settings
from src.models.ops import ApiRequestMetric, OutboxEvent, IndexReprocessJob, NotificationQueue, ConnectorJob, Connector
from src.models.connectors import SyncRequest
from src.domain.connector_providers import (
    REMOTE_PROVIDERS,
    cursor_type as provider_cursor_type,
)
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


# Shared with EventBus.recover_outbox_once, which is the inline-mode equivalent of this
# task: both walk the same table, so a poison event must die at the same attempt count
# whichever one reaches it first.
OUTBOX_MAX_ATTEMPTS = 5


@celery_app.task(name="replay_outbox_task")
def replay_outbox_task():
    async def replay():
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            # Claimed under a row lock, and the whole batch is claimed in ONE transaction
            # before anything is dispatched. Selecting unlocked rows and committing per
            # row let a second beat tick (or the inline recovery loop) read the same
            # "processing" row between the commit and the dispatch, so the event was
            # handled twice. skip_locked means a concurrent tick takes different rows
            # instead of blocking, which is the same claim pattern as
            # claim_sync_request and deliver_notification_queue.
            events = (await db.execute(
                select(OutboxEvent)
                .where(
                    OutboxEvent.status.in_(["pending", "failed", "processing"]),
                    OutboxEvent.next_attempt_at <= datetime.utcnow(),
                )
                .order_by(OutboxEvent.created_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )).scalars().all()
            dispatchable: list[tuple[str, dict]] = []
            for event in events:
                if event.attempts >= OUTBOX_MAX_ATTEMPTS:
                    # Terminal, and the row is kept for inspection. Without this an event
                    # whose handler always raises was re-selected and re-dispatched on
                    # every tick, forever.
                    event.status = "dead"
                    continue
                event.status = "processing"
                event.attempts += 1
                event.next_attempt_at = datetime.utcnow() + timedelta(
                    minutes=min(30, 2 ** min(event.attempts, 5))
                )
                payload = dict(event.payload)
                payload["_outbox_id"] = str(event.id)
                dispatchable.append((event.event_type, payload))
            await db.commit()
            # After the commit that releases the locks: a task handed to Celery before
            # its claim is durable can be executed, and finish, before the claiming
            # transaction is visible to anyone else.
            for event_type, payload in dispatchable:
                handle_domain_event_task.delay(event_type, payload)

    sync_run(replay())


# Small enough that one pass finishes well inside the timeout below, repeated until the
# table is clean. A month of api_request_metrics is millions of rows, and as ONE unbounded
# DELETE it could never complete within the shared engine's 5 s budget: the retention
# sweep failed on every run and the table only ever grew.
_PRUNE_BATCH_ROWS = 5_000
_PRUNE_COMMAND_TIMEOUT_SECONDS = 120


@celery_app.task(name="prune_operational_metrics")
def prune_operational_metrics() -> None:
    """Bound telemetry growth and remove physically expired answer caches."""

    async def prune_batched(db, table, predicate) -> int:
        """Delete matching rows in bounded passes; returns the total removed."""
        removed = 0
        while True:
            # A subquery over the primary key, because Postgres has no DELETE ... LIMIT.
            victims = select(table.c.id).where(predicate).limit(_PRUNE_BATCH_ROWS)
            deleted = (
                await db.execute(table.delete().where(table.c.id.in_(victims)))
            ).rowcount or 0
            await db.commit()
            removed += deleted
            if deleted < _PRUNE_BATCH_ROWS:
                return removed

    async def prune() -> None:
        # Its own engine, because the shared one pins command_timeout to 5 s (deps.py) so
        # that no request can hold a pooled connection longer than that. asyncpg enforces
        # that ceiling CLIENT-side, which is why a server-side `SET LOCAL
        # statement_timeout` cannot lift it — the driver cancels the statement before the
        # server's limit is ever consulted. NullPool plus an explicit dispose keeps this
        # from leaving idle connections behind between daily runs.
        prune_engine = create_async_engine(
            settings.DATABASE_URL,
            poolclass=NullPool,
            connect_args={
                "timeout": 10,
                "command_timeout": _PRUNE_COMMAND_TIMEOUT_SECONDS,
            },
        )
        try:
            factory = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
                bind=prune_engine,
                class_=AsyncSession,
            )
            async with factory() as db:
                await set_database_context(db, None, True)
                cutoff = datetime.utcnow() - timedelta(
                    days=settings.METRICS_RETENTION_DAYS
                )
                metrics_removed = await prune_batched(
                    db,
                    ApiRequestMetric.__table__,
                    ApiRequestMetric.created_at < cutoff,
                )
                # Cache answers can contain authorized document passages. Their
                # six-hour expiry must remove storage as well as disable reads.
                cache_removed = await prune_batched(
                    db,
                    AiCache.__table__,
                    AiCache.expires_at < datetime.utcnow(),
                )
            logger.info(
                "Operational retention sweep completed",
                api_request_metrics_deleted=metrics_removed,
                ai_cache_deleted=cache_removed,
            )
        finally:
            await prune_engine.dispose()

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


@celery_app.task(name="link_related_articles")
def link_related_articles_task() -> None:
    """Merge topical matches (domain/article_linking.py) into recently changed articles.

    Runs nightly rather than once at publish time: an article's best topical matches can
    change as OTHER articles are added later, so a link computed only when an article
    itself is created or edited goes stale as the corpus grows around it. Scoped to
    articles touched in the last day so a large corpus does not mean an
    every-article-against-every-other-article scan on every run.
    """
    async def link() -> None:
        from src.domain.article_linking import find_topical_matches
        from src.models.article import Article

        cutoff = datetime.utcnow() - timedelta(hours=24)
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            articles = (
                await db.execute(
                    select(Article).where(
                        Article.status == "published",
                        Article.lifecycle_status == "active",
                        Article.updated_at >= cutoff,
                    )
                )
            ).scalars().all()
            for article in articles:
                try:
                    matches = await find_topical_matches(db, article)
                except Exception as exc:
                    # One article's linking failure (a malformed embedding, a transient
                    # DB error) must not abort the whole nightly run for every other
                    # article that would otherwise have linked cleanly.
                    logger.warning(
                        "Topical linking failed for one article",
                        article_id=str(article.id),
                        error=str(exc),
                    )
                    continue
                if not matches:
                    continue
                existing = set(article.related_article_ids or [])
                merged = sorted(existing | set(matches))
                if merged != sorted(existing):
                    article.related_article_ids = merged
            await db.commit()
    sync_run(link())


async def _run_approval_agent_sweep() -> None:
    """Apply active approval rules to every company's pending queue.

    The agent (src/domain/approval_agent.py) and its API were reachable from day one, but
    nothing ever called them except a human hitting POST /governance/approval-agent/run --
    a correctly-scoped, authority-granted rule still left every new draft in the human
    queue until somebody remembered to run it again. This is that missing trigger.
    """
    from src.domain import approval_agent
    from src.models.governance import ApprovalRule
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        domains = (
            await db.execute(
                select(ApprovalRule.company_domain)
                .where(ApprovalRule.active.is_(True))
                .distinct()
            )
        ).scalars().all()
        for domain in domains:
            await approval_agent.run(db, str(domain), dry_run=False)


@celery_app.task(name="run_approval_agent")
def run_approval_agent() -> None:
    sync_run(_run_approval_agent_sweep())


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
                    payload = {
                        "event": "draft_overdue",
                        "draft_id": str(draft.id),
                        "to": recipient.email,
                        "subject": f"Approval overdue: {draft.title}",
                        "text": f"The draft '{draft.title}' has been awaiting approval beyond the {settings.REVIEW_SLA_DAYS}-day SLA.",
                    }
                    # The dedupe check and the insert are ONE statement, so the window
                    # between them cannot be interleaved: a SELECT followed by db.add()
                    # let two concurrent runs — or a Celery retry of this task — both see
                    # no recent notification and both queue an escalation, so the
                    # approver got the same overdue email twice.
                    #
                    # Expressed in-query rather than with a unique index because a
                    # partial-unique index on a JSON payload plus a 24-hour window is not
                    # something a UNIQUE constraint can state, and adding one would need
                    # a migration owned elsewhere. Postgres evaluates the NOT EXISTS
                    # against the same snapshot that performs the insert, which is what
                    # closes the race.
                    duplicate = select(NotificationQueue.id).where(
                        NotificationQueue.recipient_user_id == recipient.id,
                        NotificationQueue.type == "email",
                        NotificationQueue.created_at >= datetime.utcnow() - timedelta(hours=24),
                        NotificationQueue.payload["event"].as_string() == "draft_overdue",
                        NotificationQueue.payload["draft_id"].as_string() == str(draft.id),
                    )
                    await db.execute(
                        NotificationQueue.__table__.insert().from_select(
                            ["id", "recipient_user_id", "type", "payload", "status", "attempts", "created_at", "updated_at"],
                            select(
                                literal(uuid.uuid4(), type_=Uuid),
                                literal(recipient.id, type_=Uuid),
                                literal("email"),
                                literal(payload, type_=JSON),
                                literal("pending"),
                                literal(0),
                                literal(datetime.utcnow(), type_=DateTime),
                                literal(datetime.utcnow(), type_=DateTime),
                            ).where(~exists(duplicate)),
                        )
                    )
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
    from src.domain.department_routing import route_document_candidates_llm
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
            #
            # NOT for a draft that IS a committed split product. `commit_candidates`
            # creates each child with restructure_status="lossless_ready", the UI offers
            # "Retry AI format" on anything that is not "llm", and re-splitting a child
            # by department yields >1 candidate again -> batch_review_required -> commit
            # -> more children, forever, fanning out on every pass. Reformatting a
            # child's reading view is still useful, so only the re-split is skipped.
            is_split_product = (
                (draft.content_metadata or {}).get("submission_kind") == "split_candidate"
            )
            if not is_split_product:
                await db.execute(
                    delete(DraftCandidate).where(DraftCandidate.draft_id == draft.id)
                )
                for item in await route_document_candidates_llm(
                    draft.title, result.body_md, departments
                ):
                    db.add(
                        DraftCandidate(
                            draft_id=draft.id,
                            **item,
                        )
                    )
            # Same automatic treatment as department routing above: a reviewer who typed
            # tags at upload time keeps them exactly as typed (never silently replaced),
            # but a draft that arrives with none gets AI suggestions to review/edit
            # instead of an empty field. Best-effort -- see domain/auto_tagging.py --
            # so a failure here never touches the restructuring result already committed.
            if not draft.tags:
                from src.domain.auto_tagging import suggest_tags_for_document
                from src.models.article import TagCatalog

                # Same governance rule as the manual bulk endpoint (auto_tag_articles in
                # articles.py): only suggest tags already in the tenant's approved
                # vocabulary. An automatic, less-reviewed path is exactly where that
                # matters MORE, not less.
                catalogue_rows = (
                    await db.execute(
                        select(TagCatalog.tag, TagCatalog.normalized_tag).where(
                            TagCatalog.company_domain == draft.company_domain,
                            TagCatalog.active.is_(True),
                        )
                    )
                ).all()
                catalogue = {normalized for _tag, normalized in catalogue_rows}
                draft.tags = await suggest_tags_for_document(
                    draft.title,
                    result.body_md,
                    (draft.content_metadata or {}).get("type", ""),
                    catalogue=catalogue,
                    catalogue_examples=[tag for tag, _normalized in catalogue_rows],
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
        # Atomic increments, not read-modify-write. Each iteration used its own session to
        # load the row, add one in Python and commit, so two workers on the same job (a
        # broker redelivery, or a manual re-queue) could each read `completed = 7` and
        # each write 8 — losing a completion and leaving a progress bar that never reaches
        # its total. `completed = completed + 1` is resolved by the database.
        #
        # `updated_at` is set explicitly because a Core UPDATE bypasses the ORM's onupdate
        # hook, and the stale-job sweep uses that column as this job's only heartbeat.
        try:
            await index_article(article_id)
            progress_values: dict[str, Any] = {
                "completed": IndexReprocessJob.completed + 1,
                "updated_at": datetime.utcnow(),
            }
        except Exception as exc:
            progress_values = {
                "failed": IndexReprocessJob.failed + 1,
                "last_error": str(exc)[:2000],
                "updated_at": datetime.utcnow(),
            }
        async with SessionLocal() as progress_db:
            await set_database_context(progress_db, None, True)
            await progress_db.execute(
                update(IndexReprocessJob)
                .where(IndexReprocessJob.id == uuid.UUID(job_id_str))
                .values(**progress_values)
            )
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


# A reprocess run has no heartbeat other than the `updated_at` each article's progress
# write touches, so "no progress for this long" is the only available liveness signal.
# Generous, because one article can involve a slow embedding pass and a job with a single
# very large article must not be declared dead while it is still working.
INDEX_JOB_STALE_MINUTES = 60
# Re-queued at most this many times. Past that the job is a persistent failure rather than
# an interrupted one, and re-dispatching it forever would re-index the same set on every
# sweep.
INDEX_JOB_MAX_RETRIES = 2


@celery_app.task(name="recover_stale_index_reprocess_jobs")
def recover_stale_index_reprocess_jobs() -> int:
    """Resume or fail reprocess jobs abandoned by a dead worker.

    ``run_reprocess_index_job`` sets ``running`` and only clears it after the whole loop,
    so a worker killed mid-run left the row ``running`` forever: the operator saw a frozen
    progress bar, and nothing ever retried the remaining articles.
    """

    async def recover() -> int:
        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            cutoff = datetime.utcnow() - timedelta(minutes=INDEX_JOB_STALE_MINUTES)
            jobs = (await db.execute(
                select(IndexReprocessJob)
                .where(
                    IndexReprocessJob.status == "running",
                    IndexReprocessJob.updated_at < cutoff,
                )
                .limit(20)
                .with_for_update(skip_locked=True)
            )).scalars().all()
            requeued: list[uuid.UUID] = []
            for job in jobs:
                if job.retry_count >= INDEX_JOB_MAX_RETRIES:
                    job.status = "failed"
                    job.completed_at = datetime.utcnow()
                    job.last_error = (
                        "Abandoned by an interrupted worker and past the retry limit"
                    )
                    continue
                job.retry_count += 1
                job.status = "queued"
                job.last_error = "Recovered after worker/process interruption"
                requeued.append(job.id)
            await db.commit()
        # After the commit, for the same reason as replay_outbox_task: a task dispatched
        # before its claim is durable can start, and finish, against the pre-claim row.
        for job_id in requeued:
            reprocess_index_job_task.delay(str(job_id))
        if jobs:
            logger.info(
                "Stale index reprocess sweep completed",
                examined=len(jobs),
                requeued=len(requeued),
                failed=len(jobs) - len(requeued),
            )
        return len(requeued)

    return sync_run(recover())


@celery_app.task(name="recompute_permissions_task")
def recompute_permissions_task(article_id_str: str):
    article_id = uuid.UUID(article_id_str)
    logger.info(
        "Recomputing permission metadata snapshot on chunks", article_id=article_id
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

            await chunk_repo.update_permissions(
                article_id=article_id,
                sensitivity=article.sensitivity,
                visibility=article.visibility,
                dept=article.dept,
            )
            logger.info(
                "Chunk permission metadata updated successfully",
                article_id=article_id,
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
            # ChunkRepository stages the delete without committing, so that a rebuild can
            # be one transaction. A standalone deletion has to commit it.
            await db.commit()
            logger.info("Article chunks deleted successfully", article_id=article_id)

    sync_run(process())


# Deliberately NOT autoretrying. Retry policy for a connector sync lives in the durable
# queue: finish_sync_request puts the request back with exponential backoff and an
# attempt ceiling. Celery retrying on top of that meant a failed sync was requeued AND
# re-executed, so the retry and the queue dispatcher could walk the same drive at the
# same time, each undoing the other's cursor. One authority for retries, and it is the
# one that survives a worker restart.
@celery_app.task(name="sync_cloud_connector_task")
def sync_cloud_connector_task(connector_id_str: str, job_id_str: str, sync_request_id_str: str | None = None):
    """Run an idempotent provider delta sync from a durable cursor."""

    async def process():
        from src.domain.cloud_sync import sync_cloud_connector
        from src.domain.sync_queue import finish_sync_request, mark_sync_request_running

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            sync_request_id = uuid.UUID(sync_request_id_str) if sync_request_id_str else None
            if sync_request_id:
                await mark_sync_request_running(db, sync_request_id)
            sync_request = await db.get(SyncRequest, sync_request_id) if sync_request_id else None
            connector = await db.get(Connector, uuid.UUID(connector_id_str))
            job = await db.get(ConnectorJob, uuid.UUID(job_id_str))
            if not connector or not job:
                if sync_request_id:
                    await finish_sync_request(
                        db,
                        sync_request_id,
                        success=False,
                        error="Connector or job no longer exists",
                        retryable=False,
                    )
                return
            try:
                await sync_cloud_connector(
                    db,
                    connector,
                    job,
                    scope_id=sync_request.scope_id if sync_request else None,
                )
            except Exception as exc:
                if sync_request_id:
                    await finish_sync_request(
                        db,
                        sync_request_id,
                        success=False,
                        error=str(exc),
                        retryable=bool(getattr(exc, "retryable", True)),
                    )
                raise
            if sync_request_id:
                await finish_sync_request(db, sync_request_id, success=True)

    sync_run(process())


@celery_app.task(name="schedule_cloud_connector_syncs")
def schedule_cloud_connector_syncs():
    """Polling fallback: wake every active cloud connector periodically."""

    async def process():
        from datetime import timedelta
        from src.domain.sync_queue import claim_sync_request, enqueue_connector_sync
        from src.models.connectors import SourceScope

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            connectors = (
                (
                    await db.execute(
                        select(Connector).where(
                            Connector.system.in_(sorted(REMOTE_PROVIDERS)),
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
                request = await enqueue_connector_sync(
                    db,
                    connector.id,
                    reason="polling",
                    requested_by=connector.created_by,
                    priority=50,
                )
                await db.commit()
                # Claim before dispatching. Handing an unclaimed request to Celery left
                # it in "queued", so dispatch_pending_sync_requests picked the same row
                # up and ran a second, concurrent sync of the same connector. None means
                # the dispatcher got there first, which is equally fine.
                claimed = await claim_sync_request(db, request.id)
                if claimed and claimed.job_id:
                    sync_cloud_connector_task.delay(
                        str(connector.id), str(claimed.job_id), str(claimed.id)
                    )

    sync_run(process())


@celery_app.task(name="reconcile_cloud_connectors")
def reconcile_cloud_connectors():
    """Force a complete provider walk on a bounded cadence.

    Webhooks and delta cursors are the fast path. This pass is the safety net
    for missed notifications, expired delta state, moves, and provider outages.
    """

    async def process():
        from src.domain.sync_queue import enqueue_connector_sync
        from src.models.connectors import SourceScope, SyncCursor

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            connectors = (
                await db.execute(
                    select(Connector).where(
                        Connector.system.in_(sorted(REMOTE_PROVIDERS)),
                        Connector.status.in_(["active", "error"]),
                    )
                )
            ).scalars().all()
            for connector in connectors:
                selected_scopes = (
                    await db.execute(
                        select(SourceScope).where(
                            SourceScope.connector_id == connector.id,
                            SourceScope.selected.is_(True),
                        )
                    )
                ).scalars().all()
                for scope in selected_scopes:
                    cursor = (
                        await db.execute(
                            select(SyncCursor).where(
                                SyncCursor.connector_id == connector.id,
                                SyncCursor.scope_id == scope.id,
                            )
                        )
                    ).scalar_one_or_none()
                    if cursor is None:
                        cursor = SyncCursor(
                            connector_id=connector.id,
                            scope_id=scope.id,
                            cursor_type=provider_cursor_type(connector.system),
                        )
                        db.add(cursor)
                    cursor.full_sync_required = True
                    cursor.status = "reconcile"
                    request = await enqueue_connector_sync(
                        db,
                        connector.id,
                        scope_id=scope.id,
                        reason="reconcile",
                        requested_by=connector.created_by,
                        priority=40,
                    )
                    # Deliberately not dispatched here: dispatch_pending_sync_requests
                    # claims it under a row lock. The commit below only has to make the
                    # cursor flag and the coalesced request durable.
            await db.commit()

    sync_run(process())


@celery_app.task(name="dispatch_pending_sync_requests")
def dispatch_pending_sync_requests():
    """Recover webhook requests that were persisted before a process restart."""

    async def process():
        from src.domain.sync_queue import claim_sync_request, recover_stale_sync_requests

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            await recover_stale_sync_requests(db)
            request_ids = (
                await db.execute(
                    select(SyncRequest.id)
                    .where(
                        SyncRequest.status == "queued",
                        SyncRequest.available_at <= datetime.utcnow(),
                    )
                    .order_by(SyncRequest.priority.asc(), SyncRequest.created_at.asc())
                    .limit(25)
                )
            ).scalars().all()
        for request_id in request_ids:
            async with SessionLocal() as claim_db:
                await set_database_context(claim_db, None, True)
                claimed = await claim_sync_request(claim_db, request_id)
                if not claimed or not claimed.job_id:
                    continue
                sync_cloud_connector_task.delay(
                    str(claimed.connector_id),
                    str(claimed.job_id),
                    str(claimed.id),
                )

    sync_run(process())


@celery_app.task(name="renew_webhook_subscriptions")
def renew_webhook_subscriptions():
    """Keep provider push subscriptions alive, and put back the ones that died.

    Two passes, because renewal alone is not enough: a Graph subscription can be
    extended in place, a Google Drive channel can only be replaced, and a subscription
    deleted at the tenant comes back as an error. Both live in the domain layer so the
    inline (JOB_MODE != celery) dispatch loop runs exactly the same repair.

    Runs every 10 minutes against a 20-minute horizon, so a subscription gets several
    attempts before it expires and one failed run is not enough to drop it.
    """

    async def process():
        from src.domain.webhook_subscriptions import (
            renew_due_subscriptions,
            repair_webhook_subscriptions,
        )

        async def restore_context(session) -> None:
            await set_database_context(session, None, True)

        async with SessionLocal() as db:
            await set_database_context(db, None, True)
            await renew_due_subscriptions(db)
            await repair_webhook_subscriptions(db, set_context=restore_context)

    sync_run(process())


