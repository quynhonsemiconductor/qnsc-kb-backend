"""Delete every piece of knowledge-base content for one tenant, in one transaction.

WHY THIS EXISTS. Resetting a test corpus by hand means deleting articles one at a time
through the UI, and that does not actually reset anything: the sync state survives, so the
next connector run reports every provider file "unchanged" and re-imports nothing. The KB
ends up permanently empty instead of freshly empty.

WHAT IT IS NOT. This is content-only. Users, roles, RBAC, departments, access groups, tag
vocabulary, feature flags and audit logs are all KEPT — a purge is not a factory reset, and
audit_logs in particular is the record that the purge happened.

THE ORDER MATTERS AND IS NOT ARBITRARY. Two traps, both silent:

1. `document_sources.article_id` is the one FK into articles that is ondelete=SET NULL
   (models/article.py). A bulk `DELETE FROM articles` therefore NULLs every source row
   rather than removing it, and because that table's RLS policy is a join through articles,
   the orphans become invisible to tenant-scoped queries forever — along with the R2 objects
   they point at. So document_sources is deleted BEFORE articles.

2. `external_documents` caches `revision` + `content_hash` per provider file, and the sync
   path skips ingest when both match. Deleting articles while keeping that cache means the
   next full reconcile walk decides nothing changed and re-imports nothing. So the cache is
   deleted and the cursors are reset together, or the purge quietly bricks the corpus.

Everything else rides on ondelete=CASCADE: one DELETE on `articles` removes parent_chunks,
article_chunks, chunk_metadata, article_versions, article_user_permissions, article_tags,
article_departments, comments, votes, bookmarks, article_followers and
article_edit_requests. One DELETE on `connectors` removes the entire sync tree.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ai import AiCache, AiConversation, AiUsageLog
from src.models.article import Article, DocumentSource
from src.models.connectors import (
    ConnectorNotification,
    DocumentVersion,
    ExternalDocument,
    SyncCursor,
    SyncRequest,
)
from src.models.governance import (
    ConflictRecord,
    Gap,
    IngestionFingerprint,
    PendingDraft,
)
from src.models.ops import Connector, IndexReprocessJob, NotificationQueue, OutboxEvent
from src.models.user import User

logger = structlog.get_logger()


@dataclass
class PurgeCounts:
    """What the purge removed, per table. Reported whether or not it was a dry run."""

    articles: int = 0
    document_sources: int = 0
    pending_drafts: int = 0
    external_documents: int = 0
    connectors: int = 0
    sync_cursors_reset: int = 0
    sync_requests: int = 0
    connector_notifications: int = 0
    ingestion_fingerprints: int = 0
    gaps: int = 0
    conflict_records: int = 0
    index_reprocess_jobs: int = 0
    ai_cache: int = 0
    ai_conversations: int = 0
    ai_usage_logs: int = 0
    notifications: int = 0
    outbox_events: int = 0
    storage_objects: int = 0
    storage_failures: list[str] = field(default_factory=list)
    #: The keys the purge orphaned, for the caller to delete AFTER its commit. Not part of
    #: the reported payload -- it is a work list, not a count.
    storage_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"storage_failures", "storage_keys"}
        }
        payload["storage_failures"] = len(self.storage_failures)
        return payload

    @property
    def total_rows(self) -> int:
        """Everything that was or would be deleted, excluding resets and object storage."""
        return sum(
            value
            for key, value in self.__dict__.items()
            if isinstance(value, int)
            and key not in {"storage_objects", "sync_cursors_reset"}
        )


async def _tenant_user_ids(db: AsyncSession, company_domain: str) -> list[Any]:
    """The tenant's users, for the AI tables that carry no company_domain of their own.

    ai_cache, ai_conversations and ai_usage_logs are owner-scoped, not tenant-scoped, so
    they can only be reached through this join. Their RLS policies are owner-based too,
    which is why the purge has to run as a global admin to see other users' rows at all.
    """
    rows = await db.execute(select(User.id).where(User.company_domain == company_domain))
    return list(rows.scalars().all())


async def _storage_keys(db: AsyncSession, company_domain: str) -> list[str]:
    """Every R2 object key the purge is about to orphan, collected BEFORE any delete.

    Once the rows are gone the keys are unrecoverable from the database, so they are read
    up front. Missing one is survivable — `cleanup_orphaned_source_objects` sweeps
    unreferenced keys daily — but leaving the objects behind means the content is still
    downloadable, which is not what "purge" should mean.
    """
    keys: set[str] = set()

    source_keys = await db.execute(
        select(DocumentSource.storage_key)
        .join(Article, Article.id == DocumentSource.article_id)
        .where(
            Article.company_domain == company_domain,
            DocumentSource.storage_key.is_not(None),
        )
    )
    keys.update(key for key in source_keys.scalars().all() if key)

    draft_keys = await db.execute(
        select(PendingDraft.storage_key).where(
            PendingDraft.company_domain == company_domain,
            PendingDraft.storage_key.is_not(None),
        )
    )
    keys.update(key for key in draft_keys.scalars().all() if key)

    version_keys = await db.execute(
        select(DocumentVersion.storage_key)
        .join(
            ExternalDocument,
            ExternalDocument.id == DocumentVersion.external_document_id,
        )
        .join(Connector, Connector.id == ExternalDocument.connector_id)
        .where(
            Connector.company_domain == company_domain,
            DocumentVersion.storage_key.is_not(None),
        )
    )
    keys.update(key for key in version_keys.scalars().all() if key)

    return sorted(keys)


async def _count(db: AsyncSession, statement) -> int:
    """How many rows a SELECT would return, for dry-run reporting.

    COUNT in the database, not len() over the rows. Every caller below passes a
    `select(Model.id)` covering a whole tenant, so materialising them meant pulling one
    UUID per article, chunk, source and log row into Python — tens of megabytes on a real
    corpus, to produce eleven integers a dry run then throws away.
    """
    return await db.scalar(select(func.count()).select_from(statement.subquery())) or 0


async def delete_purged_objects(counts: PurgeCounts) -> None:
    """Remove the R2 objects a completed purge orphaned. Call AFTER the commit.

    Touches no session and raises nothing. By the time this runs the deletions are durable,
    so the purge has already succeeded from the caller's point of view — letting an
    object-store failure escape would turn a completed purge into a 500 telling the operator
    nothing happened, and invite them to run it again. Failures are logged and the keys are
    left to `cleanup_orphaned_source_objects`, whose daily sweep now finds them genuinely
    unreferenced. Same reasoning as the draft rejection path in domain/governance.py.

    `storage_objects` is re-derived here rather than left at len(storage_keys): before the
    commit it is a forecast, and the number in the response has to be what was actually
    destroyed.
    """
    counts.storage_objects = 0
    if not counts.storage_keys:
        return
    try:
        from src.domain.source_storage import delete_source
    except Exception as exc:  # pragma: no cover - depends on optional storage deps
        counts.storage_failures.extend(counts.storage_keys)
        logger.warning("Purge could not load object storage client", error=str(exc))
        return

    for key in counts.storage_keys:
        try:
            await asyncio.to_thread(delete_source, key)
            counts.storage_objects += 1
        except Exception as exc:  # pragma: no cover - depends on object store
            counts.storage_failures.append(key)
            logger.warning("Purge could not delete source object", key=key, error=str(exc))


async def purge_knowledge_base(
    db: AsyncSession, company_domain: str, *, dry_run: bool = True
) -> PurgeCounts:
    """Delete every article, document, chunk and connector for one tenant.

    One transaction, so the background workers that run every 30 seconds observe either the
    full corpus or none of it, never a half-deleted one. The caller commits, and then calls
    `delete_purged_objects(counts)` -- object storage has no rollback, so destroying the R2
    objects before that commit is what would make a failed commit unrecoverable.

    A dry run counts exactly what a real run would delete and writes nothing, which is the
    same opt-in shape the approval agent uses: the destructive path has to be asked for.
    """
    counts = PurgeCounts()
    user_ids = await _tenant_user_ids(db, company_domain)
    article_ids_stmt = select(Article.id).where(Article.company_domain == company_domain)
    connector_ids_stmt = select(Connector.id).where(
        Connector.company_domain == company_domain
    )

    # Materialised, not left as a subquery: STEP 7 has to match outbox payloads by article
    # id AFTER the articles are gone, at which point the subquery would return nothing.
    article_id_rows = await db.execute(article_ids_stmt)
    article_ids = {str(value) for value in article_id_rows.scalars().all()}

    # Read the object keys before anything is deleted; afterwards they are unrecoverable.
    keys = await _storage_keys(db, company_domain)

    if dry_run:
        counts.articles = await _count(db, article_ids_stmt)
        counts.connectors = await _count(db, connector_ids_stmt)
        counts.document_sources = await _count(
            db,
            select(DocumentSource.id).where(
                DocumentSource.article_id.in_(article_ids_stmt)
            ),
        )
        counts.pending_drafts = await _count(
            db,
            select(PendingDraft.id).where(PendingDraft.company_domain == company_domain),
        )
        counts.external_documents = await _count(
            db,
            select(ExternalDocument.id).where(
                ExternalDocument.connector_id.in_(connector_ids_stmt)
            ),
        )
        counts.ingestion_fingerprints = await _count(
            db,
            select(IngestionFingerprint.id).where(
                IngestionFingerprint.company_domain == company_domain
            ),
        )
        counts.gaps = await _count(
            db, select(Gap.id).where(Gap.company_domain == company_domain)
        )
        counts.conflict_records = await _count(
            db,
            select(ConflictRecord.id).where(
                ConflictRecord.company_domain == company_domain
            ),
        )
        counts.index_reprocess_jobs = await _count(
            db,
            select(IndexReprocessJob.id).where(
                IndexReprocessJob.company_domain == company_domain
            ),
        )
        counts.sync_cursors_reset = await _count(
            db,
            select(SyncCursor.id).where(SyncCursor.connector_id.in_(connector_ids_stmt)),
        )
        if user_ids:
            counts.ai_cache = await _count(
                db, select(AiCache.id).where(AiCache.owner_user_id.in_(user_ids))
            )
            counts.ai_conversations = await _count(
                db,
                select(AiConversation.id).where(AiConversation.user_id.in_(user_ids)),
            )
            counts.ai_usage_logs = await _count(
                db, select(AiUsageLog.id).where(AiUsageLog.user_id.in_(user_ids))
            )
        counts.storage_objects = len(keys)
        logger.info(
            "Knowledge base purge dry run",
            company_domain=company_domain,
            **counts.as_dict(),
        )
        return counts

    # ---- STEP 1. AI surfaces. Owner-scoped, so they need the users join, and they hold
    # answers quoting passages that are about to disappear. Cascades reach ai_feedback
    # and ai_messages.
    if user_ids:
        result = await db.execute(
            delete(AiCache).where(AiCache.owner_user_id.in_(user_ids))
        )
        counts.ai_cache = result.rowcount or 0
        result = await db.execute(
            delete(AiConversation).where(AiConversation.user_id.in_(user_ids))
        )
        counts.ai_conversations = result.rowcount or 0
        result = await db.execute(
            delete(AiUsageLog).where(AiUsageLog.user_id.in_(user_ids))
        )
        counts.ai_usage_logs = result.rowcount or 0
        result = await db.execute(
            delete(NotificationQueue).where(
                NotificationQueue.recipient_user_id.in_(user_ids)
            )
        )
        counts.notifications = result.rowcount or 0

    # ---- STEP 2. Flat tenant-scoped tables. No incoming FKs, so order among them is free.
    # ingestion_fingerprints is NOT optional: an approved fingerprint outliving its article
    # refuses the same file forever with 409 duplicate_document, and deleting every article
    # does not clear it. That is documented at the upload call site in routers/articles.py.
    for model, attribute in (
        (IngestionFingerprint, "ingestion_fingerprints"),
        (Gap, "gaps"),
        (ConflictRecord, "conflict_records"),
        (IndexReprocessJob, "index_reprocess_jobs"),
    ):
        result = await db.execute(
            delete(model).where(model.company_domain == company_domain)
        )
        setattr(counts, attribute, result.rowcount or 0)

    # ---- STEP 3. Drafts, cascading their transitions and candidates.
    result = await db.execute(
        delete(PendingDraft).where(PendingDraft.company_domain == company_domain)
    )
    counts.pending_drafts = result.rowcount or 0

    # ---- STEP 4. document_sources BEFORE articles. This is the trap: the FK is SET NULL,
    # so deleting articles first strands these rows with a NULL article_id, and their RLS
    # policy is a join through articles — the orphans become permanently invisible to the
    # tenant, keeping their R2 objects alive with them.
    result = await db.execute(
        delete(DocumentSource).where(DocumentSource.article_id.in_(article_ids_stmt))
    )
    counts.document_sources = result.rowcount or 0

    # ---- STEP 5. Articles. One statement, thirteen dependent tables, all via CASCADE:
    # parent_chunks, article_chunks, chunk_metadata, article_versions,
    # article_user_permissions, article_tags, article_departments,
    # comments, votes, bookmarks, article_followers, article_edit_requests.
    result = await db.execute(
        delete(Article).where(Article.company_domain == company_domain)
    )
    counts.articles = result.rowcount or 0

    # ---- STEP 6. The provider-side skip cache, and the cursors that decide what a sync
    # considers new. Deleting the documents without resetting the cursors is the second
    # trap: the next reconcile walk would report every file unchanged and re-import
    # nothing, leaving the KB permanently empty rather than freshly empty.
    #
    # Connector rows themselves are KEPT — they hold the OAuth grant, and deleting them
    # would force the operator to reconnect SharePoint by hand after every test reset.
    result = await db.execute(
        delete(ExternalDocument).where(
            ExternalDocument.connector_id.in_(connector_ids_stmt)
        )
    )
    counts.external_documents = result.rowcount or 0

    result = await db.execute(
        update(SyncCursor)
        .where(SyncCursor.connector_id.in_(connector_ids_stmt))
        .values(
            cursor_value=None,
            full_sync_required=True,
            status="reconcile",
            last_success_at=None,
            last_error=None,
        )
    )
    counts.sync_cursors_reset = result.rowcount or 0

    # Queued work for those connectors would otherwise execute against the state we just
    # removed, re-importing documents moments after the purge returned.
    result = await db.execute(
        delete(SyncRequest).where(SyncRequest.connector_id.in_(connector_ids_stmt))
    )
    counts.sync_requests = result.rowcount or 0
    result = await db.execute(
        delete(ConnectorNotification).where(
            ConnectorNotification.connector_id.in_(connector_ids_stmt)
        )
    )
    counts.connector_notifications = result.rowcount or 0

    # ---- STEP 7. Undispatched outbox events naming content that is now gone. The replay
    # loop runs every 30 seconds and tolerates missing rows, but the work is pointless and
    # its error noise hides real failures.
    #
    # Matched on payload["article_id"], NOT on the event name. That is the field the replay
    # path itself keys off (domain/events.py dispatches ArticlePublished / ArticleUpdated /
    # ArticleDeleted / PermissionChanged by article id and completes the row immediately
    # when it is absent), so matching the id catches every content event including
    # PermissionChanged, which no name prefix would have caught.
    if article_ids:
        # Filtered in SQL. This used to SELECT every pending outbox row in the database --
        # all tenants, no predicate -- and decide in Python, so one tenant's purge dragged
        # the whole backlog through the app and read other tenants' payloads to do it.
        # `payload` is JSON, so the id is compared as text, exactly as the notification
        # sweep in workers/tasks.py does it.
        result = await db.execute(
            delete(OutboxEvent).where(
                OutboxEvent.status == "pending",
                OutboxEvent.payload["article_id"].as_string().in_(article_ids),
            )
        )
        counts.outbox_events = result.rowcount or 0

    # ---- STEP 8. The objects are NOT deleted here. The caller still has an open
    # transaction: if its commit fails, every row above comes back while the R2 objects
    # would already be destroyed, leaving live document_sources rows pointing at storage
    # that no longer exists -- downloads 404 forever and nothing sweeps it, because the
    # rows still reference the keys. So the keys are handed back and
    # `delete_purged_objects` runs after the commit.
    counts.storage_keys = keys
    counts.storage_objects = len(keys)

    logger.warning(
        "Knowledge base purged",
        company_domain=company_domain,
        **counts.as_dict(),
    )
    return counts
