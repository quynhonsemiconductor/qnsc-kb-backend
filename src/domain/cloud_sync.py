"""Incremental cloud connector synchronization and version handoff."""

from __future__ import annotations

import hashlib
import json
import asyncio
import uuid
from datetime import datetime, timedelta

import structlog

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.domain.connector_adapters import (
    ConnectorProviderError,
    NormalizedChange,
    adapter_for,
)
from src.domain.connector_auth import ensure_connector_authorized
from src.domain.events import event_bus
from src.domain.source_extraction import (
    SUPPORTED_EXTENSIONS,
    extract_source_markdown,
    extract_source_pages,
)
from src.domain.source_storage import delete_source, save_source
from src.models.article import Article, ArticleUserPermission, DocumentSource
from src.models.user import AccessGroup, user_groups
from src.models.user import ExternalIdentity
from src.models.connectors import (
    DocumentVersion,
    ExternalAclPrincipal,
    ExternalDocument,
    ExternalGroupMapping,
    PermissionSnapshot,
    SourceScope,
    SyncCursor,
    SyncError,
)
from src.models.governance import (
    PendingDraft,
    DraftTransition,
    DraftCandidate,
    AuditLog,
)
from src.models.user import Department, User
from src.models.ops import Connector, ConnectorJob, NotificationQueue
from src.repositories.governance import GovernanceRepository
from src.repositories.article import ArticleRepository
from src.domain.governance import GovernanceService
from src.domain.department_routing import route_document_candidates

logger = structlog.get_logger()


async def _routed_candidate_items(
    db: AsyncSession, connector: Connector, title: str, text: str
) -> list[dict]:
    """Add deterministic department suggestions to every connector draft."""

    departments = (
        await db.execute(
            select(Department).where(
                Department.company_domain == connector.company_domain,
                Department.active.is_(True),
            )
        )
    ).scalars().all()
    return route_document_candidates(title, text, departments)


async def _persist_connector_draft(
    db: AsyncSession, connector: Connector, draft: PendingDraft, text: str
) -> None:
    """Persist connector input as Draft, then submit through the same workflow."""
    publication_mode = settings.CONNECTOR_AUTO_PUBLISH_MODE.strip().lower()
    if publication_mode != "governed":
        raise ConnectorProviderError(
            "CONNECTOR_AUTO_PUBLISH_MODE currently supports only governed publication",
            retryable=False,
            code="unsupported_publication_mode",
        )
    draft.content_metadata = {
        **(draft.content_metadata or {}),
        "connector_publication_mode": publication_mode,
        "connector_id": str(connector.id),
        "connector_name": connector.name,
    }
    db.add(draft)
    await db.flush()
    db.add(
        DraftTransition(
            draft_id=draft.id,
            from_status=None,
            to_status="draft",
            actor_id=draft.created_by,
            reason="Connector source imported",
            outcome="applied",
        )
    )
    for item in await _routed_candidate_items(db, connector, draft.title, text):
        db.add(DraftCandidate(draft_id=draft.id, **item))
    actor = await db.get(User, draft.created_by) if draft.created_by else None
    if actor:
        await GovernanceService(
            GovernanceRepository(db), ArticleRepository(db)
        ).submit_draft(
            actor, draft.id, "Connector source submitted for independent approval"
        )
    else:
        draft.status = "pending"
        db.add(
            DraftTransition(
                draft_id=draft.id,
                from_status="draft",
                to_status="pending",
                actor_id=None,
                reason="Connector source submitted",
                outcome="applied",
            )
        )
        await db.flush()


def _acl_hash(permissions: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(
                permissions,
                key=lambda item: (
                    item.get("principal_type", ""),
                    item.get("principal_id", ""),
                ),
            ),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _record_permission_change_audits(
    db: AsyncSession, article_ids: list[uuid.UUID], actor_id: uuid.UUID | None
) -> None:
    """Record the Article targets affected by a provider ACL reconciliation."""
    for article_id in dict.fromkeys(article_ids):
        db.add(
            AuditLog(
                user_id=actor_id,
                action="permission_change",
                target_type="article",
                target_id=str(article_id),
                outcome="success",
            )
        )


async def _replace_split_candidates(
    db: AsyncSession, draft: PendingDraft, text: str
) -> None:
    """Keep an existing pending connector draft aligned with its new source text."""
    await db.execute(delete(DraftCandidate).where(DraftCandidate.draft_id == draft.id))
    # The caller's connector is not stored on PendingDraft, so route using the
    # tenant key from the draft and keep the helper provider-independent.
    departments = (
        await db.execute(
            select(Department).where(
                Department.company_domain == draft.company_domain,
                Department.active.is_(True),
            )
        )
    ).scalars().all()
    for item in route_document_candidates(draft.title, text, departments):
        db.add(DraftCandidate(draft_id=draft.id, **item))


async def _cleanup_unreferenced_source_keys(
    db: AsyncSession, storage_keys: list[str]
) -> None:
    """Remove transient R2 objects after their database reference is replaced.

    DocumentSource and DocumentVersion rows intentionally retain historical
    source objects. Only keys with no remaining draft, source, or version
    reference are eligible for deletion, so deterministic content-addressed
    keys shared by multiple records are never removed accidentally.
    """
    for storage_key in dict.fromkeys(key for key in storage_keys if key):
        referenced = False
        for model in (PendingDraft, DocumentSource, DocumentVersion):
            if await db.scalar(
                select(model.id).where(model.storage_key == storage_key).limit(1)
            ):
                referenced = True
                break
        if referenced:
            continue
        try:
            await asyncio.to_thread(delete_source, storage_key)
        except Exception:
            # Object cleanup is deliberately best effort. The database no
            # longer treats this key as live, and an operator/GC pass can
            # retry a provider outage without failing the sync transaction.
            continue


async def _handle_deleted_document(
    db: AsyncSession, document: ExternalDocument, actor_id: uuid.UUID | None
) -> tuple[list[str], uuid.UUID | None]:
    """Apply a provider deletion to drafts, Articles, and audit history."""
    document.state = "deleted"
    cleanup_keys: list[str] = []
    drafts = (
        (
            await db.execute(
                select(PendingDraft).where(
                    PendingDraft.external_document_id == document.id,
                    PendingDraft.status.in_(("draft", "pending")),
                )
            )
        )
        .scalars()
        .all()
    )
    deletion_reason = "Source document was deleted by the provider during sync"
    for draft in drafts:
        if draft.storage_key:
            cleanup_keys.append(draft.storage_key)
            # The rejected draft retains its review history but no longer
            # points at a physical object that should remain in R2.
            draft.storage_key = None
        db.add(
            DraftTransition(
                draft_id=draft.id,
                from_status=draft.status,
                to_status="rejected",
                actor_id=actor_id,
                reason=deletion_reason,
                outcome="applied",
            )
        )
        draft.status = "rejected"
        draft.reviewed_by = actor_id
        draft.reviewed_at = datetime.utcnow()
        draft.review_note = deletion_reason

    article_id: uuid.UUID | None = None
    if document.article_id:
        article = await db.get(Article, document.article_id)
        if article:
            article.lifecycle_status = "inactive"
            article_id = article.id
            db.add(
                AuditLog(
                    user_id=actor_id,
                    action="delete",
                    target_type="article",
                    target_id=str(article.id),
                    outcome="success",
                )
            )
    db.add(
        AuditLog(
            user_id=actor_id,
            action="delete",
            target_type="external_document",
            target_id=str(document.id),
            outcome="success",
        )
    )
    await db.flush()
    return cleanup_keys, article_id


def _needs_content_ingest(
    *,
    is_file: bool,
    previous_exists: bool,
    previous_revision: str | None,
    current_revision: str | None,
    has_content_hash: bool,
    pending_draft_needs_candidates: bool,
) -> bool:
    """Ensure new/unmaterialized files are ingested even if a provider flag is weak."""
    return bool(
        is_file
        and (
            not previous_exists
            or previous_revision != current_revision
            or not has_content_hash
            or pending_draft_needs_candidates
        )
    )


# Providers hand back every object in the drive: videos, archives, OneNote sections,
# Drive shortcuts. Deciding from the name and mime type BEFORE the download saves the
# bandwidth and, more importantly, keeps a file that could never have been indexed from
# being reported to an admin as a synchronization failure.
_INGESTIBLE_GOOGLE_EDITORS = frozenset({"document", "presentation", "spreadsheet"})


def _unsupported_reason(name: str, mime_type: str | None) -> str | None:
    """Return why this item cannot be indexed, or None when it can."""

    if mime_type and mime_type.startswith("application/vnd.google-apps."):
        editor = mime_type.rsplit(".", 1)[-1]
        if editor in _INGESTIBLE_GOOGLE_EDITORS:
            return None
        return f"Google Drive {editor} files cannot be indexed"
    suffix = f".{name.rsplit('.', 1)[-1].lower()}" if "." in name else ""
    if suffix in SUPPORTED_EXTENSIONS:
        return None
    return f"Unsupported file type ({suffix or 'no extension'})"


# Codes that describe the CONNECTOR, not the file in front of it. Quarantining one item
# for any of these would quarantine every item in the drive, one at a time, and report a
# corrupt corpus to the admin instead of an expired credential or a throttled tenant.
_SCOPE_FATAL_CODES = frozenset(
    {
        "401",
        "403",
        "429",
        "500",
        "502",
        "503",
        "504",
        "not_authorized",
        "invalid_auth_mode",
        "no_access_token",
        "unsupported_provider",
        "unsupported_publication_mode",
        "resync_required",
        "untrusted_url",
    }
)

# How many times one unchanged revision may fail before the sync stops paying for it.
_MAX_ITEM_ATTEMPTS = 3
# A whole drive failing one file at a time is a connector problem wearing a file
# problem's clothes. Give up on the scope rather than walking forty thousand items to
# record forty thousand identical errors.
_MAX_CONSECUTIVE_ITEM_FAILURES = 25


def _is_item_level_failure(exc: BaseException) -> bool:
    """Whether this failure belongs to one document rather than the whole scope."""

    if isinstance(exc, ConnectorProviderError):
        return str(exc.code or "") not in _SCOPE_FATAL_CODES
    return True


def _ingest_failure(document: ExternalDocument | None) -> dict:
    value = (document.metadata_json or {}).get("ingest_failure") if document else None
    return value if isinstance(value, dict) else {}


def _is_quarantined(document: ExternalDocument, change: NormalizedChange) -> bool:
    """Whether this exact revision has already failed often enough to stop retrying.

    A new revision always gets a fresh attempt: replacing the broken file at the
    provider is precisely how a person fixes this, and it must be enough — nobody
    should have to find an admin screen to release a document they already repaired.
    """

    failure = _ingest_failure(document)
    if not failure or failure.get("revision") != (change.revision or ""):
        return False
    return int(failure.get("attempts", 0) or 0) >= _MAX_ITEM_ATTEMPTS


def _clear_ingest_failure(document: ExternalDocument) -> None:
    metadata = document.metadata_json or {}
    if "ingest_failure" in metadata:
        document.metadata_json = {
            key: value for key, value in metadata.items() if key != "ingest_failure"
        }


async def _record_item_failure(
    db: AsyncSession,
    *,
    connector_id: uuid.UUID,
    job_id: uuid.UUID,
    change: NormalizedChange,
    reason: str,
    code: str | None,
    retryable: bool,
) -> int:
    """Quarantine one document on a clean transaction and return its attempt count.

    Called only after a rollback, so nothing here may touch an ORM object loaded before
    the failure: every one of them is expired, and refreshing an expired attribute from
    inside this loop raises rather than reloads.
    """

    document = (
        await db.execute(
            select(ExternalDocument).where(
                ExternalDocument.connector_id == connector_id,
                ExternalDocument.corpus_id == change.corpus_id,
                ExternalDocument.external_id == change.external_id,
            )
        )
    ).scalar_one_or_none()
    previous = _ingest_failure(document)
    attempts = (
        int(previous.get("attempts", 0) or 0) + 1
        if previous.get("revision") == (change.revision or "")
        else 1
    )
    if document is not None:
        document.metadata_json = {
            **(document.metadata_json or {}),
            "ingest_failure": {
                "revision": change.revision or "",
                "reason": reason[:500],
                "code": code,
                "attempts": attempts,
                "at": datetime.utcnow().isoformat(),
            },
        }
    db.add(
        SyncError(
            connector_id=connector_id,
            job_id=job_id,
            external_document_id=document.id if document is not None else None,
            stage="ingest",
            error_code=(code or "ingest_failed")[:80],
            message=reason[:4000],
            retryable=retryable,
            attempts=attempts,
        )
    )
    await db.commit()
    return attempts


def _notify_connector_state_change(
    db: AsyncSession, connector: Connector, *, event: str, detail: str | None
) -> None:
    """Tell the connector's owner that its state changed, on the transition only.

    A connector that breaks is otherwise entirely silent: the polling loop keeps
    retrying, the job list fills with failures, and nobody looks at a job list until
    somebody notices the knowledge base has stopped answering questions about a
    document that has been in SharePoint for a fortnight. Queued on the TRANSITION, so
    a connector failing every ten minutes produces one notification, not one hundred
    and forty-four a day.
    """

    if not connector.created_by:
        return
    db.add(
        NotificationQueue(
            recipient_user_id=connector.created_by,
            type="in_app",
            payload={
                "event": event,
                "connector_id": str(connector.id),
                "connector_name": connector.name,
                "provider": connector.system,
                "detail": (detail or "")[:500] or None,
            },
        )
    )


def _note_item_outcome(
    summary: dict[str, object],
    *,
    name: str,
    action: str,
    scope_name: str,
    web_url: str | None,
    reason: str | None = None,
) -> None:
    """Record one file's outcome for the admin reading the job afterwards."""

    items = summary["items"]
    if isinstance(items, list) and len(items) < 200:
        entry: dict[str, object] = {
            "name": name,
            "action": action,
            "scope": scope_name,
            "web_url": web_url,
        }
        if reason:
            entry["reason"] = reason
        items.append(entry)
    if reason:
        errors = summary["errors"]
        # Kept separate from `items`, which is capped and mixes successes in. A person
        # opening a sync that "worked" still needs to see what it could not take.
        if isinstance(errors, list) and len(errors) < 50:
            errors.append(
                {"name": name, "action": action, "scope": scope_name, "reason": reason, "web_url": web_url}
            )


async def _upsert_document(
    db: AsyncSession, connector: Connector, scope: SourceScope, change: NormalizedChange
) -> ExternalDocument:
    document = (
        await db.execute(
            select(ExternalDocument).where(
                ExternalDocument.connector_id == connector.id,
                ExternalDocument.corpus_id == change.corpus_id,
                ExternalDocument.external_id == change.external_id,
            )
        )
    ).scalar_one_or_none()
    if document is None:
        document = ExternalDocument(
            connector_id=connector.id,
            scope_id=scope.id,
            corpus_id=change.corpus_id,
            external_id=change.external_id,
            name=change.name,
        )
        db.add(document)
        await db.flush()
    document.scope_id = scope.id
    # A Drive deletion carries no file resource at all: no name, no mime type, no URL.
    # Copying that emptiness over the row erased the very identity the audit trail and
    # the admin's job report need to say WHAT was removed, and left `mime_type` NULL so
    # the deletion was not even counted as a file. A tombstone updates state only.
    if change.state != "deleted" or change.metadata:
        document.name = change.name
        document.parent_external_id = change.parent_external_id
        document.mime_type = change.mime_type
        document.web_url = change.web_url
        document.revision = change.revision
    document.metadata_json = {
        **(document.metadata_json or {}),
        **(change.metadata or {}),
    }
    document.state = change.state
    return document


async def _save_permissions(
    db: AsyncSession,
    connector: Connector,
    document: ExternalDocument,
    permissions: list[dict[str, str]],
) -> bool:
    acl_hash = _acl_hash(permissions)
    acl_changed = document.acl_hash != acl_hash
    if acl_changed:
        await db.execute(
            PermissionSnapshot.__table__.update()
            .where(PermissionSnapshot.external_document_id == document.id)
            .values(active=False)
        )
        snapshot = (
            await db.execute(
                select(PermissionSnapshot).where(
                    PermissionSnapshot.external_document_id == document.id,
                    PermissionSnapshot.acl_hash == acl_hash,
                )
            )
        ).scalar_one_or_none()
        if snapshot is None:
            snapshot = PermissionSnapshot(
                external_document_id=document.id,
                acl_hash=acl_hash,
                permissions_json=permissions,
                active=True,
            )
            db.add(snapshot)
            await db.flush()
            for item in permissions:
                db.add(
                    ExternalAclPrincipal(
                        permission_snapshot_id=snapshot.id,
                        principal_type=item.get("principal_type", "user"),
                        principal_id=item.get("principal_id", ""),
                        role=item.get("role", "reader"),
                    )
                )
        else:
            snapshot.active = True
        document.acl_hash = acl_hash
    # Mapping/identity changes must reconcile an unchanged provider ACL too.
    # Keep these lists deterministic so a repeated sync is idempotent.
    group_ids = sorted(
        {
            item.get("principal_id", "")
            for item in permissions
            if item.get("principal_type") in {"group", "siteGroup"}
            and item.get("principal_id")
        }
    )
    mappings = (
        (
            await db.execute(
                select(ExternalGroupMapping)
                .join(
                    AccessGroup, AccessGroup.id == ExternalGroupMapping.access_group_id
                )
                .where(
                    ExternalGroupMapping.connector_id == connector.id,
                    ExternalGroupMapping.active.is_(True),
                    ExternalGroupMapping.external_group_id.in_(group_ids),
                    AccessGroup.company_domain == connector.company_domain,
                )
            )
        )
        .scalars()
        .all()
        if group_ids
        else []
    )
    user_ids = sorted(
        {
            item.get("principal_id", "")
            for item in permissions
            if item.get("principal_type") in {"user", "siteUser"}
            and item.get("principal_id")
        }
    )
    identity_provider = (
        "microsoft_entra" if connector.system == "sharepoint" else connector.system
    )
    identities = (
        (
            await db.execute(
                select(ExternalIdentity)
                .join(User, User.id == ExternalIdentity.user_id)
                .where(
                    ExternalIdentity.provider == identity_provider,
                    (
                        ExternalIdentity.subject.in_(user_ids)
                        | func.lower(ExternalIdentity.email).in_({item.lower() for item in user_ids})
                    ),
                    User.company_domain == connector.company_domain,
                    User.active.is_(True),
                )
            )
        )
        .scalars()
        .all()
        if user_ids
        else []
    )
    mapped_user_ids = {str(item.subject): str(item.user_id) for item in identities}
    mapped_user_ids.update(
        {
            str(item.email).lower(): str(item.user_id)
            for item in identities
            if item.email
        }
    )
    # Google Drive ACLs commonly expose an email address without an external
    # identity row. Map it to the active tenant-local account when available.
    if connector.system == "google_drive" and user_ids:
        email_users = (
            await db.execute(
                select(User).where(
                    func.lower(User.email).in_({item.lower() for item in user_ids}),
                    User.company_domain == connector.company_domain,
                    User.active.is_(True),
                )
            )
        ).scalars().all()
        mapped_user_ids.update(
            {str(item.email).lower(): str(item.id) for item in email_users}
        )
    unmapped_principal_ids = sorted(
        f"{item.get('principal_type', 'unknown')}:{item.get('principal_id', '')}"
        for item in permissions
        if item.get("principal_type") not in {"group", "siteGroup", "user", "siteUser"}
        and item.get("principal_id")
    )
    previous_metadata = document.metadata_json or {}
    acl_present_key = (
        "sharepoint_acl_present"
        if connector.system == "sharepoint"
        else "provider_acl_present"
    )
    next_metadata = {
        **previous_metadata,
        acl_present_key: True,
        "provider_acl_present": True,
        "mapped_access_group_ids": sorted(
            {str(item.access_group_id) for item in mappings}
        ),
        "unmapped_group_ids": sorted(
            item
            for item in group_ids
            if item not in {mapping.external_group_id for mapping in mappings}
        ),
        "mapped_source_user_ids": sorted(
            mapped_user_ids[item] for item in user_ids if item in mapped_user_ids
        ),
        "unmapped_source_user_ids": sorted(
            item for item in user_ids if item not in mapped_user_ids
        ),
        "unmapped_principal_ids": unmapped_principal_ids,
    }
    document.metadata_json = next_metadata
    mapping_changed = any(
        previous_metadata.get(key) != next_metadata.get(key)
        for key in (
            "sharepoint_acl_present",
            "provider_acl_present",
            "mapped_access_group_ids",
            "unmapped_group_ids",
            "mapped_source_user_ids",
            "unmapped_source_user_ids",
            "unmapped_principal_ids",
        )
    )
    return acl_changed or mapping_changed


async def reconcile_connector_acl_mappings(
    db: AsyncSession, connector: Connector
) -> list[uuid.UUID]:
    """Reapply stored provider ACL snapshots after mapping configuration changes."""
    documents = (
        (
            await db.execute(
                select(ExternalDocument).where(
                    ExternalDocument.connector_id == connector.id,
                    ExternalDocument.state != "deleted",
                )
            )
        )
        .scalars()
        .all()
    )
    changed_article_ids: list[uuid.UUID] = []
    for document in documents:
        snapshot = (
            await db.execute(
                select(PermissionSnapshot)
                .where(
                    PermissionSnapshot.external_document_id == document.id,
                    PermissionSnapshot.active.is_(True),
                )
                .order_by(PermissionSnapshot.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if snapshot is None or not isinstance(snapshot.permissions_json, list):
            continue
        changed = await _save_permissions(
            db, connector, document, snapshot.permissions_json
        )
        if changed and document.article_id:
            await _apply_mapped_groups(db, connector, document)
            changed_article_ids.append(document.article_id)
    await db.flush()
    return changed_article_ids


def _sharepoint_acl_intersection(
    *,
    internal_visibility: str,
    internal_group_ids: set[str],
    internal_user_ids: set[str],
    source_group_ids: set[str],
    source_user_ids: set[str],
    source_group_member_ids: set[str],
    unmapped_principals: bool,
    acl_present: bool,
) -> dict[str, set[str] | str]:
    """Calculate the restrictive intersection before touching ORM state."""
    effective_group_ids = (
        set()
        if internal_visibility == "users"
        else (
            source_group_ids
            if internal_visibility == "public" or not internal_group_ids
            else source_group_ids & internal_group_ids
        )
    )
    effective_direct_user_ids = (
        source_user_ids
        if internal_visibility == "public"
        else source_user_ids & internal_user_ids
    )
    source_restricts = acl_present or bool(
        source_group_ids or source_user_ids or unmapped_principals
    )
    if not source_restricts:
        visibility = internal_visibility
    elif not effective_group_ids and not effective_direct_user_ids:
        visibility = "users"
    elif effective_direct_user_ids and not effective_group_ids:
        visibility = "users"
    elif internal_visibility == "users":
        visibility = "users"
    else:
        visibility = "department"
    internal_users_allowed_by_source = (
        internal_user_ids & (source_user_ids | source_group_member_ids)
        if source_restricts
        else set(internal_user_ids)
    )
    return {
        "group_ids": effective_group_ids,
        "direct_user_ids": effective_direct_user_ids,
        "internal_users_allowed_by_source": internal_users_allowed_by_source,
        "visibility": visibility,
    }


async def _apply_mapped_groups(
    db: AsyncSession, connector: Connector, document: ExternalDocument
) -> None:
    if not document.article_id:
        return
    article = (
        await db.execute(
            select(Article)
            .where(Article.id == document.article_id)
            .options(
                selectinload(Article.access_groups),
                selectinload(Article.user_permissions),
            )
        )
    ).scalar_one_or_none()
    if not article:
        return
    metadata = document.metadata_json or {}
    # Preserve the internal policy once. Future source updates can therefore
    # only narrow it, even when the provider ACL is changed repeatedly.
    # Older unit fixtures passed a lightweight connector object; retain the
    # historical SharePoint source marker for those callers while real
    # connectors use their provider name.
    permission_source = getattr(connector, "system", "sharepoint")
    if "internal_acl_snapshot" not in metadata:
        metadata["internal_acl_snapshot"] = {
            "visibility": article.visibility,
            "access_group_ids": [str(group.id) for group in article.access_groups],
            "allow_user_ids": [
                str(item.user_id)
                for item in article.user_permissions
                if item.effect == "allow" and item.source != permission_source
            ],
        }
    internal = metadata["internal_acl_snapshot"]
    source_group_ids = {
        str(item) for item in metadata.get("mapped_access_group_ids", [])
    }
    internal_group_ids = {str(item) for item in internal.get("access_group_ids", [])}
    source_user_ids = {str(item) for item in metadata.get("mapped_source_user_ids", [])}
    source_group_member_ids: set[str] = set()
    if source_group_ids:
        source_group_member_ids = {
            str(item)
            for item in (
                await db.execute(
                    select(user_groups.c.user_id)
                    .join(User, User.id == user_groups.c.user_id)
                    .where(
                        user_groups.c.group_id.in_(source_group_ids),
                        User.company_domain == article.company_domain,
                        User.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        }
    acl = _sharepoint_acl_intersection(
        internal_visibility=str(internal.get("visibility") or "department"),
        internal_group_ids=internal_group_ids,
        internal_user_ids={str(item) for item in internal.get("allow_user_ids", [])},
        source_group_ids=source_group_ids,
        source_user_ids=source_user_ids,
        source_group_member_ids=source_group_member_ids,
        unmapped_principals=bool(
            metadata.get("unmapped_group_ids")
            or metadata.get("unmapped_source_user_ids")
            or metadata.get("unmapped_principal_ids")
        ),
        acl_present=bool(
            metadata.get("provider_acl_present")
            or metadata.get("sharepoint_acl_present")
        ),
    )
    effective_group_ids = set(acl["group_ids"])
    article.access_groups = (
        list(
            (
                await db.execute(
                    select(AccessGroup).where(
                        AccessGroup.id.in_(effective_group_ids),
                        AccessGroup.company_domain == article.company_domain,
                    )
                )
            )
            .scalars()
            .all()
        )
        if effective_group_ids
        else []
    )

    # Remove only permissions generated by the previous SharePoint snapshot.
    await db.execute(
        delete(ArticleUserPermission).where(
            ArticleUserPermission.article_id == article.id,
            ArticleUserPermission.source == permission_source,
        )
    )
    effective_user_ids = set(acl["direct_user_ids"])
    for user_id in effective_user_ids:
        db.add(
            ArticleUserPermission(
                article_id=article.id,
                user_id=uuid.UUID(user_id),
                effect="allow",
                source=permission_source,
            )
        )
    for user_id in set(internal.get("allow_user_ids", [])) - set(
        acl["internal_users_allowed_by_source"]
    ):
        db.add(
            ArticleUserPermission(
                article_id=article.id,
                user_id=uuid.UUID(str(user_id)),
                effect="deny",
                source=permission_source,
            )
        )
    article.visibility = str(acl["visibility"])
    document.metadata_json = metadata
    await db.flush()


async def _ingest_content(
    db: AsyncSession,
    connector: Connector,
    document: ExternalDocument,
    change: NormalizedChange,
    job: ConnectorJob,
    cleanup_keys: list[str] | None = None,
) -> None:
    adapter = adapter_for(connector)
    data = await adapter.download(change)
    content_hash = hashlib.sha256(data).hexdigest()
    if document.content_hash == content_hash and document.revision == change.revision:
        # Older connector drafts may predate F23. Repair their candidate
        # envelope even when the provider reports no content revision.
        existing_draft = (
            await db.execute(
                select(PendingDraft).where(
                    PendingDraft.external_document_id == document.id,
                    PendingDraft.status == "pending",
                )
            )
        ).scalar_one_or_none()
        if existing_draft and not getattr(existing_draft, "candidates", None):
            pages = await asyncio.to_thread(extract_source_pages, change.name, data)
            text = await asyncio.to_thread(
                extract_source_markdown, change.name, data, pages
            )
            existing_draft.summary = text
            existing_draft.page_texts = pages
            await _replace_split_candidates(db, existing_draft, text)
            await db.flush()
        return
    pages = await asyncio.to_thread(extract_source_pages, change.name, data)
    text = await asyncio.to_thread(extract_source_markdown, change.name, data, pages)
    storage_key = await asyncio.to_thread(
        save_source, content_hash, change.name, data, connector.company_domain
    )
    document.content_hash = content_hash
    document.revision = change.revision
    document.state = "active"
    version = (
        await db.execute(
            select(DocumentVersion).where(
                DocumentVersion.external_document_id == document.id,
                DocumentVersion.revision == (change.revision or content_hash),
            )
        )
    ).scalar_one_or_none()
    if version is None:
        version = DocumentVersion(
            external_document_id=document.id,
            revision=change.revision or content_hash,
            content_hash=content_hash,
            storage_key=storage_key,
            parser_version="source-extraction-v1",
            chunker_version="parent-child-v1",
            status="ready",
        )
        db.add(version)
    if document.article_id:
        article = (
            await db.execute(
                select(Article)
                .where(Article.id == document.article_id)
                .options(selectinload(Article.access_groups), selectinload(Article.sources))
            )
        ).scalar_one_or_none()
        if article and article.lifecycle_status == "active":
            article.source_changed = True
            article.source_changed_at = datetime.utcnow()
            article.source_previous_hash = next((source.source_hash for source in getattr(article, "sources", []) if source.source_system == connector.system), None)
            # Connector content is external input and must pass the same
            # independent approval path as a manually submitted revision.
            existing = (
                await db.execute(
                    select(PendingDraft).where(
                        PendingDraft.external_document_id == document.id,
                        PendingDraft.status == "pending",
                    )
                )
            ).scalar_one_or_none()
            metadata = {
                "domain": article.domain,
                "type": article.type,
                "sensitivity": article.sensitivity,
                "language": article.language,
                "access_group_ids": [str(group.id) for group in article.access_groups],
                "submission_kind": "connector_update",
                "suggested_update_article_id": str(article.id),
            }
            if existing is None:
                await _persist_connector_draft(
                    db,
                    connector,
                    PendingDraft(
                        title=change.name.rsplit(".", 1)[0][:255],
                        company_domain=connector.company_domain,
                        dept=article.dept,
                        source_ref=f"{connector.system}://{change.corpus_id}/{change.external_id}",
                        source_hash=content_hash,
                        summary=text,
                        restructured_body_md=text,
                        restructure_status="not_requested",
                        storage_key=storage_key,
                        original_filename=change.name,
                        mime_type=change.mime_type,
                        page_texts=pages,
                        status="draft",
                        created_by=connector.created_by,
                        external_document_id=document.id,
                        update_target_article_id=article.id,
                        content_metadata=metadata,
                    ),
                    text,
                )
            else:
                previous_storage_key = existing.storage_key
                existing.title = change.name.rsplit(".", 1)[0][:255]
                existing.source_hash = content_hash
                existing.summary = text
                existing.restructured_body_md = text
                existing.storage_key = storage_key
                existing.page_texts = pages
                existing.original_filename = change.name
                existing.content_metadata = metadata
                await _replace_split_candidates(db, existing, text)
                if (
                    cleanup_keys is not None
                    and previous_storage_key
                    and previous_storage_key != storage_key
                ):
                    cleanup_keys.append(previous_storage_key)
            return
    existing = (
        await db.execute(
            select(PendingDraft).where(
                PendingDraft.external_document_id == document.id,
                PendingDraft.status == "pending",
            )
        )
    ).scalar_one_or_none()
    routing = connector.config_json or {}
    department_ids = [str(item) for item in routing.get("department_ids", [])]
    department_names = [str(item) for item in routing.get("department_names", [])]
    draft_metadata = (
        {
            "department_ids": department_ids,
            "department_names": department_names,
            "submission_kind": "connector_import",
        }
        if department_ids
        else None
    )
    if existing is None:
        await _persist_connector_draft(
            db,
            connector,
            PendingDraft(
                title=change.name.rsplit(".", 1)[0][:255],
                company_domain=connector.company_domain,
                dept=department_names[0] if department_names else None,
                source_ref=f"{connector.system}://{change.corpus_id}/{change.external_id}",
                source_hash=content_hash,
                summary=text,
                restructured_body_md=text,
                restructure_status="lossless_ready",
                restructure_model="connector-source",
                storage_key=storage_key,
                original_filename=change.name,
                mime_type=change.mime_type,
                page_texts=pages,
                status="draft",
                created_by=connector.created_by,
                external_document_id=document.id,
                content_metadata=draft_metadata,
            ),
            text,
        )
    else:
        previous_storage_key = existing.storage_key
        existing.source_hash = content_hash
        existing.summary = text
        existing.storage_key = storage_key
        existing.page_texts = pages
        existing.original_filename = change.name
        await _replace_split_candidates(db, existing, text)
        if (
            not existing.dept
            and department_names
            and not (existing.content_metadata or {}).get("department_ids")
        ):
            existing.dept = department_names[0]
            existing.content_metadata = draft_metadata
        if (
            cleanup_keys is not None
            and previous_storage_key
            and previous_storage_key != storage_key
        ):
            cleanup_keys.append(previous_storage_key)


async def sync_cloud_connector(
    db: AsyncSession,
    connector: Connector,
    job: ConnectorJob,
    scope_id: uuid.UUID | None = None,
) -> None:
    adapter = adapter_for(connector)
    await ensure_connector_authorized(db, connector)
    scope_query = select(SourceScope).where(
        SourceScope.connector_id == connector.id,
        SourceScope.selected.is_(True),
    )
    if scope_id is not None:
        scope_query = scope_query.where(SourceScope.id == scope_id)
    scopes = (await db.execute(scope_query)).scalars().all()
    if not scopes:
        raise ConnectorProviderError(
            "No connector scopes are selected", retryable=False, code="no_scopes"
        )
    scope_ids = [scope.id for scope in scopes]
    summary: dict[str, object] = {
        "scope_count": len(scopes),
        "scopes": [],
        "changes_seen": 0,
        "files_seen": 0,
        "imported": 0,
        "updated": 0,
        "deleted": 0,
        "unchanged": 0,
        "skipped": 0,
        "failed": 0,
        "permissions_updated": 0,
        "items": [],
        "errors": [],
    }
    job.status = "running"
    job.attempts += 1
    job.summary_json = summary
    await db.commit()
    connector_id = connector.id
    job_id = job.id
    cleanup_keys: list[str] = []
    audit_actor_id = job.requested_by or connector.created_by
    try:
        # Iterated by id, not by instance. A single item's failure rolls the session
        # back, which expires every object loaded before it — including the ones this
        # loop would otherwise keep using for the rest of the walk.
        for scope_id_value in scope_ids:
            scope = await db.get(SourceScope, scope_id_value)
            if scope is None:
                continue
            scope_summary = {
                "scope_id": str(scope.id),
                "name": scope.display_name,
                "changes": 0,
            }
            cast_scopes = summary["scopes"]
            if isinstance(cast_scopes, list):
                cast_scopes.append(scope_summary)
            cursor_row = (
                await db.execute(
                    select(SyncCursor).where(
                        SyncCursor.connector_id == connector.id,
                        SyncCursor.scope_id == scope.id,
                    )
                )
            ).scalar_one_or_none()
            reconciliation_requested = not cursor_row or bool(cursor_row.full_sync_required)
            # A sweep may only conclude "absent means deleted" from a walk that actually
            # enumerates the scope. Google's changes feed does not: with no cursor it
            # starts at startPageToken and reports changes from NOW on, so reading its
            # near-empty result as the full corpus would delete every indexed document
            # on every reconciliation pass.
            was_full_reconciliation = (
                reconciliation_requested and adapter.full_walk_is_authoritative
            )
            # And such a provider keeps its cursor: discarding it to "start over" asks for
            # changes since now and silently skips everything since the last run.
            cursor = (
                None
                if was_full_reconciliation
                else (cursor_row.cursor_value if cursor_row else None)
            )
            try:
                changes, next_cursor = await adapter.incremental_changes(
                    {
                        "external_scope_id": scope.external_scope_id,
                        "config": scope.config_json or {},
                    },
                    cursor,
                )
            except ConnectorProviderError as exc:
                if str(exc.code or "") in {"410", "resync_required", "sync_state_not_found", "invalid_delta"}:
                    if cursor_row is None:
                        cursor_row = SyncCursor(
                            connector_id=connector.id,
                            scope_id=scope.id,
                            cursor_type=("delta" if connector.system == "sharepoint" else "changes"),
                        )
                        db.add(cursor_row)
                    cursor_row.cursor_value = None
                    cursor_row.status = "invalid"
                    cursor_row.full_sync_required = True
                    cursor_row.last_error = str(exc)[:2000]
                    await db.commit()
                    raise ConnectorProviderError(
                        "Provider delta state expired; a full reconciliation has been scheduled",
                        retryable=True,
                        code="resync_required",
                    ) from exc
                raise
            scope_display_name = scope.display_name
            consecutive_failures = 0
            for change in changes:
                summary["changes_seen"] = int(summary["changes_seen"]) + 1
                scope_summary["changes"] = int(scope_summary["changes"]) + 1
                previous = (
                    await db.execute(
                        select(ExternalDocument).where(
                            ExternalDocument.connector_id == connector_id,
                            ExternalDocument.corpus_id == change.corpus_id,
                            ExternalDocument.external_id == change.external_id,
                        )
                    )
                ).scalar_one_or_none()
                if change.state == "deleted" and previous is None:
                    # A tombstone for something never indexed. Materializing a row just
                    # to mark it deleted would fill the corpus with objects the KB never
                    # held - and Drive folder scopes now forward every deletion in the
                    # drive precisely because only this layer knows what is in scope.
                    continue
                previous_revision = previous.revision if previous else None
                # The provider's own mime type is missing on a tombstone; the row we
                # already hold remembers whether this used to be a file.
                effective_mime = change.mime_type or (
                    previous.mime_type if previous else None
                )
                is_file = bool(effective_mime and not effective_mime.endswith(".folder"))
                unsupported = (
                    _unsupported_reason(change.name, change.mime_type)
                    if is_file and change.state != "deleted"
                    else None
                )
                action = "unchanged"
                reason: str | None = None
                try:
                    document = await _upsert_document(db, connector, scope, change)
                    # Folders never become articles, and an item we will never index
                    # never needs an ACL snapshot. Skipping both removes one provider
                    # round trip per folder from every reconciliation of every drive.
                    fetch_permissions = (
                        change.state != "deleted" and is_file and unsupported is None
                    )
                    permissions = (
                        await adapter.permissions(change) if fetch_permissions else []
                    )
                    acl_changed = (
                        await _save_permissions(db, connector, document, permissions)
                        if fetch_permissions
                        else False
                    )
                    if is_file:
                        summary["files_seen"] = int(summary["files_seen"]) + 1
                    # Durable BEFORE the download. An ingest failure rolls back, and if
                    # the row were still uncommitted the rollback would take it with
                    # it - leaving nowhere to record the quarantine, so a newly added
                    # broken file would fail forever, once per sync, in perpetuity.
                    await db.commit()
                    document_id = document.id
                    pending_draft_needs_candidates = False
                    if is_file and unsupported is None and change.state != "deleted":
                        pending_draft_id = await db.scalar(
                            select(PendingDraft.id)
                            .where(
                                PendingDraft.external_document_id == document_id,
                                PendingDraft.status == "pending",
                            )
                            .limit(1)
                        )
                        if pending_draft_id is not None:
                            candidate_id = await db.scalar(
                                select(DraftCandidate.id)
                                .where(DraftCandidate.draft_id == pending_draft_id)
                                .limit(1)
                            )
                            pending_draft_needs_candidates = candidate_id is None
                    if change.state == "deleted":
                        action = "deleted"
                        if is_file:
                            summary["deleted"] = int(summary["deleted"]) + 1
                        deleted_keys, deleted_article_id = await _handle_deleted_document(
                            db, document, audit_actor_id
                        )
                        await db.commit()
                        await _cleanup_unreferenced_source_keys(db, deleted_keys)
                        if deleted_article_id:
                            await event_bus.publish(
                                "ArticleDeleted", {"article_id": str(deleted_article_id)}
                            )
                    elif unsupported is not None:
                        action = "skipped"
                        reason = unsupported
                        summary["skipped"] = int(summary["skipped"]) + 1
                    elif _is_quarantined(document, change):
                        action = "skipped"
                        reason = (
                            "Import failed "
                            f"{_MAX_ITEM_ATTEMPTS} times for this version; not retried "
                            "until the file changes at the source. Last error: "
                            f"{_ingest_failure(document).get('reason')}"
                        )
                        summary["skipped"] = int(summary["skipped"]) + 1
                    elif _needs_content_ingest(
                        is_file=is_file,
                        previous_exists=previous is not None,
                        previous_revision=previous_revision,
                        current_revision=change.revision,
                        has_content_hash=bool(document.content_hash),
                        pending_draft_needs_candidates=pending_draft_needs_candidates,
                    ):
                        await _ingest_content(
                            db, connector, document, change, job, cleanup_keys
                        )
                        _clear_ingest_failure(document)
                        action = "imported" if previous is None else "updated"
                        if is_file:
                            summary[action] = int(summary[action]) + 1
                    if acl_changed and document.article_id:
                        await _apply_mapped_groups(db, connector, document)
                        summary["permissions_updated"] = (
                            int(summary["permissions_updated"]) + 1
                        )
                        _record_permission_change_audits(
                            db, [document.article_id], audit_actor_id
                        )
                        await db.commit()
                        await event_bus.publish(
                            "PermissionChanged", {"article_id": str(document.article_id)}
                        )
                    if action == "unchanged" and is_file:
                        summary["unchanged"] = int(summary["unchanged"]) + 1
                    await db.commit()
                    consecutive_failures = 0
                except Exception as exc:
                    if not _is_item_level_failure(exc):
                        raise
                    # One unreadable, corrupt or oversized document used to abort the
                    # entire walk. The cursor was then never advanced, so the next run
                    # met the same document and died in the same place: a single bad
                    # file froze the whole connector, permanently and silently, and the
                    # healthy documents behind it never arrived at all.
                    await db.rollback()
                    reason = str(exc) or exc.__class__.__name__
                    attempts = await _record_item_failure(
                        db,
                        connector_id=connector_id,
                        job_id=job_id,
                        change=change,
                        reason=reason,
                        code=getattr(exc, "code", None) or exc.__class__.__name__,
                        retryable=bool(getattr(exc, "retryable", True)),
                    )
                    action = "failed"
                    summary["failed"] = int(summary["failed"]) + 1
                    consecutive_failures += 1
                    logger.warning(
                        "Connector item ingest failed",
                        connector_id=str(connector_id),
                        external_id=change.external_id,
                        attempts=attempts,
                        error=reason,
                    )
                    if consecutive_failures >= _MAX_CONSECUTIVE_ITEM_FAILURES:
                        raise ConnectorProviderError(
                            f"Aborted after {consecutive_failures} consecutive item "
                            f"failures; last error: {reason}",
                            retryable=True,
                            code="too_many_item_failures",
                        ) from exc
                    # The rollback expired everything loaded before it. Reload the three
                    # objects the rest of the walk still writes through.
                    connector = await db.get(Connector, connector_id)
                    job = await db.get(ConnectorJob, job_id)
                    scope = await db.get(SourceScope, scope_id_value)
                    if connector is None or job is None or scope is None:
                        raise
                if is_file:
                    _note_item_outcome(
                        summary,
                        name=change.name,
                        action=action,
                        scope_name=scope_display_name,
                        web_url=change.web_url,
                        reason=reason,
                    )
            # `changes` must be non-empty: a full walk of a scope that holds indexed
            # documents always reports them, so an empty result means the walk did not
            # happen (a truncated page, a revoked scope) rather than an empty drive.
            if was_full_reconciliation and changes:
                # A full delta walk is authoritative for the selected scope.
                # Items absent from it were removed or are no longer visible;
                # mark them deleted so stale KB articles cannot survive forever.
                seen_external_ids = {change.external_id for change in changes}
                existing_documents = (
                    await db.execute(
                        select(ExternalDocument).where(
                            ExternalDocument.connector_id == connector_id,
                            ExternalDocument.scope_id == scope_id_value,
                            ExternalDocument.state != "deleted",
                        )
                    )
                ).scalars().all()
                for stale_document in existing_documents:
                    if stale_document.external_id in seen_external_ids:
                        continue
                    stale_keys, stale_article_id = await _handle_deleted_document(
                        db, stale_document, audit_actor_id
                    )
                    cleanup_keys.extend(stale_keys)
                    if stale_article_id:
                        await event_bus.publish(
                            "ArticleDeleted", {"article_id": str(stale_article_id)}
                        )
            # Re-read rather than reuse the instance loaded before the walk: an item
            # failure rolls the session back, and the cursor loaded up there would be
            # an expired object whose first attribute access raises mid-write.
            cursor_row = (
                await db.execute(
                    select(SyncCursor).where(
                        SyncCursor.connector_id == connector_id,
                        SyncCursor.scope_id == scope_id_value,
                    )
                )
            ).scalar_one_or_none()
            if cursor_row is None:
                cursor_row = SyncCursor(
                    connector_id=connector_id,
                    scope_id=scope_id_value,
                    cursor_type=(
                        "delta" if connector.system == "sharepoint" else "changes"
                    ),
                )
                db.add(cursor_row)
            cursor_row.cursor_value = next_cursor or cursor_row.cursor_value
            cursor_row.last_success_at = datetime.utcnow()
            cursor_row.status = "ready"
            cursor_row.full_sync_required = False
            # Stamped when a reconciliation pass RAN, not only when the provider could
            # walk authoritatively. The inline dispatcher throttles on this field, so
            # leaving it NULL for a provider that cannot enumerate makes it re-enqueue
            # that scope on every tick.
            cursor_row.last_reconcile_at = (
                datetime.utcnow() if reconciliation_requested else cursor_row.last_reconcile_at
            )
            cursor_row.last_error = None
            job.summary_json = summary
            await db.commit()
            await _cleanup_unreferenced_source_keys(db, cleanup_keys)
            cleanup_keys.clear()
        recovered = connector.status == "error"
        connector.last_sync = datetime.utcnow()
        connector.status = "active"
        connector.last_error = None
        if recovered:
            _notify_connector_state_change(
                db, connector, event="connector_sync_recovered", detail=None
            )
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.summary_json = summary
        db.add(
            AuditLog(
                user_id=job.requested_by or connector.created_by,
                action="sync",
                target_type="connector_job",
                target_id=str(job.id),
                outcome="success",
            )
        )
        await db.commit()
    except Exception as exc:
        # Start the failure record on a clean transaction. Whatever raised may have left
        # this one aborted, and Postgres rejects every further statement in an aborted
        # transaction — the commit below would then fail too and the connector would end
        # the run with no error recorded anywhere and its status still "active".
        await db.rollback()
        connector = await db.get(Connector, connector_id)
        job = await db.get(ConnectorJob, job_id)
        if connector is None or job is None:
            raise
        newly_broken = connector.status != "error"
        connector.status = "error"
        connector.last_error = str(exc)[:2000]
        if newly_broken:
            _notify_connector_state_change(
                db, connector, event="connector_sync_failed", detail=str(exc)
            )
        job.status = "failed"
        job.last_error = str(exc)[:2000]
        job.summary_json = summary
        db.add(
            SyncError(
                connector_id=connector_id,
                job_id=job_id,
                stage="sync",
                error_code=getattr(exc, "code", None),
                message=str(exc)[:4000],
                retryable=bool(getattr(exc, "retryable", True)),
                attempts=job.attempts,
            )
        )
        db.add(
            AuditLog(
                user_id=job.requested_by or connector.created_by,
                action="sync",
                target_type="connector_job",
                target_id=str(job_id),
                outcome="failure",
            )
        )
        await db.commit()
        raise
