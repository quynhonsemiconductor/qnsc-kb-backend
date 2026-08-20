"""Incremental cloud connector synchronization and version handoff."""

from __future__ import annotations

import hashlib
import json
import asyncio
import uuid
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.core.secrets import encrypt_secret
from src.domain.connector_adapters import (
    ConnectorProviderError,
    NormalizedChange,
    adapter_for,
)
from src.domain.events import event_bus
from src.domain.source_extraction import extract_source_markdown, extract_source_pages
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
from src.models.user import User
from src.models.ops import Connector, ConnectorJob
from src.repositories.governance import GovernanceRepository
from src.repositories.article import ArticleRepository
from src.domain.governance import GovernanceService
from src.domain.document_splitter import split_document_candidates


async def _persist_connector_draft(
    db: AsyncSession, connector: Connector, draft: PendingDraft, text: str
) -> None:
    """Persist connector input as Draft, then submit through the same workflow."""
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
    for item in split_document_candidates(draft.title, text, page_texts=draft.page_texts):
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
    for item in split_document_candidates(draft.title, text, page_texts=draft.page_texts):
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
    db: AsyncSession, connector: Connector, job: ConnectorJob
) -> None:
    adapter = adapter_for(connector)
    if connector.oauth_expires_at and connector.oauth_expires_at <= datetime.utcnow():
        tokens = await adapter.refresh_token()
        connector.oauth_access_token = encrypt_secret(tokens.get("access_token"))
        connector.oauth_refresh_token = (
            encrypt_secret(tokens.get("refresh_token")) or connector.oauth_refresh_token
        )
        connector.oauth_expires_at = datetime.utcnow() + timedelta(
            seconds=int(tokens.get("expires_in", 3600))
        )
        await db.commit()
    scopes = (
        (
            await db.execute(
                select(SourceScope).where(
                    SourceScope.connector_id == connector.id,
                    SourceScope.selected.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    if not scopes:
        raise ConnectorProviderError(
            "No connector scopes are selected", retryable=False, code="no_scopes"
        )
    summary: dict[str, object] = {
        "scope_count": len(scopes),
        "scopes": [],
        "changes_seen": 0,
        "files_seen": 0,
        "imported": 0,
        "updated": 0,
        "deleted": 0,
        "unchanged": 0,
        "permissions_updated": 0,
        "items": [],
    }
    job.status = "running"
    job.attempts += 1
    job.summary_json = summary
    await db.commit()
    cleanup_keys: list[str] = []
    audit_actor_id = job.requested_by or connector.created_by
    try:
        for scope in scopes:
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
            cursor = cursor_row.cursor_value if cursor_row else None
            changes, next_cursor = await adapter.incremental_changes(
                {
                    "external_scope_id": scope.external_scope_id,
                    "config": scope.config_json or {},
                },
                cursor,
            )
            for change in changes:
                summary["changes_seen"] = int(summary["changes_seen"]) + 1
                scope_summary["changes"] = int(scope_summary["changes"]) + 1
                previous = (
                    await db.execute(
                        select(ExternalDocument).where(
                            ExternalDocument.connector_id == connector.id,
                            ExternalDocument.corpus_id == change.corpus_id,
                            ExternalDocument.external_id == change.external_id,
                        )
                    )
                ).scalar_one_or_none()
                previous_revision = previous.revision if previous else None
                document = await _upsert_document(db, connector, scope, change)
                permissions = (
                    []
                    if change.state == "deleted"
                    else await adapter.permissions(change)
                )
                acl_changed = (
                    await _save_permissions(db, connector, document, permissions)
                    if change.state != "deleted"
                    else False
                )
                is_file = bool(
                    change.mime_type and not change.mime_type.endswith(".folder")
                )
                if is_file:
                    summary["files_seen"] = int(summary["files_seen"]) + 1
                action = "unchanged"
                pending_draft_id = None
                pending_draft_needs_candidates = False
                if is_file:
                    pending_draft_id = await db.scalar(
                        select(PendingDraft.id)
                        .where(
                            PendingDraft.external_document_id == document.id,
                            PendingDraft.status == "pending",
                        )
                        .limit(1)
                    )
                    if pending_draft_id is not None:
                        candidate_id = await db.scalar(
                            select(DraftCandidate.id)
                            .where(
                                DraftCandidate.draft_id == pending_draft_id,
                            )
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
                items = summary["items"]
                if is_file and isinstance(items, list) and len(items) < 200:
                    items.append(
                        {
                            "name": change.name,
                            "action": action,
                            "scope": scope.display_name,
                            "web_url": change.web_url,
                        }
                    )
            if cursor_row is None:
                cursor_row = SyncCursor(
                    connector_id=connector.id,
                    scope_id=scope.id,
                    cursor_type=(
                        "delta" if connector.system == "sharepoint" else "changes"
                    ),
                )
                db.add(cursor_row)
            cursor_row.cursor_value = next_cursor or cursor_row.cursor_value
            cursor_row.last_success_at = datetime.utcnow()
            job.summary_json = summary
            await db.commit()
            await _cleanup_unreferenced_source_keys(db, cleanup_keys)
            cleanup_keys.clear()
        connector.last_sync = datetime.utcnow()
        connector.status = "active"
        connector.last_error = None
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
        connector.status = "error"
        connector.last_error = str(exc)[:2000]
        job.status = "failed"
        job.last_error = str(exc)[:2000]
        job.summary_json = summary
        db.add(
            SyncError(
                connector_id=connector.id,
                job_id=job.id,
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
                target_id=str(job.id),
                outcome="failure",
            )
        )
        await db.commit()
        raise
