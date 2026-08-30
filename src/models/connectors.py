"""Provider-independent synchronization state for cloud connectors."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class SourceScope(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "source_scopes"
    __table_args__ = (UniqueConstraint("connector_id", "external_scope_id", name="uq_source_scope_external"),)

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    external_scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(40), nullable=False)  # site, drive, shared_drive, folder
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class SyncCursor(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "sync_cursors"
    __table_args__ = (UniqueConstraint("connector_id", "scope_id", name="uq_sync_cursor_scope"),)

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    scope_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_scopes.id", ondelete="CASCADE"), nullable=False)
    cursor_type: Mapped[str] = mapped_column(String(40), nullable=False)
    cursor_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ready", nullable=False)
    full_sync_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_notification_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_reconcile_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ExternalDocument(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "external_documents"
    __table_args__ = (
        UniqueConstraint("connector_id", "corpus_id", "external_id", name="uq_external_document_identity"),
        Index("ix_external_documents_connector_state", "connector_id", "state"),
    )

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_scopes.id", ondelete="SET NULL"), nullable=True)
    article_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("articles.id", ondelete="SET NULL"), nullable=True)
    corpus_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)
    parent_external_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(150), nullable=True)
    web_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    acl_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    state: Mapped[str] = mapped_column(String(30), default="active", nullable=False)  # active, deleted, inaccessible, pending
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class DocumentVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("external_document_id", "revision", name="uq_document_version_revision"),)

    external_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("external_documents.id", ondelete="CASCADE"), nullable=False)
    revision: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    chunker_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)


class PermissionSnapshot(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "permission_snapshots"
    __table_args__ = (UniqueConstraint("external_document_id", "acl_hash", name="uq_permission_snapshot_hash"),)

    external_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("external_documents.id", ondelete="CASCADE"), nullable=False)
    acl_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    permissions_json: Mapped[list | dict] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ExternalAclPrincipal(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "external_acl_principals"
    __table_args__ = (UniqueConstraint("permission_snapshot_id", "principal_type", "principal_id", name="uq_external_acl_principal"),)

    permission_snapshot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("permission_snapshots.id", ondelete="CASCADE"), nullable=False)
    principal_type: Mapped[str] = mapped_column(String(30), nullable=False)  # user, group, domain, anyone
    principal_id: Mapped[str] = mapped_column(String(512), nullable=False)
    #: What the provider calls this principal. Without it the only thing an
    #: administrator could be shown was a GUID, which is not something anyone can make
    #: an access decision about.
    principal_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(40), nullable=False)


class ExternalGroupMapping(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "external_group_mappings"
    __table_args__ = (UniqueConstraint("connector_id", "external_group_id", name="uq_external_group_mapping"),)

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    external_group_id: Mapped[str] = mapped_column(String(512), nullable=False)
    external_group_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    access_group_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("access_groups.id", ondelete="CASCADE"), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class WebhookSubscription(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "webhook_subscriptions"

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_scopes.id", ondelete="SET NULL"), nullable=True)
    provider_subscription_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    verification_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    lifecycle_notification_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reauthorization_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_notification_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_lifecycle_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    renewal_attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class SyncError(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "sync_errors"

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("connector_jobs.id", ondelete="SET NULL"), nullable=True)
    external_document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("external_documents.id", ondelete="SET NULL"), nullable=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    retryable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    attempts: Mapped[int] = mapped_column(default=0, nullable=False)


class ConnectorNotification(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Durable, deduplicated provider notification inbox."""

    __tablename__ = "connector_notifications"
    __table_args__ = (
        UniqueConstraint(
            "provider", "subscription_id", "payload_hash",
            name="uq_connector_notification_delivery",
        ),
        Index("ix_connector_notifications_status_received", "status", "received_at"),
    )

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_scopes.id", ondelete="SET NULL"), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    subscription_id: Mapped[str] = mapped_column(String(512), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    resource: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    change_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    lifecycle_event: Mapped[str | None] = mapped_column(String(50), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="received", nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class SyncRequest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Durable coalesced work request for one connector synchronization."""

    __tablename__ = "sync_requests"
    __table_args__ = (
        Index("ix_sync_requests_dispatch", "status", "available_at"),
        Index("ix_sync_requests_connector_status", "connector_id", "status"),
    )

    connector_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_scopes.id", ondelete="SET NULL"), nullable=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("connector_jobs.id", ondelete="SET NULL"), nullable=True)
    reason: Mapped[str] = mapped_column(String(80), nullable=False, default="notification")
    priority: Mapped[int] = mapped_column(default=50, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False)
    attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
