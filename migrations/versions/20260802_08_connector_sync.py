"""add provider-independent connector synchronization state"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision = "20260802_08"
down_revision = "20260802_07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The baseline migration creates current SQLAlchemy metadata for fresh
    # installs.  A database on that path already has the entire connector
    # schema; replaying its historical create_table calls would fail.
    if inspect(op.get_bind()).has_table("source_scopes"):
        return
    for statement in (
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_subject VARCHAR(255)",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_access_token TEXT",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_refresh_token TEXT",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_expires_at TIMESTAMP",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_state_hash VARCHAR(64)",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS oauth_state_expires_at TIMESTAMP",
        "ALTER TABLE connectors ADD COLUMN IF NOT EXISTS last_error TEXT",
    ):
        op.execute(statement)
    op.create_table(
        "source_scopes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_scope_id", sa.String(255), nullable=False),
        sa.Column("scope_type", sa.String(40), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("selected", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.UniqueConstraint(
            "connector_id", "external_scope_id", name="uq_source_scope_external"
        ),
    )
    op.create_table(
        "sync_cursors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scope_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_scopes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cursor_type", sa.String(40), nullable=False),
        sa.Column("cursor_value", sa.Text(), nullable=True),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("connector_id", "scope_id", name="uq_sync_cursor_scope"),
    )
    op.create_table(
        "external_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scope_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_scopes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "article_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("articles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("corpus_id", sa.String(255), nullable=False),
        sa.Column("external_id", sa.String(512), nullable=False),
        sa.Column("parent_external_id", sa.String(512), nullable=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(150), nullable=True),
        sa.Column("web_url", sa.String(2048), nullable=True),
        sa.Column("revision", sa.String(255), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("acl_hash", sa.String(64), nullable=True),
        sa.Column("state", sa.String(30), server_default="active", nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.UniqueConstraint(
            "connector_id",
            "corpus_id",
            "external_id",
            name="uq_external_document_identity",
        ),
    )
    op.create_index(
        "ix_external_documents_connector_state",
        "external_documents",
        ["connector_id", "state"],
    )
    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "external_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("external_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=True),
        sa.Column("parser_version", sa.String(80), nullable=True),
        sa.Column("chunker_version", sa.String(80), nullable=True),
        sa.Column("status", sa.String(30), server_default="pending", nullable=False),
        sa.UniqueConstraint(
            "external_document_id", "revision", name="uq_document_version_revision"
        ),
    )
    op.create_table(
        "permission_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "external_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("external_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("acl_hash", sa.String(64), nullable=False),
        sa.Column("permissions_json", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.UniqueConstraint(
            "external_document_id", "acl_hash", name="uq_permission_snapshot_hash"
        ),
    )
    op.create_table(
        "external_acl_principals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "permission_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("permission_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("principal_type", sa.String(30), nullable=False),
        sa.Column("principal_id", sa.String(512), nullable=False),
        sa.Column("role", sa.String(40), nullable=False),
        sa.UniqueConstraint(
            "permission_snapshot_id",
            "principal_type",
            "principal_id",
            name="uq_external_acl_principal",
        ),
    )
    op.create_table(
        "external_group_mappings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_group_id", sa.String(512), nullable=False),
        sa.Column("external_group_name", sa.String(255), nullable=True),
        sa.Column(
            "access_group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("access_groups.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.UniqueConstraint(
            "connector_id", "external_group_id", name="uq_external_group_mapping"
        ),
    )
    op.create_table(
        "webhook_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scope_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_scopes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "provider_subscription_id", sa.String(512), unique=True, nullable=False
        ),
        sa.Column("verification_token_hash", sa.String(64), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
    )
    op.create_table(
        "sync_errors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connector_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "external_document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("external_documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("stage", sa.String(50), nullable=False),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("retryable", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
    )
    # The FK must be added after external_documents exists; this migration is
    # also used against databases created by the compatibility bootstrap.
    op.execute(
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS external_document_id UUID REFERENCES external_documents(id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE pending_drafts DROP COLUMN IF EXISTS external_document_id")
    for table in (
        "sync_errors",
        "webhook_subscriptions",
        "external_group_mappings",
        "external_acl_principals",
        "permission_snapshots",
        "document_versions",
        "external_documents",
        "sync_cursors",
        "source_scopes",
    ):
        op.drop_table(table)
    for statement in (
        "ALTER TABLE connectors DROP COLUMN IF EXISTS last_error",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_state_expires_at",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_state_hash",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_expires_at",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_refresh_token",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_access_token",
        "ALTER TABLE connectors DROP COLUMN IF EXISTS oauth_subject",
    ):
        op.execute(statement)
