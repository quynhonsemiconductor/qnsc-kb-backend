"""Add durable connector notifications and coalesced sync requests."""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# Every other connector child table is tenant-isolated through its connector_id — see
# 20260807_28. These two carry connector_id as well, so they get the same policy: without
# it, one tenant's queue and notification inbox are readable by every other tenant.
_RLS_TABLES = (
    ("connector_notifications", "tenant_connector_notifications"),
    ("sync_requests", "tenant_sync_requests"),
)
_TENANT_EXPRESSION = (
    "current_setting('app.global_admin', true) = 'true' "
    "OR current_setting('app.global_connector_access', true) = 'true' "
    "OR EXISTS (SELECT 1 FROM connectors c WHERE c.id = connector_id "
    "AND c.company_domain = current_setting('app.company_domain', true))"
)


def _rls_enabled() -> bool:
    return os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}


revision = "20260820_63"
down_revision = "20260819_62"
branch_labels = None
depends_on = None


def _add_missing(table: str, *columns: sa.Column) -> list[str]:
    """Add only the columns this database does not already have.

    See 20260814_52: the baseline revision builds the schema with
    ``Base.metadata.create_all`` against the LIVE models, so on a database created from
    empty every column below already exists and a bare add_column fails. An existing
    database predates them and does need the ALTER.
    """
    existing = {item["name"] for item in inspect(op.get_bind()).get_columns(table)}
    added = []
    for column in columns:
        if column.name not in existing:
            op.add_column(table, column)
            added.append(column.name)
    return added


def upgrade() -> None:
    added = _add_missing(
        "sync_cursors",
        sa.Column("status", sa.String(length=30), nullable=False, server_default="ready"),
        sa.Column("full_sync_required", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_notification_at", sa.DateTime(), nullable=True),
        sa.Column("last_reconcile_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    for column in ("status", "full_sync_required"):
        if column in added:
            op.alter_column("sync_cursors", column, server_default=None)

    added = _add_missing(
        "webhook_subscriptions",
        sa.Column("resource", sa.String(length=1024), nullable=True),
        sa.Column("lifecycle_notification_url", sa.String(length=2048), nullable=True),
        sa.Column("reauthorization_required", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_notification_at", sa.DateTime(), nullable=True),
        sa.Column("last_lifecycle_at", sa.DateTime(), nullable=True),
        sa.Column("renewal_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    for column in ("reauthorization_required", "renewal_attempts"):
        if column in added:
            op.alter_column("webhook_subscriptions", column, server_default=None)

    if not inspect(op.get_bind()).has_table("connector_notifications"):
        op.create_table(
            "connector_notifications",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("connector_id", sa.Uuid(), nullable=False),
            sa.Column("scope_id", sa.Uuid(), nullable=True),
            sa.Column("provider", sa.String(length=50), nullable=False),
            sa.Column("subscription_id", sa.String(length=512), nullable=False),
            sa.Column("payload_hash", sa.String(length=64), nullable=False),
            sa.Column("resource", sa.String(length=2048), nullable=True),
            sa.Column("change_type", sa.String(length=30), nullable=True),
            sa.Column("lifecycle_event", sa.String(length=50), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=30), nullable=False, server_default="received"),
            sa.Column("received_at", sa.DateTime(), nullable=False),
            sa.Column("processed_at", sa.DateTime(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["connector_id"], ["connectors.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["scope_id"], ["source_scopes.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("provider", "subscription_id", "payload_hash", name="uq_connector_notification_delivery"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_connector_notifications_status_received ON connector_notifications (status, received_at)")

    if not inspect(op.get_bind()).has_table("sync_requests"):
        op.create_table(
                "sync_requests",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("connector_id", sa.Uuid(), nullable=False),
            sa.Column("scope_id", sa.Uuid(), nullable=True),
            sa.Column("job_id", sa.Uuid(), nullable=True),
            sa.Column("reason", sa.String(length=80), nullable=False, server_default="notification"),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="50"),
            sa.Column("status", sa.String(length=30), nullable=False, server_default="queued"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("available_at", sa.DateTime(), nullable=False),
            sa.Column("locked_at", sa.DateTime(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["connector_id"], ["connectors.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["scope_id"], ["source_scopes.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["job_id"], ["connector_jobs.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_sync_requests_dispatch ON sync_requests (status, available_at)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sync_requests_connector_status ON sync_requests (connector_id, status)")

    if _rls_enabled():
        for table, policy in _RLS_TABLES:
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            # Postgres has no CREATE POLICY IF NOT EXISTS.
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
            op.execute(
                f"CREATE POLICY {policy} ON {table} "
                f"USING ({_TENANT_EXPRESSION}) WITH CHECK ({_TENANT_EXPRESSION})"
            )


def downgrade() -> None:
    if _rls_enabled():
        for table, policy in _RLS_TABLES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    op.drop_index("ix_sync_requests_connector_status", table_name="sync_requests")
    op.drop_index("ix_sync_requests_dispatch", table_name="sync_requests")
    op.drop_table("sync_requests")
    op.drop_index("ix_connector_notifications_status_received", table_name="connector_notifications")
    op.drop_table("connector_notifications")
    for column in ("last_error", "renewal_attempts", "last_lifecycle_at", "last_notification_at", "reauthorization_required", "lifecycle_notification_url", "resource"):
        op.drop_column("webhook_subscriptions", column)
    for column in ("last_error", "last_reconcile_at", "last_notification_at", "full_sync_required", "status"):
        op.drop_column("sync_cursors", column)
