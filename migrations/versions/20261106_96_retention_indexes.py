"""Indexes for the scheduled retention sweep

The daily retention task deletes by age. Without these indexes each batch is a sequential
scan; the partial ones cover only the terminal rows that are ever deleted, so live work
costs nothing.

Guarded because the 20260802_00 baseline builds the schema with `Base.metadata.create_all`
against the LIVE models, which already includes them on a freshly migrated database.

Revision ID: 20261106_96
Revises: 20261106_95
Create Date: 2026-11-06
"""

from alembic import op
import sqlalchemy as sa

revision = "20261106_96"
down_revision = "20261106_95"
branch_labels = None
depends_on = None

_INDEXES = (
    ("ix_audit_logs_created_at", "audit_logs", "created_at", None),
    ("ix_ai_usage_logs_created_at", "ai_usage_logs", "created_at", None),
    ("ix_search_logs_created_at", "search_logs", "created_at", None),
    (
        "ix_outbox_events_terminal_created",
        "outbox_events",
        "created_at",
        "status IN ('completed', 'dead')",
    ),
    (
        "ix_notification_queue_terminal_created",
        "notification_queue",
        "created_at",
        "status IN ('sent', 'dead')",
    ),
)


def _indexes() -> set[str]:
    return {
        index["name"]
        for table in {name for _, name, _, _ in _INDEXES}
        for index in sa.inspect(op.get_bind()).get_indexes(table)
    }


def upgrade() -> None:
    existing = _indexes()
    for name, table, column, where in _INDEXES:
        if name not in existing:
            extra = {"postgresql_where": sa.text(where)} if where else {}
            op.create_index(name, table, [column], **extra)


def downgrade() -> None:
    existing = _indexes()
    for name, table, _, _ in _INDEXES:
        if name in existing:
            op.drop_index(name, table_name=table)
