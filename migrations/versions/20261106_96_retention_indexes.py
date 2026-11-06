"""Indexes and terminal timestamps for the scheduled retention sweep

The daily sweep deletes by TERMINAL time, not enqueue time: a retried outbox event can
finish days later. `outbox_events` gains `completed_at` / `dead_at`; the sweep keys on
`COALESCE(completed_at, dead_at, created_at)`.

Indexes are built CONCURRENTLY so the API keeps writing while they build, and with a
short lock_timeout so a stuck transaction fails the migration instead of holding the
build open (same shape as 20261018_83). The baseline revision builds tables from the
LIVE models, so every object is guarded.

Revision ID: 20261106_96
Revises: 20261106_95
Create Date: 2026-11-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261106_96"
down_revision = "20261106_95"
branch_labels = None
depends_on = None

LOCK_TIMEOUT = "5s"

_INDEXES = (
    ("ix_ai_usage_logs_created_at", "ai_usage_logs", "created_at", None),
    ("ix_search_logs_created_at", "search_logs", "created_at", None),
)


def _columns(table: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table)}


def _indexes(table: str) -> set[str]:
    return {index["name"] for index in inspect(op.get_bind()).get_indexes(table)}


def _index_is_invalid(name: str) -> bool:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return False
    return (
        bind.execute(
            sa.text(
                "SELECT i.indisvalid FROM pg_index i JOIN pg_class c "
                "ON c.oid = i.indexrelid WHERE c.relname = :name"
            ),
            {"name": name},
        ).scalar()
        is False
    )


def upgrade() -> None:
    columns = _columns("outbox_events")
    for name in ("completed_at", "dead_at"):
        if name not in columns:
            op.add_column(
                "outbox_events", sa.Column(name, sa.DateTime(), nullable=True)
            )

    op.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    for name, table, column, where in _INDEXES:
        existing = _indexes(table)
        if name in existing and _index_is_invalid(name):
            with op.get_context().autocommit_block():
                op.drop_index(name, table_name=table, postgresql_concurrently=True)
            existing.discard(name)
        if name not in existing:
            extra = {"postgresql_where": sa.text(where)} if where else {}
            with op.get_context().autocommit_block():
                op.create_index(
                    name, table, [column], postgresql_concurrently=True, **extra
                )


def downgrade() -> None:
    op.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    for name, table, _, _ in _INDEXES:
        if name in _indexes(table):
            with op.get_context().autocommit_block():
                op.drop_index(name, table_name=table, postgresql_concurrently=True)
    columns = _columns("outbox_events")
    for name in ("dead_at", "completed_at"):
        if name in columns:
            op.drop_column("outbox_events", name)
