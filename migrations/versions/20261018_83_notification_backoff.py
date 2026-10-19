"""notification_queue.next_attempt_at: bounded, backed-off email delivery

deliver_notification_queue retried every failed email every 30 seconds forever, ordered by creation time with a
limit of 100: a hundred permanently failing rows (a mailbox the app may not send as, for example) sat at the head
of every batch and starved all new mail. Each failure now schedules the next attempt (exponential backoff, or the
server's Retry-After), so failing rows leave the head of the queue, and a row older than 24 h or refused as
malformed becomes status "dead". A partial index on (next_attempt_at, created_at) serves the delivery query.

Nullable and NOT backfilled: NULL means "eligible now", which is exactly what every existing row is. The status
column is free text, so "dead" needs no schema change.

Guarded the way every column-add migration here is: the baseline revision builds tables from model metadata, so a
fresh database may already have the column.

Revision ID: 20261018_83
Revises: 20261011_82
Create Date: 2026-10-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261018_83"
down_revision = "20261011_82"
branch_labels = None
depends_on = None

TABLE = "notification_queue"
INDEX = "ix_notification_queue_email_due"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "next_attempt_at" not in columns:
        op.add_column(TABLE, sa.Column("next_attempt_at", sa.DateTime(), nullable=True))
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX in indexes and _index_is_invalid(op.get_bind()):
        # A CREATE INDEX CONCURRENTLY that failed or was cancelled leaves the index behind, INVALID: it is listed,
        # costs writes, and is never used. "Already exists" must not mean "done".
        with op.get_context().autocommit_block():
            op.drop_index(INDEX, table_name=TABLE, postgresql_concurrently=True)
        indexes.discard(INDEX)
    if INDEX not in indexes:
        # CONCURRENTLY: the table takes writes while the API runs. It cannot run inside a transaction.
        with op.get_context().autocommit_block():
            op.create_index(
                INDEX,
                TABLE,
                ["next_attempt_at", "created_at"],
                postgresql_where=sa.text("type = 'email' AND status IN ('pending', 'failed')"),
                postgresql_concurrently=True,
            )


def _index_is_invalid(bind) -> bool:
    """True when the index exists in pg_index with indisvalid = false (PostgreSQL only; other dialects have none)."""
    if bind.dialect.name != "postgresql":
        return False
    valid = bind.execute(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = :name"
        ),
        {"name": INDEX},
    ).scalar()
    return valid is False


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX in indexes:
        with op.get_context().autocommit_block():
            op.drop_index(INDEX, table_name=TABLE, postgresql_concurrently=True)
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "next_attempt_at" in columns:
        op.drop_column(TABLE, "next_attempt_at")
