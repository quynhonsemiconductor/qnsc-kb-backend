"""notification_queue.next_attempt_at: bounded, backed-off email delivery

deliver_notification_queue retried every failed email every 30 seconds forever, ordered by creation time with a
limit of 100: a hundred permanently failing rows (a mailbox the app may not send as, for example) sat at the head
of every batch and starved all new mail. Each failure now schedules the next attempt (exponential backoff, or the
server's Retry-After), so failing rows leave the head of the queue, and a row that exhausts its attempts or fails
permanently becomes status "dead".

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


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "next_attempt_at" not in columns:
        op.add_column(TABLE, sa.Column("next_attempt_at", sa.DateTime(), nullable=True))
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if "ix_notification_queue_next_attempt_at" not in indexes:
        op.create_index("ix_notification_queue_next_attempt_at", TABLE, ["next_attempt_at"])


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if "ix_notification_queue_next_attempt_at" in indexes:
        op.drop_index("ix_notification_queue_next_attempt_at", table_name=TABLE)
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "next_attempt_at" in columns:
        op.drop_column(TABLE, "next_attempt_at")
