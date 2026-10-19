"""Make approval notifications user-addressable.

Revision ID: 20260806_19
Revises: 20260806_18
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_19"
down_revision = "20260806_18"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE notification_queue ADD COLUMN IF NOT EXISTS recipient_user_id UUID REFERENCES users(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE notification_queue ADD COLUMN IF NOT EXISTS read_at TIMESTAMP"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notification_queue_recipient_user_id ON notification_queue (recipient_user_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_notification_queue_recipient_user_id")
    op.execute("ALTER TABLE notification_queue DROP COLUMN IF EXISTS read_at")
    op.execute("ALTER TABLE notification_queue DROP COLUMN IF EXISTS recipient_user_id")
