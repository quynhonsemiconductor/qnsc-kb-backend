"""Persist AI action metadata for refresh-safe confirmations."""

from alembic import op
import sqlalchemy as sa

revision = "20260819_60"
down_revision = "20260819_59"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_messages", sa.Column("action_data", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_messages", "action_data")
