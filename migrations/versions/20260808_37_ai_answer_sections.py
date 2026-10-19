"""Persist grounded and extended AI answer sections."""

from alembic import op

revision = "20260808_37"
down_revision = "20260808_36"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE ai_messages ADD COLUMN IF NOT EXISTS grounded_content TEXT")
    op.execute("ALTER TABLE ai_messages ADD COLUMN IF NOT EXISTS extended_content TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE ai_messages DROP COLUMN IF EXISTS extended_content")
    op.execute("ALTER TABLE ai_messages DROP COLUMN IF EXISTS grounded_content")
