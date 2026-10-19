"""retain rejected AI layouts for explicit reviewer decisions

Revision ID: 20260809_38
Revises: 20260808_37
Create Date: 2026-08-09
"""

from alembic import op

revision = "20260809_38"
down_revision = "20260808_37"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS restructure_candidate_md TEXT"
    )
    op.execute(
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS restructure_decision VARCHAR(30) NOT NULL DEFAULT 'not_reviewed'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE pending_drafts DROP COLUMN IF EXISTS restructure_decision")
    op.execute(
        "ALTER TABLE pending_drafts DROP COLUMN IF EXISTS restructure_candidate_md"
    )
