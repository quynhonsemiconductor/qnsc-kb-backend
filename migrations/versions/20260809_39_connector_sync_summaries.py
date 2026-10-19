"""store connector sync results for operator visibility

Revision ID: 20260809_39
Revises: 20260809_38
Create Date: 2026-08-09
"""

from alembic import op

revision = "20260809_39"
down_revision = "20260809_38"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE connector_jobs ADD COLUMN IF NOT EXISTS summary_json JSON")


def downgrade() -> None:
    op.execute("ALTER TABLE connector_jobs DROP COLUMN IF EXISTS summary_json")
