"""Invalidate signed tokens after account-security changes.

Revision ID: 20260806_18
Revises: 20260806_17
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_18"
down_revision = "20260806_17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_version INTEGER NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS auth_version")
