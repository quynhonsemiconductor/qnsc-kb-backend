"""Prevent duplicate history version numbers during concurrent edits.

Revision ID: 20260806_16
Revises: 20260806_15
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_16"
down_revision = "20260806_15"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_article_versions_article_version ON article_versions (article_id, version)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_article_versions_article_version")
