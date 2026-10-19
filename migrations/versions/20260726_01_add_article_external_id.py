"""add stable external article identifiers

Revision ID: 20260726_01
Revises:
"""

from alembic import op
import sqlalchemy as sa

revision = "20260726_01"
down_revision = "20260802_00"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS external_id VARCHAR(120)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_articles_external_id ON articles (external_id)"
    )


def downgrade() -> None:
    op.drop_index("ix_articles_external_id", table_name="articles")
    op.drop_column("articles", "external_id")
