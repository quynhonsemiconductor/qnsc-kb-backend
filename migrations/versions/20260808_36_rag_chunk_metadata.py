"""Add structure-aware RAG chunk metadata and versioning."""

from alembic import op

revision = "20260808_36"
down_revision = "20260807_35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE parent_chunks "
        "ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(40) NOT NULL DEFAULT 'section'"
    )
    op.execute(
        "ALTER TABLE parent_chunks " "ADD COLUMN IF NOT EXISTS heading VARCHAR(255)"
    )
    op.execute(
        "ALTER TABLE article_chunks "
        "ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(40) NOT NULL DEFAULT 'text'"
    )
    op.execute(
        "ALTER TABLE article_chunks " "ADD COLUMN IF NOT EXISTS heading VARCHAR(255)"
    )
    op.execute(
        "ALTER TABLE article_chunks "
        "ADD COLUMN IF NOT EXISTS chunking_version VARCHAR(80) "
        "NOT NULL DEFAULT 'v1-fixed-character'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE article_chunks DROP COLUMN IF EXISTS chunking_version")
    op.execute("ALTER TABLE article_chunks DROP COLUMN IF EXISTS heading")
    op.execute("ALTER TABLE article_chunks DROP COLUMN IF EXISTS chunk_type")
    op.execute("ALTER TABLE parent_chunks DROP COLUMN IF EXISTS heading")
    op.execute("ALTER TABLE parent_chunks DROP COLUMN IF EXISTS chunk_type")
