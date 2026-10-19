"""Tenant-scope search gaps.

Revision ID: 20260806_15
Revises: 20260806_14
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_15"
down_revision = "20260806_14"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE gaps ADD COLUMN IF NOT EXISTS company_domain VARCHAR(255) NOT NULL DEFAULT 'local'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_gaps_company_domain ON gaps (company_domain)"
    )
    op.execute("ALTER TABLE gaps DROP CONSTRAINT IF EXISTS gaps_query_key")
    op.execute("DROP INDEX IF EXISTS ix_gaps_query")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_gaps_company_query ON gaps (company_domain, query)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_gaps_company_query")
    op.execute("ALTER TABLE gaps DROP COLUMN IF EXISTS company_domain")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS gaps_query_key ON gaps (query)")
