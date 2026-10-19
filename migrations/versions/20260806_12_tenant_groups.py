"""Scope access groups to a company domain."""

from alembic import op

revision = "20260806_12"
down_revision = "20260806_11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE access_groups ADD COLUMN IF NOT EXISTS company_domain VARCHAR(255) NOT NULL DEFAULT 'local'"
    )
    op.execute(
        "ALTER TABLE access_groups DROP CONSTRAINT IF EXISTS access_groups_name_key"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_access_groups_company_domain ON access_groups (company_domain)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_access_groups_company_name ON access_groups (company_domain, name)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_access_groups_company_name")
    op.execute("DROP INDEX IF EXISTS ix_access_groups_company_domain")
    op.execute("ALTER TABLE access_groups DROP COLUMN IF EXISTS company_domain")
