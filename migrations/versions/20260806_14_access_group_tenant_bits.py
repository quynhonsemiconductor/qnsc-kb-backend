"""Make access-group bit positions tenant-local.

The bitmap is evaluated only after the article tenant filter, so positions may
be reused safely by different companies.  A global unique position incorrectly
limited the entire installation to fewer than 62 groups.
"""

from alembic import op

revision = "20260806_14"
down_revision = "20260806_13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE access_groups DROP CONSTRAINT IF EXISTS access_groups_bitmask_position_key"
    )
    op.execute("DROP INDEX IF EXISTS ix_access_groups_bitmask_position")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_access_groups_company_bit_position "
        "ON access_groups (company_domain, bitmask_position)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_access_groups_company_bit_position")
    op.execute(
        "ALTER TABLE access_groups ADD CONSTRAINT access_groups_bitmask_position_key UNIQUE (bitmask_position)"
    )
