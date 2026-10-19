"""Add tenant-scoped, auditable pending-draft approval assignments."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260806_11"
down_revision = "20260803_10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The baseline migration creates current metadata for a new database;
    # use idempotent DDL so this revision also upgrades that installation path.
    for statement in (
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS company_domain VARCHAR(255) DEFAULT 'local'",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS dept VARCHAR(100)",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS assigned_approver_id UUID REFERENCES users(id) ON DELETE SET NULL",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS assigned_by UUID REFERENCES users(id) ON DELETE SET NULL",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS assigned_at TIMESTAMP",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS reviewed_by UUID REFERENCES users(id) ON DELETE SET NULL",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP",
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS review_note TEXT",
    ):
        op.execute(statement)
    op.execute(
        "UPDATE pending_drafts p SET company_domain = COALESCE((SELECT u.company_domain FROM users u WHERE u.id = p.created_by), 'local') WHERE p.company_domain IS NULL OR p.company_domain = ''"
    )
    op.execute("ALTER TABLE pending_drafts ALTER COLUMN company_domain SET NOT NULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_pending_drafts_company_domain ON pending_drafts (company_domain)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_pending_drafts_assigned_approver_id ON pending_drafts (assigned_approver_id)"
    )


def downgrade() -> None:
    op.drop_index("ix_pending_drafts_assigned_approver_id", table_name="pending_drafts")
    op.drop_index("ix_pending_drafts_company_domain", table_name="pending_drafts")
    op.drop_column("pending_drafts", "review_note")
    op.drop_column("pending_drafts", "reviewed_at")
    op.drop_column("pending_drafts", "reviewed_by")
    op.drop_column("pending_drafts", "assigned_at")
    op.drop_column("pending_drafts", "assigned_by")
    op.drop_column("pending_drafts", "assigned_approver_id")
    op.drop_column("pending_drafts", "dept")
    op.drop_column("pending_drafts", "company_domain")
