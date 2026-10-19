"""Add tenant-managed departments and preserve existing department values."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "20260806_13"
down_revision = "20260806_12"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("departments"):
        op.create_table(
            "departments",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("company_domain", sa.String(255), nullable=False),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.UniqueConstraint(
                "company_domain", "name", name="uq_departments_company_name"
            ),
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_departments_company_domain ON departments (company_domain)"
    )
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute(
        "INSERT INTO departments (id, company_domain, name, active, created_at, updated_at) SELECT gen_random_uuid(), company_domain, dept, TRUE, now(), now() FROM (SELECT company_domain, dept FROM users WHERE dept IS NOT NULL AND dept <> '' UNION SELECT company_domain, dept FROM articles WHERE dept IS NOT NULL AND dept <> '' UNION SELECT company_domain, dept FROM pending_drafts WHERE dept IS NOT NULL AND dept <> '') values_to_keep ON CONFLICT (company_domain, name) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("departments")
