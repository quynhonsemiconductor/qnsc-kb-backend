"""add visible bulk indexing progress and retry state

Revision ID: 20260810_46
Revises: 20260810_45
"""

import os
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_46"
down_revision = "20260810_45"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("index_reprocess_jobs"):
        op.create_table(
            "index_reprocess_jobs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("company_domain", sa.String(255), nullable=False),
            sa.Column(
                "requested_by",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("status", sa.String(30), nullable=False, server_default="queued"),
            sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("completed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("target_article_ids", sa.JSON(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
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
        )
        op.create_index(
            "ix_index_reprocess_jobs_company_domain",
            "index_reprocess_jobs",
            ["company_domain"],
        )
        op.create_index(
            "ix_index_reprocess_jobs_status", "index_reprocess_jobs", ["status"]
        )
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE index_reprocess_jobs ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE index_reprocess_jobs FORCE ROW LEVEL SECURITY")
        op.execute(
            "DROP POLICY IF EXISTS tenant_index_reprocess_jobs ON index_reprocess_jobs"
        )
        op.execute(
            "CREATE POLICY tenant_index_reprocess_jobs ON index_reprocess_jobs USING (current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true)) WITH CHECK (current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true))"
        )


def downgrade() -> None:
    op.drop_index("ix_index_reprocess_jobs_status", table_name="index_reprocess_jobs")
    op.drop_index(
        "ix_index_reprocess_jobs_company_domain", table_name="index_reprocess_jobs"
    )
    op.drop_table("index_reprocess_jobs")
