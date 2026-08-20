"""Add permission-aware article edit requests."""

from alembic import op
import sqlalchemy as sa

revision = "20260819_59"
down_revision = "20260816_58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "article_edit_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("company_domain", sa.String(length=255), nullable=False),
        sa.Column("article_id", sa.Uuid(), nullable=False),
        sa.Column("requested_by", sa.Uuid(), nullable=True),
        sa.Column("request_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="open"),
        sa.Column("resolved_by", sa.Uuid(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requested_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["resolved_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_article_edit_requests_company_domain", "article_edit_requests", ["company_domain"])
    op.create_index("ix_article_edit_requests_company_status", "article_edit_requests", ["company_domain", "status"])
    op.create_index("ix_article_edit_requests_article", "article_edit_requests", ["article_id"])
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE article_edit_requests ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE article_edit_requests FORCE ROW LEVEL SECURITY")
        op.execute("""
            CREATE POLICY tenant_article_edit_requests ON article_edit_requests
            USING (current_setting('app.global_admin', true) = 'true'
                   OR company_domain = current_setting('app.company_domain', true))
            WITH CHECK (current_setting('app.global_admin', true) = 'true'
                        OR company_domain = current_setting('app.company_domain', true))
        """)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP POLICY IF EXISTS tenant_article_edit_requests ON article_edit_requests")
    op.drop_index("ix_article_edit_requests_article", table_name="article_edit_requests")
    op.drop_index("ix_article_edit_requests_company_status", table_name="article_edit_requests")
    op.drop_index("ix_article_edit_requests_company_domain", table_name="article_edit_requests")
    op.drop_table("article_edit_requests")
