"""Add audience kind and per-organisational-audience contact email."""

from alembic import op
import sqlalchemy as sa
import os

revision = "20260816_55"
down_revision = "20260816_54"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("departments", sa.Column("kind", sa.String(length=10), nullable=False, server_default="org"))
    op.add_column("departments", sa.Column("contact_email", sa.String(length=255), nullable=True))
    op.alter_column("departments", "kind", server_default=None)
    op.create_check_constraint("ck_departments_kind", "departments", "kind IN ('org', 'access')")
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE invitations ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE invitations FORCE ROW LEVEL SECURITY")
        op.execute(
            "CREATE POLICY tenant_invitations ON invitations USING "
            "(current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true)) "
            "WITH CHECK (current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true))"
        )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_invitations ON invitations")
    op.drop_constraint("ck_departments_kind", "departments", type_="check")
    op.drop_column("departments", "contact_email")
    op.drop_column("departments", "kind")
