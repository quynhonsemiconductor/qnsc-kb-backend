"""Add audience kind and per-organisational-audience contact email."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
import os

revision = "20260816_55"
down_revision = "20260816_54"
branch_labels = None
depends_on = None


def _missing(table: str, column: str) -> bool:
    """See 20260814_52: the baseline builds the schema from the live models, so on a
    database created from empty these columns already exist and only an older
    database actually needs the ALTER."""
    return column not in {
        item["name"] for item in inspect(op.get_bind()).get_columns(table)
    }


def upgrade() -> None:
    if _missing("departments", "kind"):
        op.add_column("departments", sa.Column("kind", sa.String(length=10), nullable=False, server_default="org"))
        op.alter_column("departments", "kind", server_default=None)
    if _missing("departments", "contact_email"):
        op.add_column("departments", sa.Column("contact_email", sa.String(length=255), nullable=True))
    existing_checks = {
        item["name"] for item in inspect(op.get_bind()).get_check_constraints("departments")
    }
    if "ck_departments_kind" not in existing_checks:
        op.create_check_constraint("ck_departments_kind", "departments", "kind IN ('org', 'access')")
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE invitations ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE invitations FORCE ROW LEVEL SECURITY")
        # Postgres has no CREATE POLICY IF NOT EXISTS, and re-running this migration on
        # a database that already carries the policy must not fail.
        op.execute("DROP POLICY IF EXISTS tenant_invitations ON invitations")
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
