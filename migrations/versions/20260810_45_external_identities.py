"""link Microsoft Entra subjects to internal users

Revision ID: 20260810_45
Revises: 20260810_44
"""

import os
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_45"
down_revision = "20260810_44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("external_identities"):
        op.create_table(
            "external_identities",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "user_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("provider", sa.String(50), nullable=False),
            sa.Column("subject", sa.String(255), nullable=False),
            sa.Column("email", sa.String(255), nullable=True),
            sa.Column("tenant_id", sa.String(255), nullable=True),
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
            sa.UniqueConstraint(
                "provider", "subject", name="uq_external_identity_provider_subject"
            ),
        )
        op.create_index(
            "ix_external_identities_user_id", "external_identities", ["user_id"]
        )
        op.create_index(
            "ix_external_identities_email", "external_identities", ["email"]
        )
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE external_identities ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE external_identities FORCE ROW LEVEL SECURITY")
        op.execute(
            "DROP POLICY IF EXISTS tenant_external_identities ON external_identities"
        )
        op.execute(
            "CREATE POLICY tenant_external_identities ON external_identities USING (current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))) WITH CHECK (current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true)))"
        )


def downgrade() -> None:
    op.drop_index("ix_external_identities_email", table_name="external_identities")
    op.drop_index("ix_external_identities_user_id", table_name="external_identities")
    op.drop_table("external_identities")
