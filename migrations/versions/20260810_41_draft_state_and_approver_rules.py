"""add explicit draft transitions and department approver rules

Revision ID: 20260810_41
Revises: 20260810_40
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_41"
down_revision = "20260810_40"
branch_labels = None
depends_on = None


def _rls_policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
    op.execute(
        f"CREATE POLICY {name} ON {table} USING ({expression}) WITH CHECK ({expression})"
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("draft_transitions"):
        op.create_table(
            "draft_transitions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "draft_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("pending_drafts.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("from_status", sa.String(30), nullable=True),
            sa.Column("to_status", sa.String(30), nullable=False),
            sa.Column(
                "actor_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column(
                "outcome", sa.String(30), nullable=False, server_default="applied"
            ),
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
            "ix_draft_transitions_draft_id", "draft_transitions", ["draft_id"]
        )
        op.create_index(
            "ix_draft_transitions_actor_id", "draft_transitions", ["actor_id"]
        )

    if not inspector.has_table("approver_rules"):
        op.create_table(
            "approver_rules",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("company_domain", sa.String(255), nullable=False),
            sa.Column("dept", sa.String(100), nullable=False),
            sa.Column(
                "approver_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column(
                "created_by",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
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
                "company_domain", "dept", name="uq_approver_rules_company_dept"
            ),
        )
        op.create_index(
            "ix_approver_rules_company_domain", "approver_rules", ["company_domain"]
        )
        op.create_index("ix_approver_rules_dept", "approver_rules", ["dept"])
        op.create_index(
            "ix_approver_rules_approver_id", "approver_rules", ["approver_id"]
        )

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        admin = "current_setting('app.global_admin', true) = 'true'"
        company = "current_setting('app.company_domain', true)"
        _rls_policy(
            "draft_transitions",
            "tenant_draft_transitions",
            f"{admin} OR EXISTS (SELECT 1 FROM pending_drafts d WHERE d.id = draft_id AND d.company_domain = {company})",
        )
        _rls_policy(
            "approver_rules",
            "tenant_approver_rules",
            f"{admin} OR company_domain = {company}",
        )


def downgrade() -> None:
    op.drop_index("ix_approver_rules_approver_id", table_name="approver_rules")
    op.drop_index("ix_approver_rules_dept", table_name="approver_rules")
    op.drop_index("ix_approver_rules_company_domain", table_name="approver_rules")
    op.drop_table("approver_rules")
    op.drop_index("ix_draft_transitions_actor_id", table_name="draft_transitions")
    op.drop_index("ix_draft_transitions_draft_id", table_name="draft_transitions")
    op.drop_table("draft_transitions")
