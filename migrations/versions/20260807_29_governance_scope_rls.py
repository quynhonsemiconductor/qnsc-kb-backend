"""honor global governance-read permission under RLS

Revision ID: 20260807_29
Revises: 20260807_28
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_29"
down_revision = "20260807_28"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY {name} ON {table} USING ({expression})")


def _apply(global_governance: bool) -> None:
    admin = "current_setting('app.global_admin', true) = 'true'"
    governance = (
        "current_setting('app.global_governance_access', true) = 'true'"
        if global_governance
        else "false"
    )
    tenant = f"{admin} OR {governance} OR company_domain = current_setting('app.company_domain', true)"
    audit = f"{admin} OR {governance} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    for table, name, expression in (
        ("pending_drafts", "tenant_pending_drafts", tenant),
        ("gaps", "tenant_gaps", tenant),
        ("audit_logs", "tenant_audit_logs", audit),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        _policy(table, name, expression)


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        _apply(True)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        _apply(False)
