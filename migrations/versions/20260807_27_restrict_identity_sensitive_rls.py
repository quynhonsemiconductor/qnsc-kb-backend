"""keep cross-company identity scope away from sensitive session and audit data

Revision ID: 20260807_27
Revises: 20260807_26
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_27"
down_revision = "20260807_26"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY {name} ON {table} USING ({expression})")


def _tenant_user_expression() -> str:
    admin = "current_setting('app.global_admin', true) = 'true'"
    return f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    expression = _tenant_user_expression()
    for table, name in (
        ("refresh_sessions", "tenant_refresh_sessions"),
        ("audit_logs", "tenant_audit_logs"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        _policy(table, name, expression)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    admin = "current_setting('app.global_admin', true) = 'true'"
    identity = "current_setting('app.global_identity_access', true) = 'true'"
    expression = f"{admin} OR {identity} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    for table, name in (
        ("audit_logs", "tenant_audit_logs"),
        ("refresh_sessions", "tenant_refresh_sessions"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        _policy(table, name, expression)
