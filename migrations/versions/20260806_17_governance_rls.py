"""Extend production tenant RLS to governance and connector roots.

Revision ID: 20260806_17
Revises: 20260806_16
Create Date: 2026-08-06
"""

import os

from alembic import op

revision = "20260806_17"
down_revision = "20260806_16"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = current_schema() AND tablename = '{table}' AND policyname = '{name}') THEN CREATE POLICY {name} ON {table} USING ({expression}); END IF; END $$"
    )


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    tenant = "current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true)"
    connector_tenant = "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM connectors c WHERE c.id = connector_id AND c.company_domain = current_setting('app.company_domain', true))"
    _policy("pending_drafts", "tenant_pending_drafts", tenant)
    _policy("gaps", "tenant_gaps", tenant)
    _policy("departments", "tenant_departments", tenant)
    _policy("connectors", "tenant_connectors", tenant)
    _policy("connector_jobs", "tenant_connector_jobs", connector_tenant)


def downgrade() -> None:
    for table, name in (
        ("connector_jobs", "tenant_connector_jobs"),
        ("connectors", "tenant_connectors"),
        ("departments", "tenant_departments"),
        ("gaps", "tenant_gaps"),
        ("pending_drafts", "tenant_pending_drafts"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
