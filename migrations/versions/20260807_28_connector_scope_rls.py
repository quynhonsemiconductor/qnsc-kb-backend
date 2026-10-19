"""honor global connector-management permission under RLS

Revision ID: 20260807_28
Revises: 20260807_27
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_28"
down_revision = "20260807_27"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY {name} ON {table} USING ({expression})")


def _replace(table: str, name: str, expression: str) -> None:
    op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
    _policy(table, name, expression)


def _expressions(include_global_connector: bool) -> tuple[str, str, str, str]:
    admin = "current_setting('app.global_admin', true) = 'true'"
    connector_global = (
        "current_setting('app.global_connector_access', true) = 'true'"
        if include_global_connector
        else "false"
    )
    root = f"{admin} OR {connector_global} OR company_domain = current_setting('app.company_domain', true)"
    child = f"{admin} OR {connector_global} OR EXISTS (SELECT 1 FROM connectors c WHERE c.id = connector_id AND c.company_domain = current_setting('app.company_domain', true))"
    document_child = f"{admin} OR {connector_global} OR EXISTS (SELECT 1 FROM external_documents d JOIN connectors c ON c.id = d.connector_id WHERE d.id = external_document_id AND c.company_domain = current_setting('app.company_domain', true))"
    snapshot_child = f"{admin} OR {connector_global} OR EXISTS (SELECT 1 FROM permission_snapshots p JOIN external_documents d ON d.id = p.external_document_id JOIN connectors c ON c.id = d.connector_id WHERE p.id = permission_snapshot_id AND c.company_domain = current_setting('app.company_domain', true))"
    return root, child, document_child, snapshot_child


def _apply(include_global_connector: bool) -> None:
    root, child, document_child, snapshot_child = _expressions(include_global_connector)
    _replace("connectors", "tenant_connectors", root)
    _replace("connector_jobs", "tenant_connector_jobs", child)
    for table in (
        "source_scopes",
        "sync_cursors",
        "external_group_mappings",
        "sync_errors",
        "webhook_subscriptions",
    ):
        _replace(table, f"tenant_{table}", child)
    _replace("external_documents", "tenant_external_documents", child)
    for table in ("document_versions", "permission_snapshots"):
        _replace(table, f"tenant_{table}", document_child)
    _replace(
        "external_acl_principals", "tenant_external_acl_principals", snapshot_child
    )


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        _apply(True)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        _apply(False)
