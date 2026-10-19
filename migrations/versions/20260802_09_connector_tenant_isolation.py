"""Make connector synchronization tables tenant-safe when RLS is enabled."""

import os

from alembic import op

revision = "20260802_09"
down_revision = "20260802_08"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = current_schema() AND tablename = '{table}' AND policyname = '{name}') THEN CREATE POLICY {name} ON {table} USING ({expression}); END IF; END $$"
    )


def upgrade() -> None:
    # Connector names are tenant-local, not platform-global. The legacy
    # schema used connectors_name_key for a global unique constraint.
    op.execute("ALTER TABLE connectors DROP CONSTRAINT IF EXISTS connectors_name_key")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_connectors_company_name ON connectors (company_domain, name)"
    )

    # Development create_all databases do not set these policies. Production
    # compose enables RLS before running migrations.
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return

    connector_tenant = "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM connectors c WHERE c.id = connector_id AND (c.company_domain = current_setting('app.company_domain', true) OR current_setting('app.global_admin', true) = 'true'))"
    snapshot_tenant = "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM permission_snapshots p JOIN external_documents d ON d.id = p.external_document_id JOIN connectors c ON c.id = d.connector_id WHERE p.id = permission_snapshot_id AND c.company_domain = current_setting('app.company_domain', true))"
    document_tenant = "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM connectors c WHERE c.id = connector_id AND c.company_domain = current_setting('app.company_domain', true))"

    for table in (
        "source_scopes",
        "external_documents",
        "external_group_mappings",
        "sync_errors",
    ):
        _policy(
            table,
            f"tenant_{table}",
            connector_tenant if table != "external_documents" else document_tenant,
        )
    _policy("sync_cursors", "tenant_sync_cursors", connector_tenant)
    _policy(
        "document_versions",
        "tenant_document_versions",
        "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM external_documents d JOIN connectors c ON c.id = d.connector_id WHERE d.id = external_document_id AND c.company_domain = current_setting('app.company_domain', true))",
    )
    _policy(
        "permission_snapshots",
        "tenant_permission_snapshots",
        "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM external_documents d JOIN connectors c ON c.id = d.connector_id WHERE d.id = external_document_id AND c.company_domain = current_setting('app.company_domain', true))",
    )
    _policy(
        "external_acl_principals", "tenant_external_acl_principals", snapshot_tenant
    )
    _policy("webhook_subscriptions", "tenant_webhook_subscriptions", connector_tenant)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_connectors_company_name")
    op.execute(
        "ALTER TABLE connectors ADD CONSTRAINT connectors_name_key UNIQUE (name)"
    )
    for table, name in (
        ("webhook_subscriptions", "tenant_webhook_subscriptions"),
        ("external_acl_principals", "tenant_external_acl_principals"),
        ("permission_snapshots", "tenant_permission_snapshots"),
        ("document_versions", "tenant_document_versions"),
        ("sync_cursors", "tenant_sync_cursors"),
        ("sync_errors", "tenant_sync_errors"),
        ("external_group_mappings", "tenant_external_group_mappings"),
        ("external_documents", "tenant_external_documents"),
        ("source_scopes", "tenant_source_scopes"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
