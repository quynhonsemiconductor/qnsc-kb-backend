"""enable optional PostgreSQL row-level tenant isolation"""

import os
from alembic import op

revision = "20260802_05"
down_revision = "20260802_04"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = current_schema() AND tablename = '{table}' AND policyname = '{name}') THEN CREATE POLICY {name} ON {table} USING ({expression}); END IF; END $$"
    )


def upgrade() -> None:
    # Local create_all development databases remain unchanged. Production
    # Compose sets ENABLE_RLS=true before running this migration.
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    tenant = "current_setting('app.global_admin', true) = 'true' OR company_domain = current_setting('app.company_domain', true)"
    _policy("articles", "tenant_articles", tenant)
    article_tenant = "current_setting('app.global_admin', true) = 'true' OR EXISTS (SELECT 1 FROM articles a WHERE a.id = article_id AND a.company_domain = current_setting('app.company_domain', true))"
    _policy("article_chunks", "tenant_article_chunks", article_tenant)
    _policy("parent_chunks", "tenant_parent_chunks", article_tenant)
    _policy("document_sources", "tenant_document_sources", article_tenant)
    role = os.getenv("APP_DATABASE_ROLE")
    if role:
        safe_role = role.replace('"', '""')
        op.execute(f'GRANT USAGE ON SCHEMA public TO "{safe_role}"')
        op.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{safe_role}"'
        )
        op.execute(
            f'GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO "{safe_role}"'
        )
        op.execute(
            f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{safe_role}"'
        )
        op.execute(
            f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO "{safe_role}"'
        )


def downgrade() -> None:
    for table, name in (
        ("document_sources", "tenant_document_sources"),
        ("parent_chunks", "tenant_parent_chunks"),
        ("article_chunks", "tenant_article_chunks"),
        ("articles", "tenant_articles"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
