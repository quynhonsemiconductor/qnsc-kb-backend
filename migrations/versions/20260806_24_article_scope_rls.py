"""separate global article access from the database-wide admin bypass

Revision ID: 20260806_24
Revises: 20260806_23
Create Date: 2026-08-06
"""

import os

from alembic import op

revision = "20260806_24"
down_revision = "20260806_23"
branch_labels = None
depends_on = None


def _policy(table: str, name: str, expression: str) -> None:
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY {name} ON {table} USING ({expression})")


def upgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    admin = "current_setting('app.global_admin', true) = 'true'"
    article_global = "current_setting('app.global_article_access', true) = 'true'"
    tenant = f"{admin} OR {article_global} OR company_domain = current_setting('app.company_domain', true)"
    article_tenant = f"{admin} OR {article_global} OR EXISTS (SELECT 1 FROM articles a WHERE a.id = article_id AND a.company_domain = current_setting('app.company_domain', true))"
    access_group_tenant = f"{admin} OR {article_global} OR company_domain = current_setting('app.company_domain', true)"

    for table, name in (
        ("articles", "tenant_articles"),
        ("article_chunks", "tenant_article_chunks"),
        ("parent_chunks", "tenant_parent_chunks"),
        ("document_sources", "tenant_document_sources"),
        ("access_groups", "tenant_access_groups"),
        ("comments", "tenant_comments"),
        ("votes", "tenant_votes"),
        ("bookmarks", "tenant_bookmarks"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")

    _policy("articles", "tenant_articles", tenant)
    _policy("article_chunks", "tenant_article_chunks", article_tenant)
    _policy("parent_chunks", "tenant_parent_chunks", article_tenant)
    _policy("document_sources", "tenant_document_sources", article_tenant)
    _policy("access_groups", "tenant_access_groups", access_group_tenant)
    _policy("comments", "tenant_comments", article_tenant)
    _policy("votes", "tenant_votes", article_tenant)
    _policy(
        "bookmarks",
        "tenant_bookmarks",
        f"({admin} OR user_id::text = NULLIF(current_setting('app.user_id', true), '')) AND ({article_tenant})",
    )

    # These derived tables carry document content but previously had no
    # database-level tenant boundary of their own.
    _policy("article_versions", "tenant_article_versions", article_tenant)
    _policy("article_tags", "tenant_article_tags", article_tenant)
    _policy("article_access", "tenant_article_access", article_tenant)


def downgrade() -> None:
    for table, name in (
        ("article_access", "tenant_article_access"),
        ("article_tags", "tenant_article_tags"),
        ("article_versions", "tenant_article_versions"),
        ("bookmarks", "tenant_bookmarks"),
        ("votes", "tenant_votes"),
        ("comments", "tenant_comments"),
        ("access_groups", "tenant_access_groups"),
        ("document_sources", "tenant_document_sources"),
        ("parent_chunks", "tenant_parent_chunks"),
        ("article_chunks", "tenant_article_chunks"),
        ("articles", "tenant_articles"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
