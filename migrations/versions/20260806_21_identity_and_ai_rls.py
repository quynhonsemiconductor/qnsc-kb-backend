"""extend tenant RLS to identity, AI, and user-owned records

Revision ID: 20260806_21
Revises: 20260806_20
Create Date: 2026-08-06
"""

import os

from alembic import op

revision = "20260806_21"
down_revision = "20260806_20"
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
    admin = "current_setting('app.global_admin', true) = 'true'"
    tenant = f"{admin} OR company_domain = current_setting('app.company_domain', true)"
    user_owned = (
        f"{admin} OR user_id::text = NULLIF(current_setting('app.user_id', true), '')"
    )
    user_tenant = f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    conversation_owned = f"{admin} OR EXISTS (SELECT 1 FROM ai_conversations c WHERE c.id = conversation_id AND c.user_id::text = NULLIF(current_setting('app.user_id', true), ''))"
    role_tenant = f"{admin} OR company_domain IS NULL OR company_domain = current_setting('app.company_domain', true)"
    role_permission_tenant = f"{admin} OR EXISTS (SELECT 1 FROM roles r WHERE r.id = role_id AND (r.company_domain IS NULL OR r.company_domain = current_setting('app.company_domain', true)))"
    user_group_tenant = f"{admin} OR (EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true)) AND EXISTS (SELECT 1 FROM access_groups g WHERE g.id = group_id AND g.company_domain = current_setting('app.company_domain', true)))"
    article_tenant = f"{admin} OR EXISTS (SELECT 1 FROM articles a WHERE a.id = article_id AND a.company_domain = current_setting('app.company_domain', true))"
    _policy("users", "tenant_users", tenant)
    _policy("access_groups", "tenant_access_groups", tenant)
    _policy("user_groups", "tenant_user_groups", user_group_tenant)
    _policy("roles", "tenant_roles", role_tenant)
    _policy("user_roles", "tenant_user_roles", user_tenant)
    _policy("role_permissions", "tenant_role_permissions", role_permission_tenant)
    _policy("refresh_sessions", "tenant_refresh_sessions", user_tenant)
    _policy(
        "notification_queue",
        "recipient_notifications",
        f"{admin} OR recipient_user_id::text = NULLIF(current_setting('app.user_id', true), '')",
    )
    _policy("ai_conversations", "owner_ai_conversations", user_owned)
    _policy("ai_messages", "owner_ai_messages", conversation_owned)
    _policy("ai_usage_logs", "owner_ai_usage_logs", user_owned)
    _policy("ai_feedback", "owner_ai_feedback", user_owned)
    _policy("search_logs", "owner_search_logs", user_owned)
    _policy("audit_logs", "tenant_audit_logs", user_tenant)
    _policy("comments", "tenant_comments", article_tenant)
    _policy("votes", "tenant_votes", article_tenant)
    _policy("bookmarks", "tenant_bookmarks", f"{user_owned} AND {article_tenant}")


def downgrade() -> None:
    for table, name in (
        ("bookmarks", "tenant_bookmarks"),
        ("votes", "tenant_votes"),
        ("comments", "tenant_comments"),
        ("audit_logs", "tenant_audit_logs"),
        ("search_logs", "owner_search_logs"),
        ("ai_feedback", "owner_ai_feedback"),
        ("ai_usage_logs", "owner_ai_usage_logs"),
        ("ai_messages", "owner_ai_messages"),
        ("ai_conversations", "owner_ai_conversations"),
        ("notification_queue", "recipient_notifications"),
        ("refresh_sessions", "tenant_refresh_sessions"),
        ("role_permissions", "tenant_role_permissions"),
        ("user_roles", "tenant_user_roles"),
        ("roles", "tenant_roles"),
        ("user_groups", "tenant_user_groups"),
        ("access_groups", "tenant_access_groups"),
        ("users", "tenant_users"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
