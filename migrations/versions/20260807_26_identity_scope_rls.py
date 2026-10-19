"""allow explicitly global identity-management roles through identity RLS

Revision ID: 20260807_26
Revises: 20260807_25
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_26"
down_revision = "20260807_25"
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
    identity = "current_setting('app.global_identity_access', true) = 'true'"
    article_global = "current_setting('app.global_article_access', true) = 'true'"
    user_tenant = f"{admin} OR {identity} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    session_tenant = f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    group_tenant = f"{admin} OR {identity} OR {article_global} OR company_domain = current_setting('app.company_domain', true)"
    role_tenant = f"{admin} OR {identity} OR company_domain IS NULL OR company_domain = current_setting('app.company_domain', true)"
    role_permission_tenant = f"{admin} OR {identity} OR EXISTS (SELECT 1 FROM roles r WHERE r.id = role_id AND (r.company_domain IS NULL OR r.company_domain = current_setting('app.company_domain', true)))"
    user_group_tenant = f"{admin} OR {identity} OR (EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true)) AND EXISTS (SELECT 1 FROM access_groups g WHERE g.id = group_id AND g.company_domain = current_setting('app.company_domain', true)))"
    tenant_users = f"{admin} OR {identity} OR company_domain = current_setting('app.company_domain', true)"
    audit_tenant = f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"

    for table, name in (
        ("users", "tenant_users"),
        ("access_groups", "tenant_access_groups"),
        ("user_groups", "tenant_user_groups"),
        ("roles", "tenant_roles"),
        ("user_roles", "tenant_user_roles"),
        ("role_permissions", "tenant_role_permissions"),
        ("refresh_sessions", "tenant_refresh_sessions"),
        ("audit_logs", "tenant_audit_logs"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")

    _policy("users", "tenant_users", tenant_users)
    _policy("access_groups", "tenant_access_groups", group_tenant)
    _policy("user_groups", "tenant_user_groups", user_group_tenant)
    _policy("roles", "tenant_roles", role_tenant)
    _policy("user_roles", "tenant_user_roles", user_tenant)
    _policy("role_permissions", "tenant_role_permissions", role_permission_tenant)
    # A user/role manager does not need authentication-token hashes or the
    # complete audit trail for another company.  Keep these on the caller's
    # tenant boundary; only the dedicated global Admin bypass can cross it.
    _policy("refresh_sessions", "tenant_refresh_sessions", session_tenant)
    _policy("audit_logs", "tenant_audit_logs", audit_tenant)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    for table, name in (
        ("audit_logs", "tenant_audit_logs"),
        ("refresh_sessions", "tenant_refresh_sessions"),
        ("role_permissions", "tenant_role_permissions"),
        ("user_roles", "tenant_user_roles"),
        ("roles", "tenant_roles"),
        ("user_groups", "tenant_user_groups"),
        ("access_groups", "tenant_access_groups"),
        ("users", "tenant_users"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")

    # Restore the policies in effect at revision 20260807_25.  Leaving forced
    # RLS enabled with no policies would make a downgrade silently deny every
    # application query.
    admin = "current_setting('app.global_admin', true) = 'true'"
    article_global = "current_setting('app.global_article_access', true) = 'true'"
    tenant = f"{admin} OR company_domain = current_setting('app.company_domain', true)"
    user_tenant = f"{admin} OR EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true))"
    access_group_tenant = f"{admin} OR {article_global} OR company_domain = current_setting('app.company_domain', true)"
    role_tenant = f"{admin} OR company_domain IS NULL OR company_domain = current_setting('app.company_domain', true)"
    role_permission_tenant = f"{admin} OR EXISTS (SELECT 1 FROM roles r WHERE r.id = role_id AND (r.company_domain IS NULL OR r.company_domain = current_setting('app.company_domain', true)))"
    user_group_tenant = f"{admin} OR (EXISTS (SELECT 1 FROM users u WHERE u.id = user_id AND u.company_domain = current_setting('app.company_domain', true)) AND EXISTS (SELECT 1 FROM access_groups g WHERE g.id = group_id AND g.company_domain = current_setting('app.company_domain', true)))"

    _policy("users", "tenant_users", tenant)
    _policy("access_groups", "tenant_access_groups", access_group_tenant)
    _policy("user_groups", "tenant_user_groups", user_group_tenant)
    _policy("roles", "tenant_roles", role_tenant)
    _policy("user_roles", "tenant_user_roles", user_tenant)
    _policy("role_permissions", "tenant_role_permissions", role_permission_tenant)
    _policy("refresh_sessions", "tenant_refresh_sessions", user_tenant)
    _policy("audit_logs", "tenant_audit_logs", user_tenant)
