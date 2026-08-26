"""Backfill legacy access groups into the unified Department audience table."""

from alembic import op

revision = "20260816_57"
down_revision = "20260816_56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep the legacy tables for connector/FK compatibility during rollout,
    # but make Department(kind='access') the authoritative audience rows.
    op.execute("""
        INSERT INTO departments (id, created_at, updated_at, company_domain, name, kind, description, active)
        SELECT id, created_at, updated_at, company_domain, name, 'access', '', true
        FROM access_groups
        ON CONFLICT (company_domain, name) DO UPDATE SET kind = CASE
            WHEN departments.kind = 'org' THEN departments.kind ELSE 'access' END
    """)
    op.execute("""
        INSERT INTO user_departments (user_id, department_id)
        SELECT ug.user_id, d.id
        FROM user_groups ug
        JOIN access_groups ag ON ag.id = ug.group_id
        JOIN departments d ON d.company_domain = ag.company_domain AND d.name = ag.name
        ON CONFLICT DO NOTHING
    """)
    op.execute("""
        INSERT INTO article_departments (article_id, department_id)
        SELECT aa.article_id, d.id
        FROM article_access aa
        JOIN access_groups ag ON ag.id = aa.group_id
        JOIN departments d ON d.company_domain = ag.company_domain AND d.name = ag.name
        ON CONFLICT DO NOTHING
    """)
    # Explicit-user rows already carry the legacy users-only audience. Once
    # migrated, all articles use the normal department/audience branch.
    op.execute("UPDATE articles SET visibility = 'department' WHERE visibility = 'users'")


def downgrade() -> None:
    # Associations are intentionally retained: they are valid Department
    # memberships and removing them would destroy data if a rollout is backed
    # out after new users have been provisioned.
    pass
