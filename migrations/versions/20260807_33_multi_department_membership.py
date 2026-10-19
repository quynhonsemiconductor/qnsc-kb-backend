"""Add multi-department membership for users and articles."""

from alembic import op
import os

revision = "20260807_33"
down_revision = "20260807_32"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_departments (
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
            PRIMARY KEY (user_id, department_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_user_departments_department_id
        ON user_departments (department_id)
    """)
    op.execute("""
        INSERT INTO user_departments (user_id, department_id)
        SELECT u.id, d.id
        FROM users u
        JOIN departments d
          ON d.company_domain = u.company_domain
         AND lower(d.name) = lower(trim(u.dept))
         AND d.active = TRUE
        WHERE u.dept IS NOT NULL AND btrim(u.dept) <> ''
        ON CONFLICT DO NOTHING
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS article_departments (
            article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
            PRIMARY KEY (article_id, department_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_article_departments_department_id
        ON article_departments (department_id)
    """)
    op.execute("""
        INSERT INTO article_departments (article_id, department_id)
        SELECT a.id, d.id
        FROM articles a
        JOIN departments d
          ON d.company_domain = a.company_domain
         AND lower(d.name) = lower(trim(a.dept))
         AND d.active = TRUE
        WHERE a.dept IS NOT NULL AND btrim(a.dept) <> ''
        ON CONFLICT DO NOTHING
    """)

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE user_departments ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_departments FORCE ROW LEVEL SECURITY")
        op.execute("DROP POLICY IF EXISTS tenant_user_departments ON user_departments")
        op.execute("""
            CREATE POLICY tenant_user_departments ON user_departments USING (
                current_setting('app.global_admin', true) = 'true'
                OR current_setting('app.global_identity_access', true) = 'true'
                OR EXISTS (
                    SELECT 1 FROM users u
                    WHERE u.id = user_id
                      AND u.company_domain = current_setting('app.company_domain', true)
                )
            )
        """)
        op.execute("ALTER TABLE article_departments ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE article_departments FORCE ROW LEVEL SECURITY")
        op.execute(
            "DROP POLICY IF EXISTS tenant_article_departments ON article_departments"
        )
        op.execute("""
            CREATE POLICY tenant_article_departments ON article_departments USING (
                current_setting('app.global_admin', true) = 'true'
                OR current_setting('app.global_article_access', true) = 'true'
                OR EXISTS (
                    SELECT 1 FROM articles a
                    WHERE a.id = article_id
                      AND a.company_domain = current_setting('app.company_domain', true)
                )
            )
        """)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(
            "DROP POLICY IF EXISTS tenant_article_departments ON article_departments"
        )
        op.execute("DROP POLICY IF EXISTS tenant_user_departments ON user_departments")
    op.drop_table("article_departments")
    op.drop_table("user_departments")
