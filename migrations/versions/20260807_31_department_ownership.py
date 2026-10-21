"""add explicit department ownership assignments

Revision ID: 20260807_31
Revises: 20260807_30
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_31"
down_revision = "20260807_30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS department_managers (
            id UUID PRIMARY KEY,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
            department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            CONSTRAINT uq_department_manager_assignment UNIQUE (department_id, user_id)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_department_managers_department_id ON department_managers (department_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_department_managers_user_id ON department_managers (user_id)"
    )

    # Preserve the current behavior for existing Department Owners while
    # converting it to explicit data. Future assignments must be deliberate.
    op.execute("""
        INSERT INTO department_managers (id, created_at, updated_at, department_id, user_id, active)
        SELECT gen_random_uuid(), now(), now(), d.id, u.id, TRUE
        FROM users u
        JOIN departments d
          ON d.company_domain = u.company_domain
         AND lower(d.name) = lower(trim(u.dept))
         AND d.active = TRUE
        WHERE u.dept IS NOT NULL
          AND trim(u.dept) <> ''
          AND (u.role = 'Department Owner' OR EXISTS (
              SELECT 1 FROM user_roles ur
              JOIN roles r ON r.id = ur.role_id
              WHERE ur.user_id = u.id AND r.name = 'Department Owner'
          ))
        ON CONFLICT (department_id, user_id) DO NOTHING
    """)

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute("ALTER TABLE department_managers ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE department_managers FORCE ROW LEVEL SECURITY")
        op.execute("""
            CREATE POLICY tenant_department_managers ON department_managers
            USING (
                current_setting('app.global_admin', true) = 'true'
                OR current_setting('app.global_identity_access', true) = 'true'
                OR EXISTS (
                    SELECT 1 FROM departments d
                    WHERE d.id = department_id
                      AND d.company_domain = current_setting('app.company_domain', true)
                )
            )
        """)


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(
            "DROP POLICY IF EXISTS tenant_department_managers ON department_managers"
        )
    op.execute("DROP TABLE IF EXISTS department_managers")
