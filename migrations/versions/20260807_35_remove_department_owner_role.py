"""Remove the legacy Department Owner role.

Department ownership is an explicit relationship in ``department_managers``.
It must not be represented as a permission role because ownership and access
permissions are independent concerns.
"""

from alembic import op

revision = "20260807_35"
down_revision = "20260807_34"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Preserve department_managers rows. They are now the authoritative
    # ownership relationship and remain editable from Users & Roles.
    op.execute("""
        DELETE FROM user_roles
        WHERE role_id IN (
            SELECT id FROM roles
            WHERE lower(name) = lower('Department Owner')
        )
    """)
    op.execute("""
        UPDATE users
        SET role = 'Staff'
        WHERE lower(role) = lower('Department Owner')
    """)
    # Ensure users left without a role still have the standard baseline role.
    # This is idempotent and also repairs any pre-existing orphaned users.
    op.execute("""
        INSERT INTO roles (id, name, company_domain, active, system)
        SELECT gen_random_uuid(), 'Staff', domains.company_domain, TRUE, TRUE
        FROM (
            SELECT DISTINCT company_domain
            FROM users
            WHERE company_domain IS NOT NULL
        ) domains
        WHERE NOT EXISTS (
            SELECT 1 FROM roles existing
            WHERE existing.name = 'Staff'
              AND existing.company_domain = domains.company_domain
        )
    """)
    op.execute("""
        INSERT INTO user_roles (user_id, role_id)
        SELECT users.id, staff.id
        FROM users
        JOIN roles staff
          ON staff.name = 'Staff'
         AND staff.company_domain = users.company_domain
        WHERE NOT EXISTS (
            SELECT 1 FROM user_roles assigned
            WHERE assigned.user_id = users.id
        )
        ON CONFLICT (user_id, role_id) DO NOTHING
    """)
    op.execute("""
        DELETE FROM roles
        WHERE lower(name) = lower('Department Owner')
    """)


def downgrade() -> None:
    # The legacy role was a misleading authorization abstraction. Ownership
    # data is intentionally not converted back into a role on downgrade.
    pass
