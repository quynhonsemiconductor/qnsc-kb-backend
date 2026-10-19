"""Enforce one active Department Owner per department."""

from alembic import op

revision = "20260807_34"
down_revision = "20260807_33"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DELETE FROM department_managers older
        USING department_managers newer
        WHERE older.department_id = newer.department_id
          AND older.active = TRUE
          AND newer.active = TRUE
          AND older.id <> newer.id
          AND (older.created_at, older.id) > (newer.created_at, newer.id)
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_department_managers_one_active_owner
        ON department_managers (department_id)
        WHERE active = TRUE
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_department_managers_one_active_owner")
