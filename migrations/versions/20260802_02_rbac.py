"""add company-scoped roles and permission assignments

Revision ID: 20260802_02
Revises: 20260726_01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "20260802_02"
down_revision = "20260726_01"
branch_labels = None
depends_on = None

uuid = postgresql.UUID(as_uuid=True)


def timestamps():
    return [
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("permissions"):
        op.create_table(
            "permissions",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("key", sa.String(100), nullable=False, unique=True),
            sa.Column("name", sa.String(150), nullable=False),
            sa.Column("description", sa.String(500)),
            sa.Column("system", sa.Boolean(), nullable=False, server_default=sa.true()),
            *timestamps(),
        )
    if not inspector.has_table("roles"):
        op.create_table(
            "roles",
            sa.Column("id", uuid, primary_key=True),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("description", sa.String(500)),
            sa.Column("company_domain", sa.String(255)),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column(
                "system", sa.Boolean(), nullable=False, server_default=sa.false()
            ),
            *timestamps(),
            sa.UniqueConstraint("company_domain", "name", name="uq_roles_company_name"),
        )
    op.create_index(
        "ix_roles_company_domain", "roles", ["company_domain"], if_not_exists=True
    )
    if not inspector.has_table("role_permissions"):
        op.create_table(
            "role_permissions",
            sa.Column("id", uuid, primary_key=True),
            sa.Column(
                "role_id",
                uuid,
                sa.ForeignKey("roles.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "permission_id",
                uuid,
                sa.ForeignKey("permissions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("scope", sa.String(20), nullable=False, server_default="company"),
            *timestamps(),
            sa.UniqueConstraint("role_id", "permission_id", name="uq_role_permission"),
        )
    if not inspector.has_table("user_roles"):
        op.create_table(
            "user_roles",
            sa.Column(
                "user_id",
                uuid,
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "role_id",
                uuid,
                sa.ForeignKey("roles.id", ondelete="CASCADE"),
                primary_key=True,
            ),
        )


def downgrade() -> None:
    op.drop_table("user_roles")
    op.drop_table("role_permissions")
    op.drop_index("ix_roles_company_domain", table_name="roles")
    op.drop_table("roles")
    op.drop_table("permissions")
