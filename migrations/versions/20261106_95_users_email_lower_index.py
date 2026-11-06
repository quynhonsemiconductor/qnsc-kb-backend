"""Index lower(email) for the Access identity lookup

The MCP Access path normalises the IdP email and, only when the exact address misses,
falls back to `lower(email)`. Without an expression index that fallback is a sequential
scan on `users`, which `get_current_user` also reads on every authenticated request.
`users` is small today, but the lookup should not depend on that.

Guarded: the 20260802_00 baseline builds the schema with `Base.metadata.create_all` against
the LIVE models, so a freshly migrated database already has the index.

Revision ID: 20261106_95
Revises: 20261106_94
Create Date: 2026-11-06
"""

from alembic import op
import sqlalchemy as sa

revision = "20261106_95"
down_revision = "20261106_94"
branch_labels = None
depends_on = None


def _indexes() -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("users")}


def upgrade() -> None:
    if "ix_users_email_lower" not in _indexes():
        op.create_index(
            "ix_users_email_lower", "users", [sa.text("lower(email)")], unique=False
        )


def downgrade() -> None:
    if "ix_users_email_lower" in _indexes():
        op.drop_index("ix_users_email_lower", table_name="users")
