"""Bind each user to the first Cloudflare Access `sub` seen for their account

The MCP endpoint maps a verified Access email to an active user (P0 gate d showed the
Entra `oid` claim does not reach the origin). An email can be reassigned, so the account
also records the first Access `sub` it saw: a later token with the same email and a
different subject is refused instead of inheriting the account. Cleared by an
administrator to rebind.

A unique index: one Access identity belongs to one account, and the inverse lookup must
not be ambiguous. NULL rows are unconstrained in PostgreSQL, so existing users are safe
until their first Access login.

Revision ID: 20261106_94
Revises: 20261104_93
Create Date: 2026-11-06
"""

from alembic import op
import sqlalchemy as sa

revision = "20261106_94"
down_revision = "20261104_93"
branch_labels = None
depends_on = None


def _columns() -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns("users")}


def _indexes() -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("users")}


def upgrade() -> None:
    # The 20260802_00 baseline builds the schema with Base.metadata.create_all against the
    # LIVE models, so a freshly migrated database already has this column and its unique
    # index. Guard both, the same way migrations 67/68/70 do.
    if "access_subject" not in _columns():
        op.add_column(
            "users", sa.Column("access_subject", sa.String(255), nullable=True)
        )
    if "ix_users_access_subject" not in _indexes():
        op.create_index(
            "ix_users_access_subject", "users", ["access_subject"], unique=True
        )


def downgrade() -> None:
    if "ix_users_access_subject" in _indexes():
        op.drop_index("ix_users_access_subject", table_name="users")
    if "access_subject" in _columns():
        op.drop_column("users", "access_subject")
