"""personal access tokens for MCP clients

An MCP client (Claude Desktop, Cursor, Claude Code) cannot run the browser sign-in, and a
short-lived JWT would force the user to paste a new one every hour. A personal access
token is a revocable, expiring credential that acts as its owner.

Only the SHA-256 of each token is stored, exactly as `refresh_sessions.token_hash` does it.
Guarded like the other additive migrations: the baseline builds tables from the model
metadata, so a fresh database already has this table.

Revision ID: 20261028_87
Revises: 20261027_86
Create Date: 2026-10-28

"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261028_87"
down_revision = "20261027_86"
branch_labels = None
depends_on = None

TABLE = "personal_access_tokens"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        op.create_table(
            TABLE,
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column(
                "user_id",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("name", sa.String(length=80), nullable=False),
            sa.Column("token_prefix", sa.String(length=16), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("scope", sa.String(length=10), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint(
                "scope IN ('read', 'write')", name="ck_personal_access_tokens_scope"
            ),
        )
        op.create_index(f"ix_{TABLE}_user_id", TABLE, ["user_id"], unique=False)
        # One UNIQUE index, which is what `unique=True, index=True` on the model means;
        # a separate constraint plus a plain index reads as drift to `alembic check`.
        op.create_index(f"ix_{TABLE}_token_hash", TABLE, ["token_hash"], unique=True)
        op.create_index(f"ix_{TABLE}_expires_at", TABLE, ["expires_at"], unique=False)

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
        tenant = (
            "current_setting('app.global_admin', true) = 'true' OR EXISTS "
            "(SELECT 1 FROM users u WHERE u.id = user_id AND "
            "u.company_domain = current_setting('app.company_domain', true))"
        )
        op.execute(f"DROP POLICY IF EXISTS tenant_personal_access_tokens ON {TABLE}")
        op.execute(
            f"CREATE POLICY tenant_personal_access_tokens ON {TABLE} "
            f"USING ({tenant}) WITH CHECK ({tenant})"
        )
    role = os.getenv("APP_DATABASE_ROLE")
    if role:
        safe_role = role.replace('"', '""')
        op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO "{safe_role}"')


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE in set(inspector.get_table_names()):
        op.execute(f"DROP POLICY IF EXISTS tenant_personal_access_tokens ON {TABLE}")
    op.execute(f"DROP TABLE IF EXISTS {TABLE}")
