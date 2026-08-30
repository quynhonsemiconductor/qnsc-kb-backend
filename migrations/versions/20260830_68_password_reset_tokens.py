"""single-use password reset grants

Invitations existed but were Entra-SSO only, and there was no password path at all: no
accept-with-password, no forgot, no reset, no change. An account created by an admin could
only be reached through Microsoft sign-in, and a user who forgot a password had no route
back other than asking an administrator to set one for them.

Only the SHA-256 of each token is stored, exactly as `invitations.token_hash` and
`refresh_sessions.token_hash` do it — a database dump must not hand out working reset
links. `used_at` marks a spent grant rather than deleting the row, so a replayed link is
distinguishable from an unknown one in the audit trail.

Guarded like 20260830_67 and 20260829_66: the baseline builds tables from the model
metadata, so a fresh database already has this table and a bare create_table would fail
with DuplicateTable.

Revision ID: 20260830_68
Revises: 20260830_67
Create Date: 2026-08-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260830_68"
down_revision = "20260830_67"
branch_labels = None
depends_on = None

TABLE = "password_reset_tokens"
ACTIVE_INDEX = "ix_password_reset_tokens_active"


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
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("used_at", sa.DateTime(), nullable=True),
            sa.Column("requested_for_email", sa.String(length=255), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("token_hash", name="uq_password_reset_tokens_token_hash"),
        )
        op.create_index(
            "ix_password_reset_tokens_user_id", TABLE, ["user_id"], unique=False
        )
        op.create_index(
            "ix_password_reset_tokens_token_hash", TABLE, ["token_hash"], unique=False
        )
        op.create_index(
            "ix_password_reset_tokens_expires_at", TABLE, ["expires_at"], unique=False
        )

    # Partial: the only lookup is "is there a live grant for this user", used to rate-limit
    # repeat requests. Spent and expired rows are the majority and are never queried.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {ACTIVE_INDEX}"
        f" ON {TABLE} (user_id, expires_at DESC)"
        " WHERE used_at IS NULL"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {ACTIVE_INDEX}")
    inspector = inspect(op.get_bind())
    if TABLE in set(inspector.get_table_names()):
        op.drop_table(TABLE)
