"""remember which device each sign-in is

`refresh_sessions` held only a token hash and expiry, so a user had no way to see where
they were signed in or to end one session. These nullable columns let the account page
list sessions and revoke one; a session is rotated on every refresh, so `family_id` and
`signed_in_at` carry the original sign-in across the rotation.

Nullable, because rows issued before this migration have none of it. Guarded like the
other additive migrations: the baseline builds tables from the model metadata, so a
fresh database already has the columns.

Revision ID: 20261026_85
Revises: 20261019_84
Create Date: 2026-10-26

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261026_85"
down_revision = "20261019_84"
branch_labels = None
depends_on = None

TABLE = "refresh_sessions"
COLUMNS = [
    ("family_id", sa.UUID()),
    ("user_agent", sa.String(length=300)),
    ("ip_address", sa.String(length=64)),
    ("signed_in_at", sa.DateTime()),
    ("last_used_at", sa.DateTime()),
]


def upgrade() -> None:
    existing = {c["name"] for c in inspect(op.get_bind()).get_columns(TABLE)}
    for name, kind in COLUMNS:
        if name not in existing:
            op.add_column(TABLE, sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    existing = {c["name"] for c in inspect(op.get_bind()).get_columns(TABLE)}
    for name, _ in reversed(COLUMNS):
        if name in existing:
            op.drop_column(TABLE, name)
