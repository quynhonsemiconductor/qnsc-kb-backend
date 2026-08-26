"""Persist AI action metadata for refresh-safe confirmations."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260819_60"
down_revision = "20260819_59"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # See 20260814_52: the baseline builds the schema from the live models, so this
    # column already exists on a database created from empty.
    existing = {item["name"] for item in inspect(op.get_bind()).get_columns("ai_messages")}
    if "action_data" not in existing:
        op.add_column("ai_messages", sa.Column("action_data", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_messages", "action_data")
