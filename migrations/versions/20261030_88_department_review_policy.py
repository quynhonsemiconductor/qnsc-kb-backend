"""let a department publish drafts without a review

`review_required` defaults to true, so every existing department keeps behaving as before.
`auto_publish_by` names the Admin/CEO who turned review off: drafts are approved as them.

Guarded like the other additive migrations: the baseline builds tables from the model
metadata, so a fresh database already has the columns.

Revision ID: 20261030_88
Revises: 20261028_87
Create Date: 2026-10-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261030_88"
down_revision = "20261028_87"
branch_labels = None
depends_on = None

TABLE = "departments"


def upgrade() -> None:
    existing = {c["name"] for c in inspect(op.get_bind()).get_columns(TABLE)}
    if "review_required" not in existing:
        op.add_column(
            TABLE,
            sa.Column(
                "review_required",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )
    if "auto_publish_by" not in existing:
        op.add_column(
            TABLE,
            sa.Column(
                "auto_publish_by",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )


def downgrade() -> None:
    existing = {c["name"] for c in inspect(op.get_bind()).get_columns(TABLE)}
    for name in ("auto_publish_by", "review_required"):
        if name in existing:
            op.drop_column(TABLE, name)
