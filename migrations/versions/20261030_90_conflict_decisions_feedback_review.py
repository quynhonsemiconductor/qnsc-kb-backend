"""conflict decisions and feedback review state

`conflict_records` learns which article a reviewer decided is right and how the conflict
was found; `ai_feedback` learns whether a reviewer has handled a thumbs-down.

Guarded like the other additive migrations: the baseline builds tables from the model
metadata, so a fresh database already has the columns.

Revision ID: 20261030_90
Revises: 20261030_89
Create Date: 2026-10-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261030_90"
down_revision = "20261030_89"
branch_labels = None
depends_on = None


def _added() -> dict[str, list[sa.Column]]:
    # Built on every call: a Column belongs to one table once it is added to it.
    return {
        "conflict_records": [
            sa.Column("kept_article_id", sa.String(length=64), nullable=True),
            sa.Column(
                "detection_method",
                sa.String(length=20),
                nullable=False,
                server_default="pattern",
            ),
        ],
        "ai_feedback": [
            sa.Column(
                "reviewed_by",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("reviewed_at", sa.DateTime(), nullable=True),
            sa.Column("review_note", sa.Text(), nullable=True),
        ],
    }


def upgrade() -> None:
    bind = op.get_bind()
    for table, columns in _added().items():
        existing = {c["name"] for c in inspect(bind).get_columns(table)}
        for column in columns:
            if column.name not in existing:
                op.add_column(table, column)


def downgrade() -> None:
    bind = op.get_bind()
    for table, columns in _added().items():
        existing = {c["name"] for c in inspect(bind).get_columns(table)}
        for column in reversed(columns):
            if column.name in existing:
                op.drop_column(table, column.name)
