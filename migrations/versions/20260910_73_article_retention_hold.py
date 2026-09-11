"""Add retention_until and legal_hold to articles

Compliance metadata independent of `lifecycle_status`. That field records what an editor
WANTS (active/archived); these two record what governance REQUIRES regardless of editor
intent: `legal_hold` blocks deletion outright, `retention_until` blocks it until a date.
Enforced in `domain/articles.py::soft_delete_article`, so a hold or open retention window
refuses even an authorized delete rather than merely warning about it.

Both default to "no restriction" (`legal_hold = false`, `retention_until = NULL`), so every
existing article is deletable exactly as it was before this migration.

Revision ID: 20260910_73
Revises: 20260901_72
Create Date: 2026-09-10
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260910_73"
down_revision = "20260901_72"
branch_labels = None
depends_on = None

TABLE = "articles"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "retention_until" not in columns:
        op.execute(f"ALTER TABLE {TABLE} ADD COLUMN retention_until DATE")
    if "legal_hold" not in columns:
        op.execute(
            f"ALTER TABLE {TABLE} ADD COLUMN legal_hold BOOLEAN NOT NULL DEFAULT false"
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "legal_hold" in columns:
        op.execute(f"ALTER TABLE {TABLE} DROP COLUMN legal_hold")
    if "retention_until" in columns:
        op.execute(f"ALTER TABLE {TABLE} DROP COLUMN retention_until")
