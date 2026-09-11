"""Add structured_metadata to articles

Holds document identity fields (document_number, issue_date, expiry_date, signed_by)
extracted best-effort from the document text -- see domain/structured_metadata.py. A
single nullable JSON column rather than four separate ones: the field set is still
settling, and every consumer already reads it as a dict with `.get(key)` rather than as
named ORM attributes, so widening it later needs no migration.

Revision ID: 20260910_75
Revises: 20260910_74
Create Date: 2026-09-10
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260910_75"
down_revision = "20260910_74"
branch_labels = None
depends_on = None

TABLE = "articles"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "structured_metadata" not in columns:
        op.execute(f"ALTER TABLE {TABLE} ADD COLUMN structured_metadata JSON")


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "structured_metadata" in columns:
        op.execute(f"ALTER TABLE {TABLE} DROP COLUMN structured_metadata")
