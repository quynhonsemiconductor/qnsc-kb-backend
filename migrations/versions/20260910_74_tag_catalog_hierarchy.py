"""Add parent_id to tag_catalog for a hierarchical tag taxonomy

The catalog was a flat, tenant-owned vocabulary (uq_tag_catalog_company_normalized). This
adds one self-referencing, nullable column so a tag may optionally name a parent already
in the same tenant's catalog -- "PCCC" under "An toàn", say. A tag with no parent is a
root, exactly the flat shape every existing row already has, so this is additive only:
no existing tag becomes invalid and no query that ignores the column changes behavior.

ON DELETE SET NULL, not CASCADE: deprecating or deleting a parent tag must not silently
delete every tag beneath it -- those children just become roots, which a human can then
re-parent deliberately rather than losing them to a cascade.

Revision ID: 20260910_74
Revises: 20260910_73
Create Date: 2026-09-10
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260910_74"
down_revision = "20260910_73"
branch_labels = None
depends_on = None

TABLE = "tag_catalog"
FK_NAME = "fk_tag_catalog_parent_id"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "parent_id" not in columns:
        op.execute(f"ALTER TABLE {TABLE} ADD COLUMN parent_id UUID")
        op.execute(
            f"ALTER TABLE {TABLE} ADD CONSTRAINT {FK_NAME} "
            f"FOREIGN KEY (parent_id) REFERENCES {TABLE}(id) ON DELETE SET NULL"
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}

    if "parent_id" in columns:
        op.execute(f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {FK_NAME}")
        op.execute(f"ALTER TABLE {TABLE} DROP COLUMN parent_id")
