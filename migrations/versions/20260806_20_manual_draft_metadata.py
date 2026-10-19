"""store submitted metadata for manually authored approval drafts

Revision ID: 20260806_20
Revises: 20260806_19
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_20"
down_revision = "20260806_19"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The baseline migration builds ``Base.metadata`` for fresh installs, so
    # it can already include this model field.  Keep upgrades safe for both a
    # clean database and an existing database at revision 19.
    op.execute(
        "ALTER TABLE pending_drafts ADD COLUMN IF NOT EXISTS content_metadata JSON"
    )


def downgrade() -> None:
    op.drop_column("pending_drafts", "content_metadata")
