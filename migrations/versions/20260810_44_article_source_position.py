"""store source position for split article provenance

Revision ID: 20260810_44
Revises: 20260810_43
"""

from alembic import op
import sqlalchemy as sa

revision = "20260810_44"
down_revision = "20260810_43"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("articles")}
    if "source_position" not in columns:
        op.add_column(
            "articles", sa.Column("source_position", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    op.drop_column("articles", "source_position")
