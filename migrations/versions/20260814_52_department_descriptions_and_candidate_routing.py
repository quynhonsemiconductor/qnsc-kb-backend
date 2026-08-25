"""add department descriptions and split-candidate routing suggestions

Revision ID: 20260814_52
Revises: 20260810_51
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260814_52"
down_revision = "20260810_51"
branch_labels = None
depends_on = None


def _missing(table: str, column: str) -> bool:
    """Whether this column still has to be added.

    The baseline revision (20260802_00) builds the schema with
    ``Base.metadata.create_all`` against the LIVE models, so on a database created
    from empty every column the models already declare exists before this migration
    runs, and a bare add_column fails with DuplicateColumn. An existing database
    predates those model fields and does need the ALTER. Both have to work: the
    first is how a new environment is provisioned, the second is what production is.
    """
    return column not in {
        item["name"] for item in inspect(op.get_bind()).get_columns(table)
    }


def upgrade() -> None:
    if _missing("departments", "description"):
        op.add_column(
            "departments",
            sa.Column(
                "description", sa.String(length=500), nullable=False, server_default=""
            ),
        )
        op.alter_column("departments", "description", server_default=None)
    for column in ("department_ids", "department_suggestions", "proposed_department"):
        if _missing("draft_candidates", column):
            op.add_column("draft_candidates", sa.Column(column, sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("draft_candidates", "proposed_department")
    op.drop_column("draft_candidates", "department_suggestions")
    op.drop_column("draft_candidates", "department_ids")
    op.drop_column("departments", "description")
