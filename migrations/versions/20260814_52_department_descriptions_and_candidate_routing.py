"""add department descriptions and split-candidate routing suggestions

Revision ID: 20260814_52
Revises: 20260810_51
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_52"
down_revision = "20260810_51"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "departments",
        sa.Column(
            "description", sa.String(length=500), nullable=False, server_default=""
        ),
    )
    op.alter_column("departments", "description", server_default=None)
    op.add_column(
        "draft_candidates", sa.Column("department_ids", sa.JSON(), nullable=True)
    )
    op.add_column(
        "draft_candidates",
        sa.Column("department_suggestions", sa.JSON(), nullable=True),
    )
    op.add_column(
        "draft_candidates", sa.Column("proposed_department", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("draft_candidates", "proposed_department")
    op.drop_column("draft_candidates", "department_suggestions")
    op.drop_column("draft_candidates", "department_ids")
    op.drop_column("departments", "description")
