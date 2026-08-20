"""Record permission-leakage evidence for evaluation runs."""

from alembic import op
import sqlalchemy as sa


revision = "20260819_62"
down_revision = "20260819_61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("permission_leakage", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("eval_runs", "permission_leakage", server_default=None)


def downgrade() -> None:
    op.drop_column("eval_runs", "permission_leakage")
