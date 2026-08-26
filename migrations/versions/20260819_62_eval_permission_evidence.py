"""Record permission-leakage evidence for evaluation runs."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260819_62"
down_revision = "20260819_61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # See 20260814_52: the baseline builds the schema from the live models, so this
    # column already exists on a database created from empty.
    existing = {item["name"] for item in inspect(op.get_bind()).get_columns("eval_runs")}
    if "permission_leakage" not in existing:
        op.add_column(
            "eval_runs",
            sa.Column("permission_leakage", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.alter_column("eval_runs", "permission_leakage", server_default=None)


def downgrade() -> None:
    op.drop_column("eval_runs", "permission_leakage")
