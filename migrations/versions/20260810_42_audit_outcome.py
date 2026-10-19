"""record audit action outcome

Revision ID: 20260810_42
Revises: 20260810_41
"""

from alembic import op
import sqlalchemy as sa

revision = "20260810_42"
down_revision = "20260810_41"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("audit_logs")}
    if "outcome" not in columns:
        op.add_column(
            "audit_logs",
            sa.Column(
                "outcome", sa.String(30), nullable=True, server_default="success"
            ),
        )
        op.execute("UPDATE audit_logs SET outcome = 'success' WHERE outcome IS NULL")
        op.alter_column(
            "audit_logs", "outcome", nullable=False, server_default="success"
        )


def downgrade() -> None:
    op.drop_column("audit_logs", "outcome")
