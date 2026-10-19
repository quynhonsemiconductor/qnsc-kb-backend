"""record the actor that requested a connector sync

Revision ID: 20260810_49
Revises: 20260810_48
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_49"
down_revision = "20260810_48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("connector_jobs")
    }
    if "requested_by" not in columns:
        op.add_column(
            "connector_jobs",
            sa.Column(
                "requested_by",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index(
            "ix_connector_jobs_requested_by", "connector_jobs", ["requested_by"]
        )


def downgrade() -> None:
    op.drop_index("ix_connector_jobs_requested_by", table_name="connector_jobs")
    op.drop_column("connector_jobs", "requested_by")
