"""keep the exception that caused a 500, not just the fact that one happened

The request middleware always caught every unhandled exception and logged it, but only the
status code reached the database. Diagnosing a production failure therefore required
reading CloudWatch, which needs an AWS role switch -- so a 500 was effectively
undiagnosable by anyone without console access. The information existed and was discarded
one line after being formatted.

Both columns are nullable and written only for failures, so successful traffic does not
grow the row. The partial index covers the only query that matters -- recent failures,
newest first -- rather than indexing the 200s that make up almost all of the table.

Guarded like 20260829_66 and 20260820_63: the baseline builds tables from the model
metadata, so a fresh database already has these columns and a plain add_column would fail
with DuplicateColumnError. Measured -- that is exactly how this migration failed first.

Revision ID: 20260830_67
Revises: 20260829_66
Create Date: 2026-08-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260830_67"
down_revision = "20260829_66"
branch_labels = None
depends_on = None

TABLE = "api_request_metrics"
FAILURE_INDEX = "ix_api_request_metrics_failures"
NEW_COLUMNS = (
    ("error_type", sa.String(length=100)),
    ("error_detail", sa.Text()),
)


def upgrade() -> None:
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns(TABLE)}
    for name, column_type in NEW_COLUMNS:
        if name not in columns:
            op.add_column(TABLE, sa.Column(name, column_type, nullable=True))
    # Partial and DESC: the endpoint reads the newest failures, and indexing every 200
    # would cover most of the table for none of the queries that are actually run.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {FAILURE_INDEX}"
        f" ON {TABLE} (created_at DESC, path)"
        " WHERE status_code >= 500"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {FAILURE_INDEX}")
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns(TABLE)}
    for name, _column_type in reversed(NEW_COLUMNS):
        if name in columns:
            op.drop_column(TABLE, name)
