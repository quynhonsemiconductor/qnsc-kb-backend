"""remove duplicated user content from operational telemetry

Revision ID: 20260807_25
Revises: 20260806_24
Create Date: 2026-08-07
"""

from alembic import op

revision = "20260807_25"
down_revision = "20260806_24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Search and AI usage telemetry is consumed only as aggregates. Historic
    # raw question/answer copies are therefore unnecessary sensitive data.
    op.execute(
        "UPDATE search_logs SET query = '[redacted]' WHERE query <> '[redacted]'"
    )
    op.execute(
        "UPDATE ai_usage_logs SET question = '[redacted]', answer = '[redacted]' WHERE question <> '[redacted]' OR answer <> '[redacted]'"
    )


def downgrade() -> None:
    # Redaction intentionally cannot restore private content.
    pass
