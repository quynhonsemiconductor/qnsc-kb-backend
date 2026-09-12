"""add contradiction_type to conflict_records

Reviewer triage on the Coverage page currently has nothing to sort or filter open
conflicts by beyond the free-text `fact` label. `fact` is already one of a small, fixed
vocabulary the detection regex produces (`_EXPLICIT_FACT_RE` in ai_service.py) --
"effective date", "deadline", "status", "limit", "owner", ... -- so classifying it into a
taxonomy (date / status / numerical / ownership / other,
`ai_service.classify_fact_type`) is a lookup over that fixed set, not a new detector.

Nullable and backfilled rather than a hard requirement: existing open conflicts predate
this column and still need to display correctly without it, and any future detection path
that does not classify must still be able to write a valid ConflictRecord.

Revision ID: 20260912_79
Revises: 20260912_78
Create Date: 2026-09-12
"""

from alembic import op
import sqlalchemy as sa


revision = "20260912_79"
down_revision = "20260912_78"
branch_labels = None
depends_on = None

# Mirrors src/domain/ai_service.py::FACT_TAXONOMY. Kept as a literal here rather than
# imported: a migration must keep working after the application code that inspired it
# changes shape, and this mapping is over a closed, historical set of fact labels that
# will not change retroactively for rows already in the table.
_FACT_TAXONOMY = {
    "effective date": "date",
    "deadline": "date",
    "approval deadline": "date",
    "retention period": "date",
    "status": "status",
    "limit": "numerical",
    "owner": "ownership",
}


def upgrade() -> None:
    op.add_column(
        "conflict_records",
        sa.Column("contradiction_type", sa.String(length=40), nullable=True),
    )
    connection = op.get_bind()
    for fact_label, category in _FACT_TAXONOMY.items():
        connection.execute(
            sa.text(
                "UPDATE conflict_records SET contradiction_type = :category "
                "WHERE fact = :fact_label AND contradiction_type IS NULL"
            ),
            {"category": category, "fact_label": fact_label},
        )
    # Anything already stored under a label outside the fixed set above (should not
    # happen given the regex, but this is a backfill of historical data, not a
    # constraint) is left NULL rather than guessed at -- "other" is reserved for the
    # application's own classify_fact_type fallback going forward, not retrofitted onto
    # rows this migration cannot actually verify.


def downgrade() -> None:
    op.drop_column("conflict_records", "contradiction_type")
