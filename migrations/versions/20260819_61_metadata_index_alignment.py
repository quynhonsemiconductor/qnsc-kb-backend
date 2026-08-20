"""Align tenant indexes declared by the ORM with the live schema.

The tenant-owned tag, conflict, and evaluation-set models declare an index on
company_domain.  Earlier table-creation migrations intentionally created only
their composite/unique indexes, so a fresh database and an existing database
could disagree with the ORM metadata.  Keep the single-column indexes explicit
and idempotent so Alembic validation is clean on both paths.
"""

from alembic import op


revision = "20260819_61"
down_revision = "20260819_60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_conflict_records_company_domain",
        "conflict_records",
        ["company_domain"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_eval_sets_company_domain",
        "eval_sets",
        ["company_domain"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_tag_catalog_company_domain",
        "tag_catalog",
        ["company_domain"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_tag_catalog_company_domain", table_name="tag_catalog")
    op.drop_index("ix_eval_sets_company_domain", table_name="eval_sets")
    op.drop_index("ix_conflict_records_company_domain", table_name="conflict_records")
