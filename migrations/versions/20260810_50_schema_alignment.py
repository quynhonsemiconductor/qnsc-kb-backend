"""align current metadata with the persisted compatibility schema

Revision ID: 20260810_50
Revises: 20260810_49
"""

from alembic import op

revision = "20260810_50"
down_revision = "20260810_49"
branch_labels = None
depends_on = None


_INDEXES = (
    ("ix_articles_company_domain", "articles", "company_domain"),
    ("ix_articles_lifecycle_status", "articles", "lifecycle_status"),
    ("ix_connectors_company_domain", "connectors", "company_domain"),
    ("ix_pending_drafts_dept", "pending_drafts", "dept"),
    (
        "ix_pending_drafts_external_document_id",
        "pending_drafts",
        "external_document_id",
    ),
    ("ix_users_active", "users", "active"),
    ("ix_users_company_domain", "users", "company_domain"),
)


def upgrade() -> None:
    for name, table, column in _INDEXES:
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})")


def downgrade() -> None:
    for name, _, _ in reversed(_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name}")
