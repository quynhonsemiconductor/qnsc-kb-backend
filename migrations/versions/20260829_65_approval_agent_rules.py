"""add approval_rules for the draft approval agent

Structured filters decide which drafts a rule looks at; the instruction decides what to
do with them. The two authority columns default to FALSE at the database level as well as
in the model: a rule that nobody has explicitly allowed to act can still be written,
reviewed and dry-run, and will do nothing. Publishing to a whole company is not an
authority to acquire by leaving a field unset, so the safe default belongs here too and
not only in the application.

`created_by` is ON DELETE SET NULL rather than CASCADE. Deleting a person should not
silently delete the rules they wrote — that would change what the agent does as a side
effect of an HR action. A rule whose author is gone simply stops acting, because the
agent runs as its author and there is then nobody to run as.

The table creation is guarded and the indexes use IF NOT EXISTS, following
20260820_63. The baseline revision builds every table from the model metadata, so
declaring a new model creates this table during the baseline as well; an unguarded
create_table fails with DuplicateTableError on a FRESH database while working fine on an
existing one, which is the wrong way round for a migration to break.

Revision ID: 20260829_65
Revises: 20260828_64
Create Date: 2026-08-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision = "20260829_65"
down_revision = "20260828_64"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("approval_rules"):
        op.create_table(
            "approval_rules",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("company_domain", sa.String(length=255), nullable=False),
            sa.Column("name", sa.String(length=150), nullable=False),
            sa.Column(
                "active", sa.Boolean(), nullable=False, server_default=sa.text("true")
            ),
            sa.Column(
                "priority", sa.Integer(), nullable=False, server_default=sa.text("100")
            ),
            sa.Column("connector_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("dept", sa.String(length=100), nullable=True),
            sa.Column("file_extensions", sa.JSON(), nullable=True),
            sa.Column("max_similarity_score", sa.Float(), nullable=True),
            sa.Column("instruction", sa.Text(), nullable=False),
            # Fail closed. See the module docstring.
            sa.Column(
                "can_approve",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
            sa.Column(
                "can_reject",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
            sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
            ),
            sa.Column(
                "updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
            ),
            sa.ForeignKeyConstraint(
                ["connector_id"], ["connectors.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_rules_company_domain "
        "ON approval_rules (company_domain)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_rules_company_active "
        "ON approval_rules (company_domain, active)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_rules_connector_id "
        "ON approval_rules (connector_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_rules_created_by "
        "ON approval_rules (created_by)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS approval_rules")
