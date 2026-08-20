"""Completion-plan foundations: audit details, invitations, delivery state."""

from alembic import op
import sqlalchemy as sa

revision = "20260816_54"
down_revision = "20260816_53"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.add_column("audit_logs", sa.Column("detail_json", sa.JSON(), nullable=True))
    op.add_column("articles", sa.Column("self_approved", sa.Boolean(), nullable=False, server_default="false"))
    op.alter_column("articles", "self_approved", server_default=None)
    op.add_column("notification_queue", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("notification_queue", sa.Column("last_error", sa.Text(), nullable=True))
    op.alter_column("notification_queue", "attempts", server_default=None)
    op.create_table(
        "invitations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="Staff"),
        sa.Column("company_domain", sa.String(length=255), nullable=False),
        sa.Column("audience_ids", sa.JSON(), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("invited_by", sa.UUID(), nullable=True),
        sa.Column("accepted_user_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["accepted_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_invitations_token_hash"),
    )
    op.create_index("ix_invitations_email", "invitations", ["email"])
    op.create_index("ix_invitations_company_domain", "invitations", ["company_domain"])
    op.create_index("ix_invitations_email_domain", "invitations", ["email", "company_domain"])


def downgrade() -> None:
    op.drop_index("ix_invitations_email_domain", table_name="invitations")
    op.drop_index("ix_invitations_company_domain", table_name="invitations")
    op.drop_index("ix_invitations_email", table_name="invitations")
    op.drop_table("invitations")
    op.drop_column("notification_queue", "last_error")
    op.drop_column("notification_queue", "attempts")
    op.drop_column("audit_logs", "detail_json")
    op.drop_column("articles", "self_approved")
