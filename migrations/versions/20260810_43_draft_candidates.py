"""store structure-aware article candidates for batch review

Revision ID: 20260810_43
Revises: 20260810_42
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_43"
down_revision = "20260810_42"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("draft_candidates"):
        op.create_table(
            "draft_candidates",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "draft_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("pending_drafts.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("position", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("body_md", sa.Text(), nullable=False),
            sa.Column("source_start", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("source_end", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("heading", sa.String(255), nullable=True),
            sa.Column(
                "status", sa.String(30), nullable=False, server_default="candidate"
            ),
            sa.Column("review_note", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.UniqueConstraint(
                "draft_id", "position", name="uq_draft_candidate_position"
            ),
        )
        op.create_index(
            "ix_draft_candidates_draft_id", "draft_candidates", ["draft_id"]
        )


def downgrade() -> None:
    op.drop_index("ix_draft_candidates_draft_id", table_name="draft_candidates")
    op.drop_table("draft_candidates")
