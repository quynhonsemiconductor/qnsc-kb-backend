"""add Article visibility and explicit user overrides

Revision ID: 20260810_40
Revises: 20260809_39
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260810_40"
down_revision = "20260809_39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("articles")}
    if "visibility" not in columns:
        op.add_column("articles", sa.Column("visibility", sa.String(30), nullable=True))
        op.execute(
            "UPDATE articles SET visibility = CASE WHEN sensitivity = 'public' THEN 'public' ELSE 'department' END"
        )
        op.alter_column(
            "articles", "visibility", nullable=False, server_default="department"
        )

    if not inspector.has_table("article_user_permissions"):
        op.create_table(
            "article_user_permissions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "article_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("articles.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("effect", sa.String(10), nullable=False, server_default="allow"),
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
                "article_id", "user_id", name="uq_article_user_permission"
            ),
            sa.CheckConstraint(
                "effect IN ('allow', 'deny')", name="ck_article_user_permission_effect"
            ),
        )
        op.create_index(
            "ix_article_user_permissions_article_id",
            "article_user_permissions",
            ["article_id"],
        )
        op.create_index(
            "ix_article_user_permissions_user_id",
            "article_user_permissions",
            ["user_id"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_article_user_permissions_user_id", table_name="article_user_permissions"
    )
    op.drop_index(
        "ix_article_user_permissions_article_id", table_name="article_user_permissions"
    )
    op.drop_table("article_user_permissions")
    op.drop_column("articles", "visibility")
