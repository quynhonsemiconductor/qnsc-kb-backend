"""mark connector-managed Article user permissions

Revision ID: 20260810_47
Revises: 20260810_46
"""

from alembic import op
import sqlalchemy as sa

revision = "20260810_47"
down_revision = "20260810_46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column["name"] for column in inspector.get_columns("article_user_permissions")
    }
    if "source" not in columns:
        op.add_column(
            "article_user_permissions",
            sa.Column("source", sa.String(40), nullable=True),
        )
    constraints = {
        item["name"]
        for item in inspector.get_unique_constraints("article_user_permissions")
    }
    if "uq_article_user_permission" in constraints:
        op.drop_constraint(
            "uq_article_user_permission", "article_user_permissions", type_="unique"
        )
    indexes = {
        item["name"]
        for item in sa.inspect(bind).get_indexes("article_user_permissions")
    }
    if "uq_article_user_permission_internal" not in indexes:
        op.create_index(
            "uq_article_user_permission_internal",
            "article_user_permissions",
            ["article_id", "user_id"],
            unique=True,
            postgresql_where=sa.text("source IS NULL"),
        )
    if "uq_article_user_permission_source" not in indexes:
        op.create_index(
            "uq_article_user_permission_source",
            "article_user_permissions",
            ["article_id", "user_id", "source"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index(
        "uq_article_user_permission_source", table_name="article_user_permissions"
    )
    op.drop_index(
        "uq_article_user_permission_internal", table_name="article_user_permissions"
    )
    op.create_unique_constraint(
        "uq_article_user_permission",
        "article_user_permissions",
        ["article_id", "user_id"],
    )
    op.drop_column("article_user_permissions", "source")
