"""keep a short digest of each document for questions that span many documents

A question such as "summarise our HR policies" cannot be answered from the nearest passages,
and rereading every document per question is the cost this table exists to avoid. A digest is
made once per version of a document's content (`content_hash`) and reused until it changes.

Guarded like the other additive migrations: the baseline builds tables from the model
metadata, so a fresh database already has this one. The embedding column is created through
`Vector(...)` with the configured dimension, exactly as `article_chunks.embedding` is.

Revision ID: 20261027_86
Revises: 20261026_85
Create Date: 2026-10-27
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

from src.core.config import settings
from src.models.chunk import Vector

revision = "20261027_86"
down_revision = "20261026_85"
branch_labels = None
depends_on = None

TABLE = "article_digests"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE not in set(inspector.get_table_names()):
        op.create_table(
            TABLE,
            sa.Column("id", sa.UUID(), primary_key=True),
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
            sa.Column(
                "article_id",
                sa.UUID(),
                sa.ForeignKey("articles.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("content_hash", sa.String(64), nullable=False),
            sa.Column("digest_version", sa.String(50), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("summary", sa.Text(), nullable=False),
            sa.Column("topics", sa.JSON(), nullable=True),
            sa.Column("entities", sa.JSON(), nullable=True),
            sa.Column("key_numbers", sa.JSON(), nullable=True),
            sa.Column("doc_language", sa.String(10), nullable=True),
            sa.Column("truncated", sa.Boolean(), nullable=False),
            sa.Column("model", sa.String(100), nullable=True),
            sa.Column("tokens", sa.Integer(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("attempts", sa.Integer(), nullable=False),
            sa.Column("embedding", Vector(settings.EMBEDDING_DIMENSION), nullable=True),
            sa.Column("embedding_version", sa.String(50), nullable=True),
            sa.CheckConstraint(
                "status IN ('ready', 'failed')", name="ck_article_digests_status"
            ),
        )

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_article_digests ON {TABLE} USING "
            "(current_setting('app.global_admin', true) = 'true' OR EXISTS "
            "(SELECT 1 FROM articles a WHERE a.id = article_id AND "
            "a.company_domain = current_setting('app.company_domain', true))) "
            "WITH CHECK (current_setting('app.global_admin', true) = 'true' OR EXISTS "
            "(SELECT 1 FROM articles a WHERE a.id = article_id AND "
            "a.company_domain = current_setting('app.company_domain', true)))"
        )
    role = os.getenv("APP_DATABASE_ROLE")
    if role:
        safe_role = role.replace('"', '""')
        op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO "{safe_role}"')


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE in set(inspector.get_table_names()):
        op.execute(f"DROP POLICY IF EXISTS tenant_article_digests ON {TABLE}")
    op.execute(f"DROP TABLE IF EXISTS {TABLE}")
