"""add bounded document evidence and link it to searchable chunks

Revision ID: 20261011_82
Revises: 20260912_81
Create Date: 2026-10-11
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20261011_82"
down_revision = "20260912_81"
branch_labels = None
depends_on = None

TABLE = "document_evidence"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
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
            ),
            sa.Column(
                "document_source_id",
                sa.UUID(),
                sa.ForeignKey("document_sources.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(30), nullable=False),
            sa.Column("page_number", sa.Integer(), nullable=False),
            sa.Column("bounding_box", sa.JSON(), nullable=True),
            sa.Column("page_width", sa.Float(), nullable=True),
            sa.Column("page_height", sa.Float(), nullable=True),
            sa.Column("label", sa.String(255), nullable=False),
            sa.Column("search_text", sa.Text(), nullable=False),
            sa.Column("structured_data", sa.JSON(), nullable=True),
            sa.Column("content_hash", sa.String(64), nullable=False),
            sa.Column("extraction_version", sa.String(50), nullable=False),
            sa.CheckConstraint(
                "kind IN ('table', 'figure', 'page_visual')",
                name="ck_document_evidence_kind",
            ),
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_evidence_article_kind ON document_evidence (article_id, kind)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_evidence_source_page ON document_evidence (document_source_id, page_number)"
    )

    inspector = inspect(op.get_bind())
    for chunk_table in ("parent_chunks", "article_chunks"):
        if chunk_table not in set(inspector.get_table_names()):
            continue
        columns = {column["name"] for column in inspector.get_columns(chunk_table)}
        if "evidence_id" not in columns:
            op.add_column(
                chunk_table, sa.Column("evidence_id", sa.UUID(), nullable=True)
            )
            op.create_foreign_key(
                f"fk_{chunk_table}_evidence_id",
                chunk_table,
                TABLE,
                ["evidence_id"],
                ["id"],
                ondelete="CASCADE",
            )
        op.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{chunk_table}_evidence_id ON {chunk_table} (evidence_id)"
        )

    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_document_evidence ON {TABLE} USING "
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

    # Existing published documents already have searchable text chunks. Keep those live
    # while the durable outbox re-indexes each source atomically; changing index_status to
    # pending would be undone by startup reconciliation because the old chunks still exist,
    # and deleting them here would create a corpus-wide search outage during deployment.
    # The reason marker makes this idempotent on a manually re-run guarded migration.
    if {"articles", "document_sources", "outbox_events"} <= tables:
        op.execute("""
            INSERT INTO outbox_events (
                id, created_at, updated_at, event_type, payload, status,
                attempts, next_attempt_at, last_error
            )
            SELECT
                gen_random_uuid(), now(), now(), 'ArticleUpdated',
                json_build_object(
                    'article_id', article.id::text,
                    'reason', 'document_evidence_backfill'
                ),
                'pending', 0, now(), NULL
            FROM articles AS article
            WHERE article.status = 'published'
              AND article.lifecycle_status = 'active'
              AND EXISTS (
                  SELECT 1
                  FROM document_sources AS source
                  WHERE source.article_id = article.id
                    AND source.storage_key IS NOT NULL
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM outbox_events AS existing
                  WHERE existing.event_type = 'ArticleUpdated'
                    AND existing.payload ->> 'article_id' = article.id::text
                    AND existing.payload ->> 'reason' = 'document_evidence_backfill'
                    AND existing.status IN ('pending', 'processing', 'dispatched')
              )
            """)


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE in set(inspector.get_table_names()):
        op.execute(f"DROP POLICY IF EXISTS tenant_document_evidence ON {TABLE}")
    for chunk_table in ("article_chunks", "parent_chunks"):
        if chunk_table not in set(inspector.get_table_names()):
            continue
        columns = {column["name"] for column in inspector.get_columns(chunk_table)}
        if "evidence_id" in columns:
            op.execute(f"DROP INDEX IF EXISTS ix_{chunk_table}_evidence_id")
            op.drop_constraint(
                f"fk_{chunk_table}_evidence_id", chunk_table, type_="foreignkey"
            )
            op.drop_column(chunk_table, "evidence_id")
    if TABLE in set(inspect(op.get_bind()).get_table_names()):
        op.drop_table(TABLE)
