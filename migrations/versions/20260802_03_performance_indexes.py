"""add retrieval and operational indexes"""

from alembic import op

revision = "20260802_03"
down_revision = "20260802_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE ai_cache ADD COLUMN IF NOT EXISTS article_ids JSON")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_chunks_embedding_hnsw ON article_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_chunks_article_id ON article_chunks (article_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_chunks_permission_lookup ON article_chunks (article_id, department_id, visibility) WHERE embedding IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_chunks_fts ON article_chunks USING gin (to_tsvector('simple', chunk_text))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_versions_article_id ON article_versions (article_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_sources_hash ON document_sources (source_hash)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_article_access_group_id ON article_access (group_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_ai_cache_expiry ON ai_cache (expires_at)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs (created_at)"
    )


def downgrade() -> None:
    for name in (
        "ix_audit_logs_created_at",
        "ix_ai_cache_expiry",
        "ix_article_access_group_id",
        "ix_document_sources_hash",
        "ix_article_versions_article_id",
        "ix_article_chunks_fts",
        "ix_article_chunks_permission_lookup",
        "ix_article_chunks_article_id",
        "ix_article_chunks_embedding_hnsw",
    ):
        op.execute(f"DROP INDEX IF EXISTS {name}")
