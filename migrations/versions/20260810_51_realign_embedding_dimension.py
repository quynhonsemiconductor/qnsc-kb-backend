"""re-align the pgvector column width after moving back to a local embedding model

20260810_36 narrowed `article_chunks.embedding` to whatever EMBEDDING_DIMENSION resolved
to at the time — 768, for the hosted gemini-embedding-001. EMBEDDING_MODEL is now
BAAI/bge-m3 again, which emits 1024, and indexing fails on every chunk with:

    asyncpg.exceptions.DataError: expected 768 dimensions, not 1024

A second migration rather than an edit to 20260810_36, because that one has already been
applied where this is deployed; Alembic will not re-run it. Both read
settings.EMBEDDING_DIMENSION and no-op when the column already matches, so on a database
migrated after the model change this one finds the width correct and does nothing.

Existing vectors CANNOT be converted: a 768-dimension vector is not a truncation of the
1024-dimension one, it is a point in a different space. So this refuses to run if any
embeddings are present rather than silently discarding them or, worse, leaving a corpus
where some vectors are comparable and others are not. Re-embedding is the only correct
path, and it is the operator's decision to schedule:

    DELETE FROM article_chunks;
    -- then re-run this migration and let the API's startup sweep re-index

Revision ID: 20260810_51
Revises: 20260810_36
Create Date: 2026-08-10
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings

revision = "20260810_51"
down_revision = "20260810_36"
branch_labels = None
depends_on = None

HNSW_INDEX = "ix_article_chunks_embedding_hnsw"


def _current_dimension(connection) -> int | None:
    """The declared width of article_chunks.embedding, or None if it is not a vector."""
    return connection.execute(
        sa.text(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_type t ON t.oid = a.atttypid "
            "WHERE c.relname = 'article_chunks' AND a.attname = 'embedding' AND t.typname = 'vector'"
        )
    ).scalar()


def upgrade() -> None:
    connection = op.get_bind()
    target = settings.EMBEDDING_DIMENSION
    current = _current_dimension(connection)
    if current is None or current == target:
        return

    populated = connection.execute(
        sa.text("SELECT count(*) FROM article_chunks WHERE embedding IS NOT NULL")
    ).scalar()
    if populated:
        raise RuntimeError(
            f"article_chunks.embedding is vector({current}) but EMBEDDING_DIMENSION is "
            f"{target}, and {populated} chunk(s) already carry embeddings. Vectors of "
            "different widths are not comparable, so this cannot be converted in place. "
            "Delete the chunks and re-index every article at the new width, then re-run "
            "this migration."
        )

    # HNSW indexes are bound to the column width and block the ALTER.
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")
    op.execute(
        f"ALTER TABLE article_chunks ALTER COLUMN embedding TYPE vector({target})"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} ON article_chunks "
        "USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL"
    )

    # Articles whose chunks were never written are still marked processing/ready from the
    # failed attempts; put them back in the queue the API's startup sweep drains.
    op.execute(
        "UPDATE articles SET index_status = 'pending', index_error = NULL "
        "WHERE status = 'published' AND index_status <> 'pending'"
    )


def downgrade() -> None:
    # The previous width is not recorded, and re-widening would not restore comparable
    # vectors anyway. Re-embedding is the only meaningful reverse, so this is a no-op.
    pass
