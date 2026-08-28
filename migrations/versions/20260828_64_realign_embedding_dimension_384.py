"""re-align the pgvector column width to 384 after moving to MiniLM

#77 changed EMBEDDING_MODEL to paraphrase-multilingual-MiniLM-L12-v2, which emits 384,
and said 20260810_51 would re-align the column. IT CANNOT. That revision is already in
alembic_version everywhere this is deployed, and Alembic never re-runs an applied
revision — which is the very reason 20260810_51 itself had to exist rather than editing
20260810_36. The same trap, one revision later.

So the column stayed vector(1024) from the bge-m3 era while the model began emitting 384,
and every article failed to index:

    asyncpg.exceptions.DataError: expected 1024 dimensions, not 384

surfacing in the UI as "Search index: failed" and, because no chunk was ever written, as
an assistant that answers nothing about any published article.

Both earlier revisions read settings.EMBEDDING_DIMENSION and no-op when the column
already matches, so this one is harmless on a database created after the model change.

Existing vectors CANNOT be converted: a 1024-dimension vector is not a truncation of the
384-dimension one, it is a point in a different space. So this refuses to run if any
embeddings are present rather than silently discarding them. In practice none can exist
here — before #77 base.py rejected the 384-wide output before it ever reached the
database, so nothing was ever stored — but the guard is kept rather than assumed away:

    DELETE FROM article_chunks;
    -- then re-run this migration and let the API's startup sweep re-index

Revision ID: 20260828_64
Revises: 20260820_63
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings


revision = "20260828_64"
down_revision = "20260820_63"
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
    op.execute(f"ALTER TABLE article_chunks ALTER COLUMN embedding TYPE vector({target})")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} ON article_chunks "
        "USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL"
    )

    # Every published article is currently "failed" from the width mismatch. Put them
    # back in the queue the API's startup sweep drains, or they stay unsearchable with no
    # further attempt: nothing retries a failed index on its own.
    op.execute(
        "UPDATE articles SET index_status = 'pending', index_error = NULL "
        "WHERE status = 'published' AND index_status <> 'pending'"
    )


def downgrade() -> None:
    # The previous width is not recorded, and re-widening would not restore comparable
    # vectors anyway. Re-embedding is the only meaningful reverse, so this is a no-op.
    pass
