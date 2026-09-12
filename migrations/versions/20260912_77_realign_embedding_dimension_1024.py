"""re-align the pgvector column width to 1024 after moving to multilingual-e5-large-instruct

EMBEDDING_MODEL moved from intfloat/multilingual-e5-small (384-dim) to
intfloat/multilingual-e5-large-instruct (1024-dim) -- see the long note beside
EMBEDDING_MODEL in src/core/config.py for the VN-MTEB numbers behind the swap. Unlike the
e5-small adoption (same 384 width as the MiniLM it replaced), this one changes the column
width, so it needs the ALTER-COLUMN-and-rebuild-the-HNSW-index template that
20260810_51/20260828_64 established, not the delete-and-requeue template
20260831_69 used for the same-width e5-small swap.

Both earlier width-change revisions read settings.EMBEDDING_DIMENSION and no-op when the
column already matches, so this is harmless on a database already at 1024 for some other
reason.

Existing vectors CANNOT be converted: a 1024-dimension vector is not an extension of the
384-dimension one, it is a point in a different space. So this refuses to run if any
embeddings are present rather than silently discarding them:

    DELETE FROM article_chunks;
    -- then re-run this migration and let the API's startup sweep re-index

Revision ID: 20260912_77
Revises: 20260911_76
Create Date: 2026-09-12
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings


revision = "20260912_77"
down_revision = "20260911_76"
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

    # Every published article is currently unsearchable at the new width. Put them back in
    # the queue the API's startup sweep drains, or they stay unsearchable with no further
    # attempt: nothing retries a failed index on its own.
    op.execute(
        "UPDATE articles SET index_status = 'pending', index_error = NULL "
        "WHERE status = 'published' AND index_status <> 'pending'"
    )


def downgrade() -> None:
    # The previous width is not recorded, and re-narrowing would not restore comparable
    # vectors anyway. Re-embedding at the old model/width is the only meaningful reverse,
    # so this is a no-op.
    pass
