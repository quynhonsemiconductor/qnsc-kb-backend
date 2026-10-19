"""align the pgvector column width with the configured embedding dimension

The `article_chunks.embedding` column was created at 1024 dimensions, back when the
default EMBEDDING_MODEL was BGE-M3. The deployed model is gemini-embedding-001 asked for
768 dimensions, so indexing failed on every chunk with:

    asyncpg.exceptions.DataError: expected 1024 dimensions, not 768

pgvector fixes the width in the column type, so the model definition
(`Vector(settings.EMBEDDING_DIMENSION)`) has no effect on a table that already exists —
only a migration can change it.

Existing vectors CANNOT be converted: a 1024-dimension vector is not a truncation of the
768-dimension one, it is a point in a different space. So this migration refuses to run
if any embeddings are present rather than silently discarding them or, worse, leaving a
corpus where some vectors are comparable and others are not. Re-embedding is the only
correct path, and it is the operator's decision to schedule.

Revision ID: 20260810_36
Revises: 20260810_50
Create Date: 2026-08-10

Authored on main against 20260807_35, then re-pointed onto develop's head when main was
merged in: the two branches had both added migrations after 20260807_35, and Alembic
takes one linear head. Nothing between 20260807_35 and 20260810_50 touches
article_chunks.embedding, so running later changes nothing about what this does.
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings

revision = "20260810_36"
down_revision = "20260810_50"
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
