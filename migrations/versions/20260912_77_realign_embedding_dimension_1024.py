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
384-dimension one, it is a point in a different space.

WHAT THIS REFUSES ON, AND WHAT IT NO LONGER REFUSES ON. As first written this raised
whenever ANY embedding was present, which deadlocked every environment it was supposed to
fix: develop held a full corpus of e5-small vectors, so the migration could not run, so
revisions 78-82 behind it could not run either, and indexing failed on every article with
either

    asyncpg.exceptions.DataError: expected 384 dimensions, not 1024

or -- once the chain had fallen far enough behind -- a missing column from one of the
revisions stuck behind this one. Neither symptom names this migration, and the operator's
only documented way out was to hand-delete the corpus on a live database.

So the refusal is now narrowed to the case it was actually protecting: vectors stamped
with the CURRENT EMBEDDING_VERSION, which would mean genuinely current-space data that
someone should look at before it goes. Vectors stamped with a superseded model are cleared
instead, because they are unusable on two independent grounds -- wrong width by
construction, and already filtered out of hybrid_search by their stamp -- and because
article_chunks/parent_chunks are DERIVED tables, rebuilt from articles.content by the
indexing sweep this migration re-queues. No source document is touched.

That is the same reasoning 20260831_69 (e5-small encoder swap) and 20260912_78 (contextual
chunk headers) already apply for same-width re-embeds; both delete outright with no guard
at all. This keeps a guard, and points it at the one case where a human should decide.

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

    # Only vectors stamped with the CURRENT version are grounds to stop: those would be
    # current-space data, and losing them should be somebody's decision rather than a side
    # effect of running migrations. A stamp naming a superseded model cannot be
    # current-space by construction -- and note this branch only runs when the column
    # width already disagrees with EMBEDDING_DIMENSION, so any vector in here is the wrong
    # width for the configured model no matter what it is stamped.
    current_space = connection.execute(
        sa.text(
            "SELECT count(*) FROM article_chunks "
            "WHERE embedding IS NOT NULL AND embedding_version = :version"
        ),
        {"version": settings.EMBEDDING_VERSION},
    ).scalar()
    if current_space:
        raise RuntimeError(
            f"article_chunks.embedding is vector({current}) but EMBEDDING_DIMENSION is "
            f"{target}, and {current_space} chunk(s) are stamped with the current "
            f"EMBEDDING_VERSION ({settings.EMBEDDING_VERSION}). That combination should be "
            "impossible -- a current-version vector of the wrong width -- so it is not "
            "cleared automatically. Inspect the corpus before re-running."
        )

    # Stale-space vectors, cleared rather than refused on. Unusable twice over: wrong width
    # for the configured model, and excluded from hybrid_search by their stamp. Both tables
    # are rebuilt from articles.content by the sweep this migration re-queues below, so
    # this costs re-indexing time and loses no source content. Children go first:
    # article_chunks.parent_chunk_id references parent_chunks.id, so the reverse order
    # would fail the FK. Same order as 20260831_69 and 20260912_78, for the same reason.
    op.execute("DELETE FROM article_chunks")
    op.execute("DELETE FROM parent_chunks")

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
