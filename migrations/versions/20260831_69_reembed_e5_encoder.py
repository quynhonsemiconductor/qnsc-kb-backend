"""re-embed the corpus after switching the encoder to multilingual-e5-small

The default EMBEDDING_MODEL moved from paraphrase-multilingual-MiniLM-L12-v2 to
intfloat/multilingual-e5-small. Both emit 384-wide vectors, so the pgvector column and
the HNSW index are UNCHANGED -- but the two spaces are unrelated. A MiniLM vector and an
e5 vector of the same width are not comparable; cosine between them is noise.

hybrid_search filters candidates on embedding_version, so the stored chunks (stamped
"minilm-l12-v1") would simply stop matching once EMBEDDING_VERSION becomes "e5-small-v1"
-- the corpus would go invisible rather than wrong. Either way it must be re-embedded.

This migration does the non-negotiable part: it removes the stale-space vectors and puts
every published article back in the queue the API's startup sweep drains, which
re-chunks and re-embeds them with the new model. It does NOT itself run the model (a
migration has no business loading ONNX weights); it hands the work to the sweep.

Idempotent: it keys off the stored stamp, so it no-ops on a database whose chunks are
already e5 (a fresh deploy, or a re-run). It does not touch the column type, because the
width is the same -- 20260828_64 already owns that.

Revision ID: 20260831_69
Revises: 20260830_68
Create Date: 2026-08-31
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings


revision = "20260831_69"
down_revision = "20260830_68"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    target_version = settings.EMBEDDING_VERSION  # "e5-small-v1" at this revision

    # Only chunks whose stamp is not the target need clearing. If none exist, or they are
    # already the target, this is a no-op (fresh deploy or re-run).
    stale = connection.execute(
        sa.text(
            "SELECT count(*) FROM article_chunks WHERE embedding_version <> :target"
        ),
        {"target": target_version},
    ).scalar()
    if not stale:
        return

    # The old-space vectors are unusable and cannot be converted. Remove the chunks so the
    # sweep rebuilds them; parent_chunks/chunk_metadata are rebuilt from the article on
    # re-index, so they go too. Order respects any FK from children to parents.
    op.execute("DELETE FROM article_chunks")
    op.execute("DELETE FROM parent_chunks")

    # Re-queue every published article. Nothing retries a non-pending index on its own, so
    # without this the corpus would stay empty until each article was edited by hand.
    op.execute(
        "UPDATE articles SET index_status = 'pending', index_error = NULL "
        "WHERE status = 'published' AND index_status <> 'pending'"
    )


def downgrade() -> None:
    # Re-embedding is the only meaningful reverse, and the previous vectors are gone. A
    # downgrade would re-queue against whatever EMBEDDING_MODEL is configured at the time,
    # which the upgrade already does, so this is a no-op.
    pass
