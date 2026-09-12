"""re-index the corpus after adding contextual chunk headers

CHUNKING_VERSION moved from "v2-structure-aware" to "v3-contextual-headers"
(src/core/config.py). Chunk boundaries and stored chunk_text did NOT change -- only the
embedding INPUT did: indexing.py now prepends a short, section-level context header
(src/rag/contextual_header.py) before a child chunk is embedded, so a snippet like
"submit the form at least three days in advance" is no longer embedded in isolation from
which document/section it came from.

CHUNKING_VERSION is stamped per chunk but not read back anywhere to trigger anything
automatically -- bumping the setting alone changes what NEW indexing runs produce, not
what is already stored. Existing chunks keep their pre-header vectors, stamped with the
old version, until something re-indexes their article. Left alone, that "something" is
only a future manual edit -- most of the corpus would never benefit.

This does the same thing 20260831_69 did for the e5-small encoder swap: delete the
stale-version chunks and put every published article back in the queue the API's startup
sweep drains, which re-chunks (with headers this time) and re-embeds them. It does NOT
generate headers itself -- a migration has no business making LLM calls -- it hands the
work to the sweep, same as every embedding-model migration before it.

Idempotent: keys off the stored stamp, so it no-ops on a database whose chunks are
already on v3 (a fresh deploy, or a re-run).

Revision ID: 20260912_78
Revises: 20260912_77
Create Date: 2026-09-12
"""

from alembic import op
import sqlalchemy as sa

from src.core.config import settings


revision = "20260912_78"
down_revision = "20260912_77"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    target_version = settings.CHUNKING_VERSION  # "v3-contextual-headers" at this revision

    stale = connection.execute(
        sa.text(
            "SELECT count(*) FROM article_chunks WHERE chunking_version <> :target"
        ),
        {"target": target_version},
    ).scalar()
    if not stale:
        return

    # Chunk boundaries are unchanged, but the corresponding vectors were embedded without
    # a header -- rebuild both tables from scratch via the sweep rather than trying to
    # patch vectors in place. parent_chunks goes too since re-indexing rebuilds it from the
    # article regardless.
    op.execute("DELETE FROM article_chunks")
    op.execute("DELETE FROM parent_chunks")

    # Re-queue every published article. Nothing retries a non-pending index on its own, so
    # without this the corpus would stay empty until each article was edited by hand.
    op.execute(
        "UPDATE articles SET index_status = 'pending', index_error = NULL "
        "WHERE status = 'published' AND index_status <> 'pending'"
    )


def downgrade() -> None:
    # Re-indexing is the only meaningful reverse, and the previous chunks are gone. A
    # downgrade would re-queue against whatever CHUNKING_VERSION is configured at the
    # time, which the upgrade already does, so this is a no-op.
    pass
