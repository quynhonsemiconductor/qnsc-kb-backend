"""Index the review queue for server-side search and paging.

`GET /governance/pending-drafts` now filters and pages in SQL instead of shipping up to
500 rows for the browser to filter. That moved the cost from the network to the database,
which is only an improvement if the database can answer the query without reading the
whole table -- and `pending_drafts` rows are large (they carry `restructured_body_md` and
`restructure_candidate_md`), so a sequential scan reads the largest columns in the schema
to answer a question about titles.

Three indexes, one per access path the endpoint actually uses:

  * trigram GIN on the accent-folded title and source_ref, for the `search` parameter.
    `immutable_unaccent` is the wrapper migration 58 created for exactly this purpose;
    reusing it (rather than a second folding function) is what keeps the index usable by
    the query -- PostgreSQL matches expression indexes by expression equality, so
    `unaccent(...)` in the query would not hit an `immutable_unaccent(...)` index.

  * a descending composite on (created_at, id), matching the ORDER BY the page query
    issues. The tiebreaker column belongs in the index too, or the sort still needs a
    pass to break ties on identical timestamps.

  * (status, created_at DESC), because the queue is nearly always filtered to one status
    and then ordered -- letting one index serve both halves.

All created CONCURRENTLY: `pending_drafts` is a live table read by the reviewer queue on
every page load, and a plain CREATE INDEX takes an ACCESS EXCLUSIVE lock that blocks
those reads for the duration of the build.
"""

from alembic import op

revision = "20260913_82"
down_revision = "20260912_81"
branch_labels = None
depends_on = None


# CONCURRENTLY cannot run inside a transaction block, and Alembic wraps migrations in one
# by default. autocommit_block() suspends that for these statements only.
_INDEXES = (
    (
        "ix_pending_drafts_title_unaccent_trgm",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pending_drafts_title_unaccent_trgm
        ON pending_drafts USING gin (immutable_unaccent(title) gin_trgm_ops)
        """,
    ),
    (
        "ix_pending_drafts_source_ref_unaccent_trgm",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pending_drafts_source_ref_unaccent_trgm
        ON pending_drafts USING gin (
            immutable_unaccent(coalesce(source_ref, '')) gin_trgm_ops
        )
        """,
    ),
    (
        "ix_pending_drafts_created_at_id_desc",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pending_drafts_created_at_id_desc
        ON pending_drafts (created_at DESC, id DESC)
        """,
    ),
    (
        "ix_pending_drafts_status_created_at",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pending_drafts_status_created_at
        ON pending_drafts (status, created_at DESC)
        """,
    ),
)


def upgrade() -> None:
    # pg_trgm and immutable_unaccent are created by migration 58; assert rather than
    # assume, so a database migrated out of order fails here with a clear cause instead
    # of an opaque "operator class gin_trgm_ops does not exist".
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    with op.get_context().autocommit_block():
        for _, statement in _INDEXES:
            op.execute(statement)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _ in _INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
