"""Align the keyword search query with indexed accent-folded text."""

from alembic import op

revision = "20260816_58"
down_revision = "20260816_57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # unaccent(text) is STABLE, so PostgreSQL will not allow it directly in
    # an expression index. This immutable wrapper pins the extension dictionary
    # used by the application search contract. The call is schema-qualified:
    # PostgreSQL 17 and later run maintenance (restore, VACUUM, ANALYZE, index
    # builds) with a safe search_path, so an unqualified unaccent in this body
    # does not resolve there. Databases that applied this revision before the
    # qualification are rewritten by 20261019_84.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("""
        CREATE OR REPLACE FUNCTION immutable_unaccent(value text)
        RETURNS text
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, value) $$
    """)
    op.execute("DROP INDEX IF EXISTS ix_article_chunks_fts")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_article_chunks_fts
        ON article_chunks USING gin (to_tsvector('simple', immutable_unaccent(chunk_text)))
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_articles_title_unaccent_trgm
        ON articles USING gin (immutable_unaccent(title) gin_trgm_ops)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_articles_title_unaccent_trgm")
    op.execute("DROP INDEX IF EXISTS ix_article_chunks_fts")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_article_chunks_fts
        ON article_chunks USING gin (to_tsvector('simple', chunk_text))
    """)
