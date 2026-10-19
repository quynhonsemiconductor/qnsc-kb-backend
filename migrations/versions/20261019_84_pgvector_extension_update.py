"""pgvector: move the installed extension to the server's default version after a PostgreSQL major upgrade

The develop database moves from PostgreSQL 16 to 18 in place (RDS major upgrade). A major upgrade keeps the
extension at the version it was created with (pgvector 0.8.1 on PG16) even when the new engine ships a newer
default (0.8.2 on PG18.4 and later). ALTER EXTENSION vector UPDATE brings it to the default version.

Guarded: it runs only when the installed version differs from the server's default, so fresh databases and
already-updated ones are no-ops. After the update, an existing HNSW index must be valid; a missing or invalid
index fails the migration instead of letting search run on a broken index.

Downgrade is not provided: an extension version cannot be reverted in place. Restore the pre-upgrade snapshot.
"""

from alembic import op

revision = "20261019_84"
down_revision = "20261018_83"
branch_labels = None
depends_on = None

HNSW_INDEX = "ix_article_chunks_embedding_hnsw"


def upgrade() -> None:
    op.execute("""
        DO $$
        DECLARE
            installed text;
            available text;
        BEGIN
            SELECT extversion INTO installed FROM pg_extension WHERE extname = 'vector';
            SELECT default_version INTO available FROM pg_available_extensions WHERE name = 'vector';
            IF installed IS NOT NULL AND installed IS DISTINCT FROM available THEN
                EXECUTE 'ALTER EXTENSION vector UPDATE';
            END IF;
        END $$;
        """)
    op.execute(f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                WHERE c.relname = '{HNSW_INDEX}' AND NOT i.indisvalid
            ) THEN
                RAISE EXCEPTION 'HNSW index {HNSW_INDEX} is invalid after the vector extension update';
            END IF;
        END $$;
        """)


def downgrade() -> None:
    # No downgrade: an extension version cannot be reverted in place. See the module docstring.
    pass
