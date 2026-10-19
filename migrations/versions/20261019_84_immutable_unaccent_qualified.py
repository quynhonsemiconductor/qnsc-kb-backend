"""Rewrite immutable_unaccent with a schema-qualified unaccent call, on databases that applied revision 58 first

Revision 20260816_58 created immutable_unaccent with an unqualified unaccent call. Editing 58 does not reach a
database that already applied it, and a PostgreSQL 16 -> 18 major upgrade restores the function body on the new
server: the unqualified call then fails with "function unaccent(unknown, text) does not exist". PostgreSQL 17
and later run maintenance (restore, VACUUM, ANALYZE, index builds) with a safe search_path, so the call must
name its schema.

This revision replaces the function body with a schema-qualified call. The schema is read from pg_extension at
run time, not assumed to be public. CREATE OR REPLACE keeps the signature, return type and volatility, so the
index that uses the function stays valid and is not rebuilt. Idempotent: running it again rewrites the same body.
A database where the unaccent extension is missing fails here rather than later in a search.

Downgrade is not provided: restoring the old unqualified body would reintroduce the upgrade failure.
"""

from alembic import op

revision = "20261019_84"
down_revision = "20261018_83"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        DECLARE
            ext_schema text;
        BEGIN
            SELECT n.nspname INTO ext_schema
            FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname = 'unaccent';
            IF ext_schema IS NULL THEN
                RAISE EXCEPTION 'the unaccent extension is not installed';
            END IF;
            EXECUTE format(
                'CREATE OR REPLACE FUNCTION public.immutable_unaccent(value text) '
                'RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE '
                'AS $f$ SELECT %I.unaccent(%L::regdictionary, value) $f$',
                ext_schema,
                ext_schema || '.unaccent'
            );
        END $$;
        """)


def downgrade() -> None:
    # No downgrade: the unqualified body fails on PostgreSQL 17 and later. See the module docstring.
    pass
