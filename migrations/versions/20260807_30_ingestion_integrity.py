"""repair tenant group ownership and reserve source fingerprints

Revision ID: 20260807_30
Revises: 20260807_29
Create Date: 2026-08-07
"""

import os

from alembic import op

revision = "20260807_30"
down_revision = "20260807_29"
branch_labels = None
depends_on = None


def _rls() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() not in {"1", "true", "yes"}:
        return
    op.execute("ALTER TABLE ingestion_fingerprints ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE ingestion_fingerprints FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_ingestion_fingerprints ON ingestion_fingerprints
        USING (
            current_setting('app.global_admin', true) = 'true'
            OR current_setting('app.global_governance_access', true) = 'true'
            OR company_domain = current_setting('app.company_domain', true)
        )
    """)


def upgrade() -> None:
    # Materialize the legacy free-text department values before new writes
    # start requiring an active tenant department.
    op.execute("""
        INSERT INTO departments (id, created_at, updated_at, company_domain, name, active)
        SELECT gen_random_uuid(), now(), now(), source.company_domain, source.name, true
        FROM (
            SELECT DISTINCT ON (company_domain, lower(trim(dept)))
                company_domain, trim(dept) AS name
            FROM (
                SELECT company_domain, dept FROM users WHERE dept IS NOT NULL AND trim(dept) <> ''
                UNION ALL
                SELECT company_domain, dept FROM articles WHERE dept IS NOT NULL AND trim(dept) <> ''
            ) values_to_seed
            ORDER BY company_domain, lower(trim(dept)), trim(dept)
        ) source
        ON CONFLICT (company_domain, name) DO NOTHING
    """)
    # Existing deployments created groups with the historical default
    # company_domain='local'. Infer the correct tenant from memberships before
    # the production RLS policies make those memberships invisible.
    op.execute("""
        UPDATE access_groups AS ag
        SET company_domain = owners.company_domain
        FROM (
            SELECT ug.group_id, min(u.company_domain) AS company_domain
            FROM user_groups ug
            JOIN users u ON u.id = ug.user_id
            GROUP BY ug.group_id
            HAVING count(DISTINCT u.company_domain) = 1
        ) AS owners
        WHERE ag.id = owners.group_id
          AND ag.company_domain = 'local'
    """)
    # External document IDs are only unique within a tenant. The old global
    # constraint allowed one company's connector to block another company's
    # legitimate ID.
    op.execute(
        "ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_external_id_key"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_articles_company_external_id ON articles (company_domain, external_id)"
    )

    # ``AUTO_CREATE_SCHEMA`` is enabled in development, so use IF NOT EXISTS
    # rather than assuming Alembic is the first schema creator.
    op.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_fingerprints (
            id UUID PRIMARY KEY,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL,
            company_domain VARCHAR(255) NOT NULL,
            source_hash VARCHAR(64) NOT NULL,
            status VARCHAR(30) DEFAULT 'pending' NOT NULL,
            draft_id UUID REFERENCES pending_drafts(id) ON DELETE SET NULL,
            article_id UUID REFERENCES articles(id) ON DELETE SET NULL,
            created_by UUID REFERENCES users(id) ON DELETE SET NULL,
            CONSTRAINT uq_ingestion_fingerprint_tenant_hash UNIQUE (company_domain, source_hash)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ingestion_fingerprints_company_domain ON ingestion_fingerprints (company_domain)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ingestion_fingerprints_draft_id ON ingestion_fingerprints (draft_id)"
    )

    # Backfill already-published sources and pending drafts. DISTINCT ON keeps
    # the migration safe even if old deployments already contain duplicates.
    op.execute("""
        INSERT INTO ingestion_fingerprints (id, company_domain, source_hash, status, article_id, created_at, updated_at)
        SELECT gen_random_uuid(), a.company_domain, a.source_hash, 'approved', a.id, now(), now()
        FROM (
            SELECT DISTINCT ON (a.company_domain, ds.source_hash)
                a.company_domain, ds.source_hash, a.id
            FROM document_sources ds
            JOIN articles a ON a.id = ds.article_id
            WHERE a.status <> 'deleted' AND a.lifecycle_status = 'active'
            ORDER BY a.company_domain, ds.source_hash, ds.ingested_at DESC
        ) a
        ON CONFLICT (company_domain, source_hash) DO NOTHING
    """)
    op.execute("""
        INSERT INTO ingestion_fingerprints (id, company_domain, source_hash, status, draft_id, created_by, created_at, updated_at)
        SELECT gen_random_uuid(), pd.company_domain, pd.source_hash, 'pending', pd.id, pd.created_by, now(), now()
        FROM (
            SELECT DISTINCT ON (company_domain, source_hash)
                id, company_domain, source_hash, created_by
            FROM pending_drafts
            WHERE status = 'pending'
            ORDER BY company_domain, source_hash, created_at DESC
        ) pd
        ON CONFLICT (company_domain, source_hash) DO NOTHING
    """)
    _rls()


def downgrade() -> None:
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        op.execute(
            "DROP POLICY IF EXISTS tenant_ingestion_fingerprints ON ingestion_fingerprints"
        )
    op.execute("DROP INDEX IF EXISTS uq_articles_company_external_id")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS articles_external_id_key ON articles (external_id)"
    )
    op.execute("DROP TABLE IF EXISTS ingestion_fingerprints")
