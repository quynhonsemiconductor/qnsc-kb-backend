"""Add the tenant knowledge graph: entities, relationships, article mentions

See src/models/graph.py and src/domain/entity_extraction.py / graph_service.py for what
populates these tables and why they are shaped this way (deduplicated nodes/edges per
tenant, provenance kept separately from the edge itself).

Revision ID: 20260911_76
Revises: 20260910_75
Create Date: 2026-09-11
"""

import os

from alembic import op
from sqlalchemy import inspect


revision = "20260911_76"
down_revision = "20260910_75"
branch_labels = None
depends_on = None

_RLS_TABLES = (
    ("graph_entities", "tenant_graph_entities"),
    ("graph_relationships", "tenant_graph_relationships"),
    ("article_entity_mentions", "tenant_article_entity_mentions"),
)


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    existing = set(inspector.get_table_names())

    if "graph_entities" not in existing:
        op.execute(
            """
            CREATE TABLE graph_entities (
                id UUID PRIMARY KEY,
                company_domain VARCHAR(255) NOT NULL,
                name VARCHAR(200) NOT NULL,
                normalized_name VARCHAR(200) NOT NULL,
                entity_type VARCHAR(40) NOT NULL,
                description TEXT,
                mention_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                updated_at TIMESTAMP NOT NULL DEFAULT now(),
                CONSTRAINT uq_graph_entity_company_normalized_type
                    UNIQUE (company_domain, normalized_name, entity_type)
            )
            """
        )
        op.execute("CREATE INDEX ix_graph_entities_company_domain ON graph_entities (company_domain)")
        op.execute(
            "CREATE INDEX ix_graph_entities_company_type ON graph_entities (company_domain, entity_type)"
        )

    if "graph_relationships" not in existing:
        op.execute(
            """
            CREATE TABLE graph_relationships (
                id UUID PRIMARY KEY,
                company_domain VARCHAR(255) NOT NULL,
                source_entity_id UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                target_entity_id UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                relation_type VARCHAR(100) NOT NULL,
                description TEXT,
                article_id UUID REFERENCES articles(id) ON DELETE SET NULL,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                updated_at TIMESTAMP NOT NULL DEFAULT now(),
                CONSTRAINT uq_graph_relationship_edge
                    UNIQUE (company_domain, source_entity_id, target_entity_id, relation_type)
            )
            """
        )
        op.execute(
            "CREATE INDEX ix_graph_relationships_company_domain ON graph_relationships (company_domain)"
        )
        op.execute(
            "CREATE INDEX ix_graph_relationships_company_source "
            "ON graph_relationships (company_domain, source_entity_id)"
        )
        op.execute(
            "CREATE INDEX ix_graph_relationships_company_target "
            "ON graph_relationships (company_domain, target_entity_id)"
        )

    if "article_entity_mentions" not in existing:
        op.execute(
            """
            CREATE TABLE article_entity_mentions (
                id UUID PRIMARY KEY,
                company_domain VARCHAR(255) NOT NULL,
                article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                entity_id UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                CONSTRAINT uq_article_entity_mention UNIQUE (article_id, entity_id)
            )
            """
        )
        op.execute(
            "CREATE INDEX ix_article_entity_mentions_company_domain "
            "ON article_entity_mentions (company_domain)"
        )
        op.execute(
            "CREATE INDEX ix_article_entity_mentions_article_id "
            "ON article_entity_mentions (article_id)"
        )
        op.execute(
            "CREATE INDEX ix_article_entity_mentions_entity_id "
            "ON article_entity_mentions (entity_id)"
        )

    # Same convention as 20260816_56_quality_operations.py: RLS policies are only
    # created when ENABLE_RLS is set at MIGRATION time, and this is the only point
    # that value is read -- verify_rls_policies() at API startup then fails fast in
    # production if a later migrator run forgot it.
    if os.getenv("ENABLE_RLS", "false").lower() in {"1", "true", "yes"}:
        for table, policy in _RLS_TABLES:
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {policy} ON {table} USING "
                "(current_setting('app.global_admin', true) = 'true' OR "
                "company_domain = current_setting('app.company_domain', true)) "
                "WITH CHECK (current_setting('app.global_admin', true) = 'true' OR "
                "company_domain = current_setting('app.company_domain', true))"
            )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    existing = set(inspector.get_table_names())
    for table, policy in _RLS_TABLES:
        if table in existing:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    for table in ("article_entity_mentions", "graph_relationships", "graph_entities"):
        if table in existing:
            op.drop_table(table)
