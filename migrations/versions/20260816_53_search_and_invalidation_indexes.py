"""add event-driven cache invalidation and hot foreign-key indexes

Cache invalidation runs on every article lifecycle event with a containment
predicate over ai_cache.article_ids cast to JSONB; without a GIN expression
index that is a full table scan per publish/update. The remaining indexes
cover hot query paths (comment threads, per-user usage/search logs, source
lookup by article, outbox drain) that previously sequential-scanned.
"""
from alembic import op

revision = "20260816_53"
down_revision = "20260814_52"
branch_labels = None
depends_on = None

_CONCURRENT_INDEXES = (
    # Expression index must match the invalidation predicate's cast exactly.
    "CREATE INDEX IF NOT EXISTS ix_ai_cache_article_ids_gin ON ai_cache USING gin (((article_ids)::jsonb) jsonb_path_ops)",
    "CREATE INDEX IF NOT EXISTS ix_comments_article_id ON comments (article_id)",
    "CREATE INDEX IF NOT EXISTS ix_comments_user_id ON comments (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_ai_usage_logs_user_id ON ai_usage_logs (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_search_logs_user_id ON search_logs (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_document_sources_article_id ON document_sources (article_id)",
    "CREATE INDEX IF NOT EXISTS ix_outbox_events_drain ON outbox_events (status, next_attempt_at)",
)

_DOWNGRADE_DROPS = (
    "ix_outbox_events_drain",
    "ix_document_sources_article_id",
    "ix_search_logs_user_id",
    "ix_ai_usage_logs_user_id",
    "ix_comments_user_id",
    "ix_comments_article_id",
    "ix_ai_cache_article_ids_gin",
)


def upgrade() -> None:
    # These tables grow without bound; avoid blocking writes during creation.
    with op.get_context().autocommit_block():
        for statement in _CONCURRENT_INDEXES:
            op.execute(statement)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in _DOWNGRADE_DROPS:
            op.execute(f"DROP INDEX IF EXISTS {name}")
