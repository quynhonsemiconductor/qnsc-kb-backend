import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from src.core.config import settings

from alembic import context
from src.models.base import Base

config = context.config
config.set_main_option("sqlalchemy.url", (settings.MIGRATION_DATABASE_URL or settings.DATABASE_URL).replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# A pre-Alembic deployment left this empty table behind. It is intentionally
# retained for backward-compatible database handoff, but it is no longer part
# of the application model and must not make every ``alembic check`` propose a
# destructive drop. The connector scheduling column is retained in the ORM
# model until an explicit data-retention decision is made.
_LEGACY_TABLES = {"connector_credentials"}

# These indexes are created by historical SQL migrations rather than ORM
# declarations. They remain required for retrieval/audit performance and are
# therefore excluded from metadata-diff removal proposals.
_MIGRATION_MANAGED_INDEXES = {
    "ix_ai_cache_expiry",
    "ix_ai_cache_article_ids_gin",
    "ix_ai_usage_logs_user_id",
    "ix_article_access_group_id",
    "ix_article_chunks_article_id",
    "ix_article_chunks_embedding_hnsw",
    "ix_article_chunks_fts",
    "ix_article_chunks_permission_lookup",
    "ix_article_departments_department_id",
    "ix_article_versions_article_id",
    "ix_articles_title_unaccent_trgm",
    "ix_audit_logs_created_at",
    "ix_comments_article_id",
    "ix_comments_user_id",
    "ix_document_sources_article_id",
    "ix_document_sources_hash",
    "ix_ingestion_fingerprints_draft_id",
    "ix_outbox_events_drain",
    "ix_search_logs_user_id",
    "ix_user_departments_department_id",
    "uq_department_managers_one_active_owner",
}


def include_object(object_, name, type_, reflected, compare_to):
    if reflected and type_ == "table" and name in _LEGACY_TABLES:
        return False
    if reflected and type_ == "index" and name in _MIGRATION_MANAGED_INDEXES:
        return False
    return True

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, include_object=include_object)

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
