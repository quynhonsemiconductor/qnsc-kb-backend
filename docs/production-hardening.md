# Production hardening

## Required environment

Production must set `ENVIRONMENT=production`, a random `SECRET_KEY` of at least 32 characters, a separate `DATA_ENCRYPTION_KEY`, `AUTO_CREATE_SCHEMA=false`, `ALLOW_SELF_REGISTRATION=false`, explicit `CORS_ORIGINS`, PostgreSQL admin and application-role credentials, Redis credentials, a specific Microsoft Entra tenant GUID/client/HTTPS redirect configuration, and the required LLM credentials. Schema changes are applied only by Alembic; do not use the development `docker-compose.yml` in production.

## Deployment

Use `docker-compose.production.yml`. Migrations run from the separate `migrator` profile service (`docker compose -f docker-compose.production.yml --profile migrate run --rm migrator`) BEFORE the API starts; the API container itself does not migrate. The worker runs Celery with Redis. The production compose file does not publish database or Redis ports. Put TLS and authentication at the reverse proxy or managed ingress layer.

## Migrations

Run migrations once per deployment from the `migrator` compose profile (or the ECS migration task) using `MIGRATION_DATABASE_URL`. Runtime schema creation is disabled in production. The API and worker use the non-superuser `DATABASE_URL`; this is required for RLS to be effective. Validate migrations against a disposable PostgreSQL/pgvector instance before release.

## Retrieval and indexing

Published article changes are dispatched through Celery when `JOB_MODE=celery`. Embeddings are generated in batches and use `EMBEDDING_VERSION` for reproducibility. Any embedding failure marks indexing as failed and records a dead-letter job; zero vectors are only available in explicit mock mode.

The performance migration creates the HNSW, full-text, permission-filter, and operational indexes. Measure with `EXPLAIN ANALYZE` after loading representative data.

## Backups and recovery

Back up PostgreSQL and source storage together. Store encrypted copies off-host, define retention, and perform a scheduled restore into an isolated environment. A successful `pg_dump` alone is not a complete knowledge-base backup because originals are stored separately.

For production, set `SOURCE_STORAGE_BACKEND=r2`, `SOURCE_STORAGE_BUCKET`, `R2_ACCOUNT_ID` (or `S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com`), and the required R2 credentials. Generic S3 endpoints and ambient AWS credentials are rejected; local-disk source storage is disabled. Enable `MALWARE_SCAN_ENABLED` only when the configured scanner command is installed and monitored.

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable distributed traces. Without it, the application remains operational with structured logs and the `/metrics` endpoint but does not emit external spans.

## Release checks

Run:

```text
pytest -q
python -m compileall -q src migrations
npm run build
```

Also run migration-up/migration-down checks, a permission endpoint matrix, retrieval golden-set evaluation, upload malware/zip-bomb tests, and a load test before a production release.

For a running production instance, use:

```text
python scripts/load_test.py --base-url https://kb.example.com --requests 1000 --concurrency 25
python scripts/security_smoke.py --base-url https://kb.example.com
```
