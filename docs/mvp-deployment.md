# QNSC Knowledge Base MVP deployment

The production compose file runs one public Caddy edge, the React frontend,
FastAPI API, Celery worker and beat, PostgreSQL/pgvector, and Redis. Only Caddy
is exposed publicly; the API, database, and Redis are private to the compose
network.

## First deployment

1. Copy `.env.example` to `.env` and replace every `replace-with-*` value. Set
   `PUBLIC_HOSTNAME` and `CORS_ORIGINS` to the real HTTPS hostname.
2. Point DNS at the VPS and allow inbound TCP 80 and 443. Caddy obtains and
   renews the TLS certificate automatically.
3. Build and start the stack:

   ```powershell
   docker compose -f docker-compose.production.yml up -d --build
   ```

4. Verify:

   ```powershell
   docker compose -f docker-compose.production.yml ps
   curl https://kb.example.com/health/live
   ```

Migrations run from the one-shot `migrator` compose profile (or the ECS
migration task), NOT from the API container. Run it before starting the stack:
`docker compose -f docker-compose.production.yml --profile migrate run --rm migrator`.
`AUTO_CREATE_SCHEMA` is disabled in every environment; schema changes must go
through Alembic.

## Cloud connectors

Configure the Microsoft and Google OAuth redirect URIs to:

```text
https://kb.example.com/api/v1/connectors/oauth/callback
```

The connector admin authorizes a provider, selects drives/sites, maps external
groups to local access groups, and starts the first sync. The worker then polls
selected scopes every ten minutes. Changes are processed by provider cursor,
content is downloaded only for a new revision, and approved documents update
automatically. New documents remain pending until governance approval.

## Operations

- Back up the PostgreSQL volume, connector runtime volume, and private Cloudflare R2 bucket together.
- Inspect connector `last_error` and Celery logs after a failed sync.
- Keep the local BGE-M3 model volume/image on a host sized for inference; use
  object storage for source files when the VPS disk is not durable.
- Provider ACLs are fail-closed until external groups are explicitly mapped to
  local access groups. The mapping is intentionally manual in the MVP.
