# QNSC Knowledge Base Backend

FastAPI-based backend application for QNSC Knowledge Base, utilizing pgvector for hybrid RAG search, Celery for asynchronous processing, and Valkey (Redis protocol) as the Celery broker and result backend, and for the shared login rate limiter. The engine is the one production runs on ElastiCache; production's Valkey node is currently disabled.

## Run it locally (Docker, from a clean clone)

You need Docker with about 10 GB of memory available to it (the API and the worker each hold
the ~1.5 GiB embedding model; their limits are 4 GB each) and about 6 GB of free disk. Nothing
else: no Python, no Postgres, no Redis, no API keys.

```bash
git clone https://github.com/quynhonsemiconductor/qnsc-kb-backend.git
cd qnsc-kb-backend
cp .env.example .env                                   # the only file you copy; it works as-is

docker compose up -d --build db redis garage garage-init       # .env.example sets KB_BAKE_EMBEDDING_MODEL=false: no model download
docker compose --profile migrate run --rm migrator     # creates the schema
docker compose up -d --build api worker beat           # API + Celery worker + Celery beat
```

Wait until it reports ready (the API loads the embedding model in the background, ~20 s):

```bash
curl -s http://localhost:8000/health/ready
# {"status":"ready","database":"ok","redis":"ok","job_mode":"celery","embedding":{"state":"ready",...},"degraded":false}
docker compose ps        # api healthy; worker healthy; beat running
```

If a port is already taken (a second project's Postgres on 5432 is the usual clash), set
`KB_DB_PORT`, `KB_REDIS_PORT` or `KB_API_PORT` in `.env` (see the top of `.env.example`) and run the
same commands. On Apple Silicon this works as-is: the worker runs from the API image. Set
`KB_WORKER_TARGET=worker` only if you need OCR of scanned files (x86_64 only).

### Object storage (local)

Uploads go to a local Garage node (an S3-compatible store, the stand-in for the R2 bucket), on `STORAGE_HOST_PORT`
(default 4570, loopback only). `garage-init` creates the bucket, imports a fixed local key, and allows the frontend
origins (`FRONTEND_DEV_PORT`, both `localhost` and `127.0.0.1`) before the API starts; it fails loudly if any step does not
take. The data lives in the `garagedata` and `garagemeta` volumes, so uploads survive `docker compose restart` and
`docker compose down`. `docker compose down -v` removes them with the database.

### Isolated test run (e2e)

The test suite points at a database it writes to, so it runs on its own stack, never on a dev stack. `scripts/e2e/run.sh` starts
Postgres and Valkey under a fresh compose project with **ephemeral loopback ports** (Docker picks them; no fixed host port is used),
creates CI's test database, migrates, runs `pytest` with CI's environment, and removes the stack and its volumes on exit.

The runner refuses to start while a `.env` exists in the repository (the settings would read it, and CI has none). Run it from a clean
clone, or move `.env` aside for the run.

```bash
poetry install                                       # once
scripts/e2e/run.sh                                   # the whole suite
scripts/e2e/run.sh tests/unit/test_local_compose.py  # or the files you name
E2E_PROJECT=qnsc-kb-e2e-alice scripts/e2e/run.sh     # a name for this run (default: random)
E2E_KEEP=1 scripts/e2e/run.sh                        # keep the stack up; the run prints how to remove it
```

### Run a second copy

To run a second copy beside this one (a second checkout, or a branch you keep running), set in `.env`: `COMPOSE_PROJECT_NAME=qnsc-kb-2`
(it prefixes the containers and volumes), and free values for `KB_DB_PORT`, `KB_REDIS_PORT`, `KB_API_PORT` and `STORAGE_HOST_PORT`.
Set the second frontend's `FRONTEND_DEV_PORT` here too: the bucket's CORS is built from it at start, so an unlisted origin fails at upload. The containers reach each other
by service name, so nothing else changes. Then run the same commands. The frontend's dev server still needs its own port (see that repository).

The second copy's frontend must also be allowed by this backend: in its `.env`, set `CORS_ORIGINS` and `FRONTEND_URL` to the second
frontend origin, e.g. `http://127.0.0.1:5174` (the frontend README explains why 127.0.0.1). Cookies are per host, not per port, so
`localhost:5173` and `localhost:5174` share one cookie jar: open the second frontend as `127.0.0.1`, or use another browser profile.

### Sign in

There is no signup route. On first start the API creates one administrator from `.env`:

| | |
|---|---|
| Email | `admin@qnsc.vn` (`BOOTSTRAP_ADMIN_EMAIL`) |
| Password | `Admin123@` (`BOOTSTRAP_ADMIN_PASSWORD`, a published development default; production refuses it) |

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d 'username=admin@qnsc.vn&password=Admin123@'        # -> {"access_token":"...", ...}
curl -s -H "Authorization: Bearer <access_token>" http://localhost:8000/api/v1/auth/me
```

The web UI is the sibling repository, `qnsc-kb-frontend` (`npm ci && npm run dev`, it proxies
`/api` to this API on port 8000). Interactive API docs: <http://localhost:8000/docs>
(`ENABLE_API_DOCS=true` in `.env`).

Permission test users (Admin, CEO, Reviewer, Staff), only for a development or UAT database:

```bash
docker compose exec -e SEED_TEST_PASSWORD='choose-a-password' api python scripts/seed_data.py
```

They are created in the `SEED_COMPANY_DOMAIN` tenant (`acme.test`), e.g. `kb-admin@acme.test`.
The password is never stored in the repository.

### What works without configuration, and what needs more

| Feature | Local status |
|---|---|
| Sign-in, articles, drafts, review, search, Celery jobs, beat schedules | Works with the example env |
| Upload, ingest and search of sources | Works: Garage stands in for R2 (`SOURCE_STORAGE_BACKEND=r2`, `S3_ENDPOINT_URL` set to the Garage node; the browser uploads to `S3_PUBLIC_ENDPOINT_URL` through a presigned URL). `garage-init` creates the bucket and CORS for the local web origins before the API starts. Uploads persist in the `garagedata` volume across restarts |
| Embeddings | Mock by default (`EMBEDDING_MODEL=mock`, `EMBEDDING_VERSION=mock-v1`): deterministic, no model download, keyword-level search quality. Real local embeddings: `KB_BAKE_EMBEDDING_MODEL=true` with the e5 model in `.env` (about 6 GB disk, 8 GB Docker memory for the bake) |
| AI answers and AI formatting | Mock by default with `LLM_PROVIDER=mock` (development only): deterministic answers that cite the first authorized passage, no key. For a real model, an administrator saves a provider and key in Admin > AI provider. A key in `.env` is not read for answers |
| Malware scanning (ClamAV) | Off locally (`MALWARE_SCAN_ENABLED=false`); deployed environments run it |
| Microsoft / Google connectors, SSO, email | Off until their credentials are set in `.env`. Microsoft sign-in and the connector also need `MICROSOFT_TENANT_ID` set to your Entra **tenant GUID** (never `common`; blank or `common` leaves both off, by design). Use the **KB Dev** app registration: `docs/runbooks/entra-apps.md` |

The local-only switches are refused outside development (`validate_production`): `LLM_PROVIDER=mock`, `EMBEDDING_MODEL=mock`,
`S3_PUBLIC_ENDPOINT_URL`. The local S3 endpoint is accepted only in development.

Secrets in `.env.example` (`SECRET_KEY`, `DATA_ENCRYPTION_KEY`) are throwaway placeholders that
work in development; generate real ones with `openssl rand -hex 32` for anything other people can
reach. `.env` is gitignored; never copy a deployed environment's values into it.

### Stop, reset, logs

```bash
docker compose logs -f api worker beat
docker compose down            # keep the data
docker compose down -v         # also drop the database (needed once if you had an older pg15 volume)
```

## Without Docker, and tests

Running the API directly on the host (Poetry, your own Postgres with pgvector and Redis) is not
part of this quickstart and has not been verified; `.env.example` explains which URLs it expects.
The commands CI runs for lint, unit and integration tests are in `.github/workflows/backend-ci.yml`.
