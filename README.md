# QNSC Knowledge Base Backend

FastAPI-based backend application for QNSC Knowledge Base, utilizing pgvector for hybrid RAG search, Celery for asynchronous processing, and Redis for caching.

## Run it locally (Docker, from a clean clone)

You need Docker with about 10 GB of memory available to it (the API and the worker each hold
the ~1.5 GiB embedding model; their limits are 4 GB each) and about 6 GB of free disk. Nothing
else: no Python, no Postgres, no Redis, no API keys.

```bash
git clone https://github.com/quynhonsemiconductor/qnsc-kb-backend.git
cd qnsc-kb-backend
cp .env.example .env                                   # the only file you copy; it works as-is

docker compose up -d --build db redis                  # first build downloads the model (~2.3 GB, several minutes)
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
| Sign-in, articles, drafts, review, search (local embeddings), Celery jobs, beat schedules | Works with the example env |
| AI answers and AI formatting | Need a chat provider. It is NOT an environment variable: an administrator saves provider, model and key in the app (Admin > AI provider). Until then the workspace answers from the retrieval fallback. To use a model on your machine (Ollama, vLLM, LM Studio) set `LLM_ALLOW_LOCAL_ENDPOINTS=true` in `.env` |
| File upload | Needs your OWN Cloudflare R2 bucket and keys in `.env` (`SOURCE_STORAGE_BUCKET`, `R2_*`). There is no local-disk or MinIO backend, by design |
| Malware scanning (ClamAV) | Off locally (`MALWARE_SCAN_ENABLED=false`); deployed environments run it |
| Microsoft / Google connectors, SSO, email | Off until their credentials are set in `.env`. Microsoft sign-in and the connector also need `MICROSOFT_TENANT_ID` set to your Entra **tenant GUID** (never `common`; blank or `common` leaves both off, by design). Use the **KB Dev** app registration: `docs/runbooks/entra-apps.md` |

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
