# syntax=docker/dockerfile:1
#
# One image definition, three targets: api, worker, migrator.
#
# It lives at the repo root and takes a target because the shared deploy pipeline
# (quynhonsemiconductor/ci .github/workflows/backend-deploy.yml) builds each service by passing
# `build-target` against ONE Dockerfile and has no per-service dockerfile input.
#
# TWO RULES SHAPE THIS FILE, both learned the expensive way:
#
#   1. Application code is copied LAST, per target. Docker invalidates every layer above
#      a changed one, so anything large must sit below the thing that changes on every
#      commit. `COPY . .` used to sit under the model bake, so each commit stored a fresh
#      2.3 GB layer — 81.6 GB of ECR across 21 builds.
#
#   2. Each target carries only what it runs. paddle is ~1 GB and only the worker extracts
#      text from scanned files, so the OCR stack stops at the worker. The embedding stack
#      (ONNX Runtime + weights) reaches BOTH api and worker, because EMBEDDING_MODEL is a local
#      model and the api embeds the search query on every search — that is the deliberate
#      cost of not sending text to a hosted embedder. The migrator gets neither.
#
# Build:
#   docker build --target api      -t qnsc-kb-api .
#   docker build --target worker   -t qnsc-kb-worker .
#   docker build --target migrator -t qnsc-kb-migrator .
#
# The embedding export is baked by default and SHOULD be: the ONNX loader reads it from
# disk and has no runtime download path, so an unbaked image cannot embed at all — it
# serves /health and then falls back to keyword-only search on every query. Pass
# BAKE_EMBEDDING_MODEL=false only where that is intended, i.e. CI proving the image
# builds. The arg used to be documented as BAKE_EMBEDDING_ONNX, which this file never
# declared, so the flag silently did nothing wherever it was passed.

# ---------------------------------------------------------------------------
# deps — the runtime dependency set every target shares. build-essential and libpq-dev
# stay here and never reach a shipped image.
#
# 3.13, NOT 3.14 — and this is a ceiling, not a lag. paddlepaddle 3.3.1 publishes
# cp39 through cp313 and no cp314, so the worker target (the only one carrying OCR)
# cannot resolve its dependency set on 3.14 at all: `poetry install --only main,ml,ocr`
# fails outright. The bump to 3.14-slim went in anyway and backend-ci has been failing
# on main since. Raise this only once paddlepaddle ships a cp314 wheel.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --only main

# ---------------------------------------------------------------------------
# deps-ml — the same, plus ONNX Runtime and Transformers. api and worker.
#
# Not optional in practice: EMBEDDING_MODEL defaults to intfloat/multilingual-e5-small
# (src/core/config.py), and src/lib/embeddings/__init__.py loads it in-process. Without
# this group the api answers /health and then raises on the first search, because the
# failure is a lazy import inside the model singleton rather than anything visible at
# startup.
# ---------------------------------------------------------------------------
FROM deps AS deps-ml

# `--only main,ml`, NOT `--only main --with ml`. `--only` is an exhaustive list, so
# combining the two silently installs main alone.
RUN poetry install --no-root --only main,ml

# ---------------------------------------------------------------------------
# deps-ml-ocr — the same again, plus the OCR stack. Worker only.
#
# src/domain/source_extraction.py imports paddle INSIDE the functions that use it, so an
# image without it serves every other path normally and fails loudly only if asked to
# OCR — which the api never is.
# ---------------------------------------------------------------------------
FROM deps-ml AS deps-ml-ocr

# Same rule as above, and the list must name EVERY group the worker needs, not just the
# one being added: `--only main,ocr` here resolves without ml, so the worker would ship
# paddle and no ONNX embedding runtime and fail on the first chunk it tried to embed.
RUN poetry install --no-root --only main,ml,ocr

# ---------------------------------------------------------------------------
# runtime — common base. NO application code: see rule 1 above.
# ---------------------------------------------------------------------------
# Kept in step with the deps stage above, including its 3.13 ceiling.
FROM python:3.13-slim AS runtime

# Debian security updates for the base image's own packages, applied in the stage every
# scanned stage derives from (api, worker, migrator).
#
# Trivy's CRITICAL gate blocks on base-image CVEs, not just ours: run 34751584705 failed
# on three perl-base advisories (CVE-2026-13221, CVE-2026-8376, CVE-2026-42496) fixed in
# 5.40.1-6+deb13u1, with zero findings in our Python dependencies. python:3.13-slim ships
# whatever Debian had at ITS build time, so this recurs whenever an advisory lands before
# the upstream tag is rebuilt — a base image bump cannot be the fix.
#
# `upgrade`, not `dist-upgrade`: security patches within the pinned Debian release only,
# never a release upgrade that could swap out the interpreter this stage is built around.
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # /app must be importable, not merely the working directory. Running a script BY
    # PATH (`python scripts/bootstrap_db_role.py`) puts /app/scripts on sys.path — not
    # /app — so `import src.core.config` raises ModuleNotFoundError. uvicorn and celery
    # hide this because they import by module name from the CWD, so the failure appeared
    # only in the migrator, only once deployed.
    PYTHONPATH=/app

# Copied as the whole lib tree, not as /usr/local/lib/python3.NN/site-packages.
# The interpreter version lives in the FROM tag, which dependabot bumps on its own;
# a hardcoded path silently stops matching it. That is exactly what happened when the
# base moved to 3.14 while these lines still said 3.11 — the build died on
# "/usr/local/lib/python3.11/site-packages: not found" and main has been red since.
# Both stages derive from the same base image, so this overlays like-for-like.
COPY --from=deps /usr/local/lib/ /usr/local/lib/
COPY --from=deps /usr/local/bin /usr/local/bin

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/storage/sources /app/storage/connectors && \
    chown -R appuser:appuser /app

# ---------------------------------------------------------------------------
# runtime-ml — the api's base, carrying ONNX Runtime, Transformers and (by default) the
# model weights themselves.
#
# BAKING THE ONNX ASSETS IS THE POINT. The runtime downloads them on first use, so an
# unbaked image pays ~2.3 GB and several minutes on the first search AFTER the task is
# already serving traffic — repeatedly, on every replacement. Baked, it is paid once at
# build time. Local builds pass BAKE_EMBEDDING_MODEL=false and use the developer's own
# cache instead of storing another copy per rebuild.
#
# HF_HOME is set for BOTH build and run so the two agree on where the weights are; a
# mismatch silently re-downloads at runtime and looks like the bake never happened. It
# sits under /opt rather than the home directory because the deploy may run this as a
# different uid.
#
# This layer is above every `COPY . .` on purpose — see rule 1.
# ---------------------------------------------------------------------------
FROM runtime AS runtime-ml

# Copied as the whole lib tree, not as /usr/local/lib/python3.NN/site-packages.
# The interpreter version lives in the FROM tag, which dependabot bumps on its own;
# a hardcoded path silently stops matching it. That is exactly what happened when the
# base moved to 3.14 while these lines still said 3.11 — the build died on
# "/usr/local/lib/python3.11/site-packages: not found" and main has been red since.
# Both stages derive from the same base image, so this overlays like-for-like.
COPY --from=deps-ml /usr/local/lib/ /usr/local/lib/
COPY --from=deps-ml /usr/local/bin /usr/local/bin

ARG BAKE_EMBEDDING_MODEL=true
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
ENV HF_HOME=/opt/huggingface
ENV EMBEDDING_ONNX_DIR=/opt/embedding-onnx

# Materialise the export where the loader actually looks. This used to snapshot `onnx/*`
# into the HF cache, but src/lib/embeddings/local_onnx.py reads EMBEDDING_ONNX_DIR and
# wants exactly two files, model.onnx and tokenizer.json, so nothing ever bridged the
# two and /opt/embedding-onnx did not exist in any image. That is why every search
# logged "Error generating local BGE embedding; continuing with keyword search".
#
# The model publishes its own ONNX export, so no optimum-cli step and no torch is needed
# anywhere — the build downloads two (sometimes three, see below) files.
#
# fp32 (`onnx/model.onnx`), NOT one of the qint8 variants, even though those are smaller
# and faster on CPU. Quantisation moves the vectors: this repo's own parity gate measured
# int8 at cosine 0.972-0.987 against the reference, and a query embedded slightly off the
# space its documents were embedded in degrades retrieval silently. Consistency wins.
#
# `onnx/model.onnx_data`: fetched ONLY IF `list_repo_files` says the repo actually has
# it. A model whose fp32 ONNX graph exceeds protobuf's 2GB limit — true for
# multilingual-e5-large-instruct, NOT true for the smaller e5-small this previously baked
# — splits its weights into this sibling file, which onnxruntime loads automatically as
# long as it sits next to model.onnx under the same name. Missing it is not a load-time
# error you would immediately connect to this cause: onnxruntime fails deep inside
# InferenceSession() with a path error naming the missing external-data file, not
# anything mentioning the Dockerfile or the bake step.
#
# NOTE for anyone editing this RUN line: `python -c` collapses every backslash-continued
# line here into ONE logical line before Python ever parses it, so only constructs valid
# as a single line survive -- a list comprehension (as both statements below use, the
# second with an `if` FILTER clause standing in for a conditional) works; `try/except` or
# an `if:` block does not, because Python's grammar will not let a compound statement
# follow a `;`-separated simple statement on the same logical line. An earlier version of
# this line used try/except and failed the build with a SyntaxError only once
# BAKE_EMBEDDING_MODEL actually ran a model through it.
# Two speed levers on top of the download itself, neither of which touches WHAT gets
# downloaded or WHEN this layer reruns (that is a build-cache question -- see this repo's
# CI, which sidesteps the whole bake with BAKE_EMBEDDING_MODEL=false and caches everything
# else via `type=gha`; the deploy pipeline's own cache policy is decided in the shared
# `quynhonsemiconductor/ci` reusable workflow, not here):
#
# - `HF_XET_HIGH_PERFORMANCE=1`: huggingface_hub 1.x already transfers through Xet (the
#   `hf_xet` package below is pulled in as its own dependency, no install step needed) --
#   this flag turns on Xet's more aggressive concurrency/chunking rather than switching on
#   acceleration that was otherwise off. `HF_HUB_ENABLE_HF_TRANSFER` -- the older lever,
#   worth knowing about only so nobody re-adds it -- is a no-op on this version: it warns
#   "hf_transfer is not used anymore" and does nothing, because Xet replaced it.
#
# - `HF_TOKEN` via a build secret, not an ARG or ENV: a secret mount exists only for this
#   RUN's own execution and is never written to a layer or `docker history`; an ARG is
#   baked into image metadata forever. `required=false` so a build with nothing passed
#   (every local build, and CI's BAKE_EMBEDDING_MODEL=false build) still succeeds exactly
#   as before -- anonymous download of a public model.
#
#   Measured in this repo (a real build, not a claim from documentation): anonymous, no
#   HF_TOKEN, no HF_XET_HIGH_PERFORMANCE — 61s for the 2.2 GB export. The Hub's own
#   download warning states unauthenticated requests get lower rate limits AND SLOWER
#   TRANSFER under Xet specifically (unlike the old CDN-blob path, where a public repo's
#   bytes were not gated behind auth at all) -- so unlike a stale assumption carried over
#   from that older mechanism, a token is worth passing here for real, not just as
#   rate-limit insurance. To use it: `docker build --secret id=hf_token,env=HF_TOKEN ...`
#   locally, or a `secrets:` entry on whatever `docker/build-push-action` step actually
#   builds the deploy image (that step lives in `quynhonsemiconductor/ci`, not this repo).
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p "$HF_HOME" "$EMBEDDING_ONNX_DIR" && \
    if [ "$BAKE_EMBEDDING_MODEL" = "true" ]; then \
        export HF_XET_HIGH_PERFORMANCE=1 && \
        if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
        python -c "\
from huggingface_hub import hf_hub_download, list_repo_files; \
import shutil, os; \
target = os.environ['EMBEDDING_ONNX_DIR']; \
[shutil.copyfile(hf_hub_download('${EMBEDDING_MODEL}', name), os.path.join(target, os.path.basename(name))) \
 for name in ('onnx/model.onnx', 'tokenizer.json')]; \
[shutil.copyfile(hf_hub_download('${EMBEDDING_MODEL}', 'onnx/model.onnx_data'), os.path.join(target, 'model.onnx_data')) \
 for _ in [0] if 'onnx/model.onnx_data' in list_repo_files('${EMBEDDING_MODEL}')]"; \
    fi && \
    chown -R appuser:appuser "$HF_HOME" "$EMBEDDING_ONNX_DIR"

# ---------------------------------------------------------------------------
# runtime-ml-ocr — the worker's base: the above, plus paddle.
# ---------------------------------------------------------------------------
FROM runtime-ml AS runtime-ml-ocr

# Copied as the whole lib tree, not as /usr/local/lib/python3.NN/site-packages.
# The interpreter version lives in the FROM tag, which dependabot bumps on its own;
# a hardcoded path silently stops matching it. That is exactly what happened when the
# base moved to 3.14 while these lines still said 3.11 — the build died on
# "/usr/local/lib/python3.11/site-packages: not found" and main has been red since.
# Both stages derive from the same base image, so this overlays like-for-like.
COPY --from=deps-ml-ocr /usr/local/lib/ /usr/local/lib/
COPY --from=deps-ml-ocr /usr/local/bin /usr/local/bin

# paddle links against the GNU OpenMP runtime, and paddleocr's opencv against the X11
# and GL client libraries. None of them are Python packages, so copying site-packages
# out of the deps stage does not bring them: the deps stage got them from
# build-essential, which deliberately never reaches a shipped image.
#
# The result was an image that BUILT and then raised
# `ImportError: libgomp.so.1: cannot open shared object file` the first time anything
# imported paddle. src/domain/source_extraction.py imports it lazily, so the worker
# started clean and only failed on the first scanned document — in production, where
# a build-only CI job could never have seen it.
RUN apt-get update && apt-get install -y --no-install-recommends     libgomp1     libglib2.0-0     libgl1     && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# api — FastAPI under uvicorn.
#
# Carries the embedding stack but NOT OCR: main.py preloads the model at startup so no
# request pays for loading it, and the api never extracts text from a scanned file.
#
# Deliberately NO entrypoint running Alembic. That is right for a single-VPS compose and
# wrong for ECS, where every task would run it — a deploy or a scale-out firing N
# concurrent migrations against one database. Migrations belong to the `migrator` target,
# which the pipeline runs once, before rolling any service.
# ---------------------------------------------------------------------------
FROM runtime-ml AS api

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Liveness only — /health/live answers 200 without touching a dependency. Never point
# this at /health/ready: a dependency-coupled probe here restarts the task whenever the
# database or cache blips, turning a hiccup into an outage. Readiness is checked once,
# after the roll, by the deploy pipeline.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# worker — Celery worker, and the image `celery beat` runs from with its own command.
# Beat is a singleton and must never be scaled past one replica.
# ---------------------------------------------------------------------------
FROM runtime-ml-ocr AS worker

COPY --chown=appuser:appuser . .

USER appuser

CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info", \
     "-Q", "celery,ingestion,connectors,permissions"]

# ---------------------------------------------------------------------------
# migrator — one-shot: ensure the least-privilege app role exists, then migrate.
#
# Built from `runtime`: migrations/env.py imports src.models for target_metadata and
# nothing that touches embeddings or OCR.
# ---------------------------------------------------------------------------
FROM runtime AS migrator

COPY --chown=appuser:appuser . .

USER appuser

ENTRYPOINT ["/app/docker/migrate-entrypoint.sh"]
CMD ["alembic", "-c", "migrations/alembic.ini", "upgrade", "head"]
