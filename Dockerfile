# syntax=docker/dockerfile:1
#
# One image definition, three targets: api, worker, migrator.
#
# It lives at the repo root and takes a target because the shared deploy pipeline
# (QNSC-VN/qnsc-ci .github/workflows/backend-deploy.yml) builds each service by passing
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
#      text from scanned files, so the OCR stack stops at the worker. The embedding
#      runtime is ONNX and reaches BOTH api and worker — EMBEDDING_MODEL is a local model
#      and the api embeds the search query on every search — so both carry the ~2.2 GB
#      fp32 ONNX export, and NEITHER carries torch: the `ml` group exists in pyproject
#      as the reference implementation for re-running the parity gate, but no shipped
#      image installs it. The migrator gets none of it.
#
# Build locally (skip the ~2.2 GB ONNX copy; the export stage still builds and caches,
# since a referenced stage cannot be skipped):
#   docker build --target api  -t qnsc-kb-api . --build-arg BAKE_EMBEDDING_ONNX=false
#   docker build --target worker   -t qnsc-kb-worker .
#   docker build --target migrator -t qnsc-kb-migrator .

# ---------------------------------------------------------------------------
# deps — the runtime dependency set every target shares. build-essential and libpq-dev
# stay here and never reach a shipped image.
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS deps

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
# deps-ml — BUILD-ONLY base for the ONNX export stage. torch + sentence-transformers +
# transformers live here and NEVER reach a shipped image: nothing below copies this
# stage's site-packages. It exists because optimum's exporter needs torch to trace the
# graph, and because the `ml` group remains the reference implementation the parity
# gate compares against when a model change needs re-proving — a build that installs
# this stage is a transitional image, not a deployable one.
#
# `--only main,ml,onnx`, NOT `--only main --with ml`. `--only` is an exhaustive list, so
# mixing `--only` and `--with` silently installs main alone.
# ---------------------------------------------------------------------------
FROM deps AS deps-ml

RUN poetry install --no-root --only main,ml,onnx

# ---------------------------------------------------------------------------
# deps-onnx — the RUNTIME dependency set for every image that embeds: main plus the
# ONNX pair, no torch. This is what the api and worker actually ship.
# ---------------------------------------------------------------------------
FROM deps AS deps-onnx

RUN poetry install --no-root --only main,onnx

# ---------------------------------------------------------------------------
# deps-onnx-ocr — the same again, plus the OCR stack. Worker only.
#
# src/domain/source_extraction.py imports paddle INSIDE the functions that use it, so an
# image without it serves every other path normally and fails loudly only if asked to
# OCR — which the api never is.
# ---------------------------------------------------------------------------
FROM deps-onnx AS deps-onnx-ocr

# The list must name EVERY group the worker needs, not just the one being added:
# `--only main,ocr` would omit onnx, and the worker would fail on the first chunk it
# tried to embed.
RUN poetry install --no-root --only main,ocr,onnx

# ---------------------------------------------------------------------------
# embedding-export — build-only: turns the model into an ONNX export. optimum and its
# exporters never ship, because they are build tools here exactly as build-essential
# is in `deps`, and pinning them in this stage keeps poetry.lock's runtime surface clean.
#
# Based on deps-ml precisely because the exporter needs torch; see that stage for why
# the torch it carries is not a runtime dependency.
#
# FP32 ON PURPOSE, NOT int8. Both dynamic-quantisation recipes were measured against the
# torch backend (2026-08-17, in the built image) and neither is close to the 0.999
# parity gate: per-tensor --avx2 cosine 0.972-0.981, --per_channel 0.981-0.987 with the
# long input collapsing to 0.908. bge-m3's XLM-R stack does not survive weight
# quantisation at retrieval-grade fidelity, and the gate exists to say no exactly here.
# fp32 measures cosine 1.000000 on every parity case, so it ships: still no torch
# (~700 MB less site-packages), one copy of the weights instead of two, and a
# seconds-long session load instead of a 13 s torch load. int8 stays parked until it can
# be re-evaluated against a full corpus re-embed (EMBEDDING_VERSION bump), which is the
# only regime where its error is self-consistent.
FROM deps-ml AS embedding-export

ARG EMBEDDING_MODEL=BAAI/bge-m3
ENV HF_HOME=/tmp/hf-export

# Exact pin on purpose: a floating optimum re-exports a different graph, which gets a
# new layer digest, and ECR bills that as another full copy of the artefact.
#
# optimum-onnx, not optimum[exporters]: the ONNX exporter moved to its own package when
# optimum 2.x split, and the old extra pins model_patcher code that imports
# `_attention_scale` from torch's legacy symbolic registry — present through torch 2.12,
# gone in the 2.13 this image carries (ImportError at CLI startup, verified in a scratch
# container). optimum-onnx 0.1.0 (optimum 2.1.0) imports clean against torch 2.13. It
# resolves transformers 4.x in this build-only stage; the runtime image keeps 5.15 from
# the lock, and the two never meet — the artefact is the only thing that crosses over.
RUN pip install --no-cache-dir 'optimum-onnx==0.1.0' && \
    optimum-cli export onnx --model "${EMBEDDING_MODEL}" \
        --task feature-extraction /opt/embedding-onnx && \
    python -c "import onnx; m = onnx.load('/opt/embedding-onnx/model.onnx'); \
               print('onnx export ok:', len(m.graph.node), 'nodes')" && \
    rm -rf "$HF_HOME"

# ---------------------------------------------------------------------------
# runtime — common base. NO application code: see rule 1 above.
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # /app must be importable, not merely the working directory. Running a script BY
    # PATH (`python scripts/bootstrap_db_role.py`) puts /app/scripts on sys.path — not
    # /app — so `import src.core.config` raises ModuleNotFoundError. uvicorn and celery
    # hide this because they import by module name from the CWD, so the failure appeared
    # only in the migrator, only once deployed.
    PYTHONPATH=/app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/storage/sources /app/storage/connectors && \
    chown -R appuser:appuser /app

# ---------------------------------------------------------------------------
# runtime-ml — the api's base, carrying the ONNX runtime pair and (by default) the fp32
# ONNX export. No torch, no sentence-transformers, no HF cache: those live in deps-ml,
# which is build-only.
#
# This layer is above every `COPY . .` on purpose — see rule 1.
# ---------------------------------------------------------------------------
FROM runtime AS runtime-ml

COPY --from=deps-onnx /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps-onnx /usr/local/bin /usr/local/bin

ARG BAKE_EMBEDDING_ONNX=true

# THE ARTEFACT LAYER MUST BE BYTE-REPRODUCIBLE. It is rebuilt whenever the build cache
# misses — `type=gha` caps at 10 GB per repository — and a rebuilt layer whose bytes
# differ by even one mtime gets a NEW digest, which ECR bills as another full copy.
# That was measured on this repository: nine distinct 2.53 GB layers, every one
# `shared_by=1` — 22.7 GB of 24.0 GB was the same weights over again. cp -a carries the
# export's build-time mtimes in; the find below erases every timestamp under /opt,
# parents included (-depth reaches the directory itself, because creating a directory
# bumps its parent's mtime and /opt alone kept the digest moving once).
#
# Verify after changing anything here — build the target twice with --no-cache and compare
# `docker buildx build --output type=oci`; the manifest's last layer digest must match.
#
# The copy rides on a bind mount from the embedding-export stage rather than a COPY,
# because COPY cannot be conditional on a build-arg: with BAKE_EMBEDDING_ONNX=false the
# artefact stays out of the image (the export stage still builds and caches — Docker
# only skips stages nothing references, and this mount references it).
RUN --mount=type=bind,from=embedding-export,source=/opt/embedding-onnx,target=/mnt/onnx-export \
    mkdir -p /opt/embedding-onnx && \
    if [ "$BAKE_EMBEDDING_ONNX" = "true" ]; then \
        cp -a /mnt/onnx-export/. /opt/embedding-onnx/; \
    fi && \
    find /opt -depth -exec touch -h -d @0 {} + && \
    chown -R appuser:appuser /opt/embedding-onnx

# ---------------------------------------------------------------------------
# runtime-ml-ocr — the worker's base: the above, plus paddle.
# ---------------------------------------------------------------------------
FROM runtime-ml AS runtime-ml-ocr

COPY --from=deps-onnx-ocr /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps-onnx-ocr /usr/local/bin /usr/local/bin

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
