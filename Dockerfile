# syntax=docker/dockerfile:1
#
# One image definition, three targets: api, worker, migrator.
#
# It lives at the repo root and takes a target because the shared deploy pipeline
# (quynhonsemiconductor/ci .github/workflows/backend-deploy.yml) builds each service by passing
# `build-target` against ONE Dockerfile and has no per-service dockerfile input.
#
# WHAT SHIPS, AND WHY THE IMAGES ARE SMALL (ECR bills per GB stored, per image):
#
#   1. A shipped image is: the base, ONE virtualenv, the embedding export, the application. Nothing that built
#      them. Compilers, libpq-dev and poetry itself (and poetry's own ~40 dependencies) live in the build stages
#      only: poetry used to be pip-installed into the same /usr/local the runtime copied wholesale.
#   2. Each target carries only what it runs.
#        migrator  main
#        api       main + onnx    (embeds the search query with ONNX Runtime; extracts uploads)
#        worker    main + onnx + ocr (paddle is ~1 GB and only the worker reads scanned files)
#      The `ml` group (torch, sentence-transformers, transformers, scipy, scikit-learn: ~0.8 GB installed) is NOT
#      installed. EMBEDDING_RUNTIME defaults to onnx and docs/embedding-runtime-onnx.md records that the ml group
#      left the images; the Dockerfile had not caught up with that. The torch backend survives in a local checkout
#      (`poetry install --with ml`); EMBEDDING_RUNTIME=torch in a shipped image raises a clear EmbeddingUnavailable.
#   3. Every layer is written ONCE. A runtime stage that COPYs a venv and a later stage that COPYs a bigger venv
#      over it keeps both in the image (a layer cannot delete what a lower layer holds): that was 434 MB of
#      main-group libraries stored twice in api and worker. The targets now branch from a common base and each
#      copies exactly one venv.
#   4. The embedding export is downloaded in its own stage and only the two/three files the loader reads are
#      copied out. hf_hub_download keeps a cache copy of every file AND the old RUN then copied it again, so the
#      weights were stored twice (a 4.5 GB layer for 2.2 GB of weights).
#
# TWO RULES SHAPE THE ORDER, both learned the expensive way:
#
#   A. Application code is copied LAST, per target. Docker invalidates every layer above a changed one, so
#      anything large must sit below the thing that changes on every commit. `COPY . .` used to sit under the
#      model bake, so each commit stored a fresh 2.3 GB layer.
#   B. The embedding stack (ONNX Runtime + weights) reaches BOTH api and worker, because EMBEDDING_MODEL is a
#      local model and the api embeds the search query on every search. The migrator gets neither.
#
# Build:
#   docker build --target api      -t qnsc-kb-api .
#   docker build --target worker   -t qnsc-kb-worker .
#   docker build --target migrator -t qnsc-kb-migrator .
#
# The embedding export is baked by default and SHOULD be: the ONNX loader reads it from disk and has no runtime
# download path, so an unbaked image cannot embed at all - it serves /health and then falls back to keyword-only
# search on every query. Pass BAKE_EMBEDDING_MODEL=false only where that is intended, i.e. CI proving the image
# builds.

# The model, named ONCE; stages redeclare `ARG EMBEDDING_MODEL` without a default and inherit this one.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct

# The commit of that model repo the build downloads. A moving branch would let the Hub's owner change the weights
# under a corpus that is already embedded with the old ones, and nothing would raise: the vectors would just stop
# matching. Bump it deliberately, with the model.
ARG EMBEDDING_REVISION=274baa43b0e13e37fafa6428dbc7938e62e5c439

# Which export the api image serves queries from: `int8` (default) or `fp32`. The deploy pipeline passes no build
# args, so the default IS production. Rolling back is a rebuild with `fp32`, not a code change. The worker always
# writes fp32 vectors (see scripts/quantize_embedding.py for why the split is safe only while it stays a split).
ARG API_EMBEDDING=int8

# ---------------------------------------------------------------------------
# python-base - the one place the interpreter image is named, pinned by digest so a rebuild cannot change it
# underneath us (Renovate keeps the digest current). 3.13, NOT 3.14 - and this is a ceiling, not a lag.
# paddlepaddle 3.3.1 publishes cp39 through cp313 and no cp314, so the worker target cannot resolve its dependency
# set on 3.14 at all. Raise this only once paddlepaddle ships a cp314 wheel.
# ---------------------------------------------------------------------------
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS python-base

# ---------------------------------------------------------------------------
# builder - compilers and poetry. Never shipped.
#
# poetry gets its own venv (/opt/poetry) and installs INTO /opt/venv through VIRTUAL_ENV, so none of poetry's
# dependencies end up in the environment that ships.
# ---------------------------------------------------------------------------
FROM python-base AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN python -m venv /opt/poetry && \
    /opt/poetry/bin/pip install --no-cache-dir poetry && \
    python -m venv /opt/venv

COPY pyproject.toml poetry.lock ./

# `--only` is an exhaustive list, so each target names EVERY group it needs: `--only main,ocr` would resolve
# without onnx, and the worker would ship paddle and no embedding runtime.

# migrator
FROM builder AS deps
RUN /opt/poetry/bin/poetry install --no-root --only main && \
    rm -rf /opt/venv/lib/python3*/site-packages/pip* /opt/venv/bin/pip*

# api: ONNX Runtime + tokenizers. Not optional in practice: the api embeds the search query in-process, and the
# failure without it is a lazy import inside the model singleton, invisible until the first search.
FROM builder AS deps-onnx
RUN /opt/poetry/bin/poetry install --no-root --only main,onnx && \
    rm -rf /opt/venv/lib/python3*/site-packages/pip* /opt/venv/bin/pip*

# worker: the same, plus OCR. src/domain/source_extraction.py imports paddle INSIDE the functions that use it, so
# an image without it serves every other path and fails loudly only if asked to OCR - which the api never is.
FROM builder AS deps-onnx-ocr
RUN /opt/poetry/bin/poetry install --no-root --only main,onnx,ocr && \
    rm -rf /opt/venv/lib/python3*/site-packages/pip* /opt/venv/bin/pip*

# ---------------------------------------------------------------------------
# hub - huggingface_hub and nothing else, so downloading the 2.2 GB export does not depend on the application's
# lockfile: a poetry.lock change re-runs the builder stages, not this one. Pinned to the version poetry.lock
# resolves, because this stage is outside it.
# ---------------------------------------------------------------------------
FROM python-base AS hub

RUN pip install --no-cache-dir "huggingface_hub==1.27.0"

# ---------------------------------------------------------------------------
# model - the fp32 embedding export, downloaded where its cache is thrown away. The WORKER's, and the api's when
# built with API_EMBEDDING=fp32.
#
# The loader (src/lib/embeddings/local_onnx.py) reads EMBEDDING_ONNX_DIR and wants model.onnx and tokenizer.json,
# plus model.onnx_data when the model's fp32 graph exceeds protobuf's 2 GB limit (true for
# multilingual-e5-large-instruct). The model publishes its own ONNX export, so no optimum step and no torch.
#
# fp32, NOT a qint8 variant, for everything that WRITES vectors: quantisation moves them (cosine 0.972-0.987 against
# the reference in this repo's parity gate), and a corpus embedded slightly off the space its queries are embedded in
# degrades retrieval silently. The api alone serves queries from int8 (embedding-int8 below).
#
# HF_HOME points at /tmp so the Hub cache (a second copy of every file) stays in this stage and is not shipped.
# File mtimes are set to the epoch so the COPY layer has the same digest on every rebuild: registry layers are
# content-addressed, and an unchanged 2.2 GB layer should be stored once, not once per build.
#
# NOTE for anyone editing the python -c: it collapses every backslash-continued line into ONE logical line, so only
# constructs valid as a single line survive (list comprehensions yes; try/except or an `if:` block no).
# ---------------------------------------------------------------------------
FROM hub AS model

ARG BAKE_EMBEDDING_MODEL=true
ARG EMBEDDING_MODEL
ARG EMBEDDING_REVISION
ENV HF_HOME=/tmp/hf \
    EMBEDDING_ONNX_DIR=/opt/embedding-export

RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p "$HF_HOME" "$EMBEDDING_ONNX_DIR" && \
    if [ "$BAKE_EMBEDDING_MODEL" = "true" ]; then \
        export HF_XET_HIGH_PERFORMANCE=1 && \
        if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
        python -c "\
from huggingface_hub import hf_hub_download, list_repo_files; \
import shutil, os; \
target = os.environ['EMBEDDING_ONNX_DIR']; \
[shutil.copyfile(hf_hub_download('${EMBEDDING_MODEL}', name, revision='${EMBEDDING_REVISION}'), os.path.join(target, os.path.basename(name))) \
 for name in ('onnx/model.onnx', 'tokenizer.json')]; \
[shutil.copyfile(hf_hub_download('${EMBEDDING_MODEL}', 'onnx/model.onnx_data', revision='${EMBEDDING_REVISION}'), os.path.join(target, 'model.onnx_data')) \
 for _ in [0] if 'onnx/model.onnx_data' in list_repo_files('${EMBEDDING_MODEL}', revision='${EMBEDDING_REVISION}')]"; \
    fi && \
    rm -rf "$HF_HOME" && \
    find "$EMBEDDING_ONNX_DIR" -exec touch -h -d @0 {} +

RUN chmod -R a+rX "$EMBEDDING_ONNX_DIR"

# ---------------------------------------------------------------------------
# embedding-int8 - the api's export: the same model quantised, so a search query is embedded in ~55 ms and ~700 MB
# instead of ~200 ms and ~1.6 GB. The api embeds ONE short question per search; the worker embeds every chunk that
# is stored, and an int8 vector written into the same column as fp32 ones would degrade retrieval without raising.
# So this stage reaches the api only, and nothing here reaches the worker.
#
# FROM deps-onnx, not a torch-carrying stage: quantising needs onnxruntime (quantize_dynamic), tokenizers, numpy
# and huggingface_hub, all in the onnx group, plus `onnx` itself, installed here and discarded with the stage. The
# download and the quantisation run only when baking is on (CI builds with BAKE_EMBEDDING_MODEL=false).
# ---------------------------------------------------------------------------
FROM deps-onnx AS embedding-int8

ARG BAKE_EMBEDDING_MODEL=true
ARG EMBEDDING_MODEL
ARG EMBEDDING_REVISION
ENV HF_HOME=/tmp/hf

RUN pip install --no-cache-dir "onnx==1.23.2"

COPY scripts/quantize_embedding.py /opt/quantize_embedding.py

# Same Hub-speed lever and optional token secret as the fp32 download above.
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /opt/embedding-export && \
    if [ "$BAKE_EMBEDDING_MODEL" = "true" ]; then \
        export HF_XET_HIGH_PERFORMANCE=1 && \
        if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
        python /opt/quantize_embedding.py \
            --download "$EMBEDDING_MODEL" \
            --revision "$EMBEDDING_REVISION" \
            --fp32-dir /opt/embedding-fp32 \
            --out-dir /opt/embedding-export; \
    fi && \
    rm -rf "$HF_HOME" /opt/embedding-fp32 && \
    find /opt/embedding-export -exec touch -h -d @0 {} +

# onnxruntime writes the quantised weights 0600, root-owned, and the api runs as appuser: unreadable, the load fails
# with a permission error deep inside InferenceSession and the api falls back to keyword-only search. Make every file
# readable by whatever uid the task runs as. (chmod changes ctime, not mtime, so the normalised digest holds.)
RUN chmod -R a+rX /opt/embedding-export

# embedding-fp32 is the model stage under the name the api's switch uses; api-embedding is whichever one
# API_EMBEDDING selects. COPY --from cannot expand a build arg, FROM can.
FROM model AS embedding-fp32

FROM embedding-${API_EMBEDDING} AS api-embedding

# ---------------------------------------------------------------------------
# runtime-base - the shipped base. NO libraries and NO application code.
#
# Debian security updates for the base image's own packages, applied in the stage every shipped image derives
# from. Trivy's CRITICAL gate blocks on base-image CVEs, not just ours (run 34751584705 failed on three perl-base
# advisories with zero findings in our Python dependencies); python:3.13-slim ships whatever Debian had at ITS
# build time, so this recurs whenever an advisory lands before the upstream tag is rebuilt.
# `upgrade`, not `dist-upgrade`: patches within the pinned release only.
# ---------------------------------------------------------------------------
FROM python-base AS runtime-base

RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# /opt/venv/bin first: uvicorn, celery and alembic are console scripts of the venv, and `python` must be the venv's.
# HF_HOME is where the compose file mounts a developer's own model cache; the baked export is EMBEDDING_ONNX_DIR.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/huggingface \
    EMBEDDING_ONNX_DIR=/opt/embedding-onnx \
    # /app must be importable, not merely the working directory. Running a script BY PATH
    # (`python scripts/bootstrap_db_role.py`) puts /app/scripts on sys.path - not /app - so
    # `import src.core.config` raises ModuleNotFoundError. uvicorn and celery hide this because they import by
    # module name from the CWD, so the failure appeared only in the migrator, only once deployed.
    PYTHONPATH=/app

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/storage/sources /app/storage/connectors "$HF_HOME" && \
    chown -R appuser:appuser /app "$HF_HOME"

# pip is build-time only (poetry installs into the venv in the deps stages), and it
# cannot simply be upgraded away: its vendored certifi copy ships without dist-info,
# so the scanner pins the version from the PEM bundle and a CRITICAL stands no matter
# what the real certifi says. Nothing at runtime installs packages, so both pip trees
# (the system one and the venv's, each with its vendored certifi) are removed here.
RUN rm -rf /usr/local/lib/python3*/site-packages/pip* \
           /opt/venv/lib/python3*/site-packages/pip* \
           /usr/local/bin/pip* /opt/venv/bin/pip*

# ---------------------------------------------------------------------------
# targets. Order inside each: libraries, then the (identical-digest) model, then code last (rule A).
# ---------------------------------------------------------------------------
FROM runtime-base AS api

COPY --from=deps-onnx /opt/venv /opt/venv
COPY --from=api-embedding /opt/embedding-export /opt/embedding-onnx
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS worker

# paddle's native dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps-onnx-ocr /opt/venv /opt/venv
COPY --from=model /opt/embedding-export /opt/embedding-onnx
COPY --chown=appuser:appuser . .

USER appuser

# --concurrency=1 because each prefork child loads its OWN embedding session, and the
# task is sized for exactly one. Celery defaults this to os.cpu_count(), which on Fargate
# reports the host's CPUs rather than the task's, so the count is neither 1 nor reliably
# related to what the task was given -- develop came up at 2.
#
# The e5-large-instruct ONNX session is ~1.5 GB resident. clamav and beat carry hard
# container limits of 2048 and 256 MB out of a 4096 MB task, leaving the worker ~1.8 GB,
# and infra/live/*/main.tf sizes that against ONE session plus a per-file PaddleOCR spike.
# Two children embedding at once do not fit: on develop, six articles picked up between
# 15:25:50 and 15:26:34 all died with
#
#     billiard.exceptions.WorkerLostError: Worker exited prematurely: signal 9 (SIGKILL)
#
# and SIGKILL raises nothing in Python, so index_article's except never ran and each row
# stayed at index_status="processing" forever -- the UI shows those articles as still
# indexing when nothing is working on them. This was latent under e5-small (384 dims, a
# much smaller session) and only started biting when develop moved to e5-large-instruct.
#
# Serial is also no throughput loss here: max_count is 1 on this service (beat is a
# singleton in the same task), so ingestion was never parallel across tasks either.
# Raising this means re-sizing the task memory for N sessions first, not just the flag.
CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info", \
     "--concurrency=1", \
     "-Q", "document-preparation,celery,ingestion,connectors,permissions"]

FROM runtime-base AS migrator

COPY --from=deps /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

USER appuser

ENTRYPOINT ["/app/docker/migrate-entrypoint.sh"]
CMD ["alembic", "-c", "migrations/alembic.ini", "upgrade", "head"]
