from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any
from urllib.parse import quote, urlparse
import re


# The password the bootstrap administrator gets when nothing else is configured. It is
# written down in this repository on purpose — a local checkout should just work — which
# is exactly why validate_production refuses to start with it. Defined here rather than
# beside the bootstrap code so that check has no import cycle to reach it.
DEVELOPMENT_DEFAULT_PASSWORD = "Admin123@"


def is_cloudflare_r2_endpoint(value: str | None) -> bool:
    """Return whether a configured URL is a Cloudflare R2 S3 endpoint."""
    if not value:
        return False
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return (
        parsed.scheme == "https"
        and bool(hostname)
        and hostname.endswith(".r2.cloudflarestorage.com")
        and hostname != "r2.cloudflarestorage.com"
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


class Settings(BaseSettings):
    PROJECT_NAME: str = "QNSC Knowledge Base"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "super-secret-key-change-in-production"
    # Separate at-rest encryption from JWT signing. Keep previous values only
    # during a deliberate rotation window.
    DATA_ENCRYPTION_KEY: str | None = None
    PREVIOUS_DATA_ENCRYPTION_KEYS: str = ""
    # Defaults to the HARDENED value, so every hardening check below is opted OUT of
    # explicitly rather than opted IN to. It was "development", which made the whole of
    # validate_production fail open: an unset ENVIRONMENT — a missed task-definition
    # variable, a container platform that drops empty values, a fresh EC2 box started by
    # hand — booted a public deployment with the committed SECRET_KEY signing its JWTs,
    # /docs published, self-registration reachable, and the bootstrap administrator
    # holding the password written down in this file. Nothing logged a warning, because
    # from the process's point of view it was a normal development run.
    #
    # Local development, the dev Compose stack and the test suite therefore set this
    # explicitly (docker-compose.yml, .env.example, tests/conftest.py).
    ENVIRONMENT: str = "production"
    CORS_ORIGINS: str = "http://localhost:5173"
    FRONTEND_URL: str = "http://localhost:5173"
    # Schema lifecycle is owned exclusively by Alembic migrations.
    AUTO_CREATE_SCHEMA: bool = False
    JOB_MODE: str = "inline"
    # How often the inline-mode outbox recovery loop retries failed/stale
    # events (the Celery beat equivalent is replay_outbox_task).
    OUTBOX_RECOVERY_INTERVAL_SECONDS: int = 300
    ENABLE_API_DOCS: bool = True
    ENABLE_RLS: bool = False
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/qnsc_kb"
    MIGRATION_DATABASE_URL: str | None = None
    APP_DATABASE_ROLE: str | None = None
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Connection PARTS — an alternative to the two URLs above ────────────────
    # A managed database hands out its credentials as a rotatable secret, and a
    # container platform injects a secret as its own environment variable. There is
    # no supported way to interpolate one into the middle of a URL string at task
    # start, so a deployment that only accepts DATABASE_URL forces the whole URL —
    # password included — to be a hand-maintained secret. That copy then silently
    # goes stale on the next endpoint change or credential rotation, and the failure
    # appears as an authentication error somewhere else entirely.
    #
    # When DATABASE_HOST is set, the URL is composed from these parts instead (see
    # model_post_init). An explicitly supplied DATABASE_URL always wins, so local
    # development, tests and Compose are untouched.
    DATABASE_HOST: str | None = None
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "qnsc_kb"
    DATABASE_USER: str | None = None
    DATABASE_PASSWORD: str | None = None
    # The master credential, used ONLY by the migrator task: migrations create
    # extensions (vector, pgcrypto) and grant privileges, neither of which the
    # least-privilege application role may do. Host, port and name are shared with
    # the application parts above.
    MIGRATION_DATABASE_USER: str | None = None
    MIGRATION_DATABASE_PASSWORD: str | None = None

    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GLM_API_KEY: str | None = None
    LLM_PROVIDER: str | None = None
    LLM_BASE_URL: str | None = None
    # Custom endpoints can forward the workspace API key to an arbitrary
    # host. Keep that capability opt-in, even for global administrators.
    LLM_ALLOW_CUSTOM_BASE_URL: bool = False
    GEMINI_API_KEY: str | None = None
    # A floating `-latest` alias, not a pinned version. Google retires pinned models for
    # new API keys — the deployed default was gemini-2.5-flash-lite, and every call
    # returned 404 "no longer available to new users". Nothing validates a model name at
    # startup, so it surfaced as broken AI answers, not as a failed deploy.
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GEMINI_API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_THINKING_LEVEL: str = "minimal"
    LLM_TIMEOUT_SECONDS: float = 90.0
    # Local, in-process, and deliberately so: no embedding text leaves the deployment and
    # no third-party key gates indexing or search.
    #
    # It is paid for in image size and memory. The API embeds the search QUERY on every
    # search, so the ONNX graph and runtime live in the api image as well as the worker's
    # — the `ml` dependency group is installed in both, and neither is small. Choosing a
    # hosted model instead (gemini-embedding-001, text-embedding-3-small) is a one-line
    # change here plus the migration below, and src/lib/embeddings.py already carries
    # that path.
    #
    # This value fixes EMBEDDING_DIMENSION, which fixes the pgvector column width and the
    # HNSW index AT MIGRATION TIME. Changing it later needs a migration and a full
    # re-embed: a query and a chunk embedded by different models are points in unrelated
    # spaces, and their distance is meaningless rather than merely wrong.
    #
    # HISTORY (superseded, kept for the reasoning trail):
    # - paraphrase-multilingual-MiniLM-L12-v2 shipped first: 12 layers/384-dim, the
    #   fastest CPU-only option, multilingual, own ONNX export.
    # - intfloat/multilingual-e5-small replaced it (see EMBEDDING_VERSION history below):
    #   MiniLM is a symmetric paraphrase/STS model, wrong training objective for
    #   question->passage retrieval. VN-MTEB (arXiv 2507.21500, Table 3): e5-small scores
    #   34.12 retrieval against MiniLM's 14.14. Measured here on MLQA/UIT-ViQuAD dev
    #   (retrieval-only R@10): MiniLM 80.5/56.5/68.8 (en/vi/ViQuAD-vi) vs e5-small
    #   91.0/61.0/67.4; end-to-end F1 en 54.28->62.20, vi 32.08->33.88, ViQuAD 48.11->51.41.
    #   Same 384 width, so no pgvector column change -- delete+re-embed only.
    #
    # CURRENT: intfloat/multilingual-e5-large-instruct, 24 layers/1024-dim. On the same
    # VN-MTEB benchmark it scores 40.88 retrieval (vs e5-small's 34.12) and the highest
    # overall average of all 18 models the paper measured — ahead of every 7B model
    # tested, including gte-Qwen2-7B-instruct and e5-Mistral-7B-instruct. e5-base was
    # measured too and rejected: 34.50 retrieval, barely above e5-small, not worth its own
    # cost. Unlike the same-width e5-small swap, this DOES change EMBEDDING_DIMENSION
    # (384->1024) and needs: a real pgvector/HNSW migration (ALTER COLUMN + index rebuild,
    # not delete+re-embed — see migrations/versions/202608*_*_realign_embedding_dimension.py
    # for the template), a fresh ONNX export (verify EMBEDDING_ONNX_POOLING against this
    # model's own 1_Pooling/config.json rather than assuming "mean" carries over), and a
    # torch/ONNX parity check (tests/unit/test_embedding_backends.py) before flipping any
    # environment. Infra (infra/live/*/main.tf, Dockerfile ARG) must move in lockstep —
    # tests/unit/test_embedding_config_matches_image.py enforces that.
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    # Names the MODEL that produced a vector, and hybrid_search filters on it, so a
    # mislabelled corpus is an invisible corpus. Rows written while this said
    # "bge-m3-v1" hold MiniLM vectors — provably, since a vector(384) column cannot
    # hold bge-m3's 1024 — and need re-stamping or re-indexing once.
    EMBEDDING_VERSION: str = "e5-large-instruct-v1"

    # HOW the model runs, kept separate from WHICH model runs.
    #
    # Defaults to onnx because it is the only value this project's dependencies can
    # satisfy: the `ml` group is onnxruntime + transformers, and neither torch nor
    # sentence-transformers is declared anywhere. The default used to be "torch", so a
    # deployment that changed nothing could not embed at all — every search fell back to
    # keyword-only and logged an error suggesting an install that would not have helped.
    #
    # onnx still requires an export (model.onnx + tokenizer.json) in EMBEDDING_ONNX_DIR;
    # see src/lib/embeddings/local_onnx.py. Until that exists, embeddings are unavailable
    # either way — but the failure now names the thing that is actually missing.
    EMBEDDING_RUNTIME: str = "onnx"
    EMBEDDING_ONNX_DIR: str = "/opt/embedding-onnx"
    # PER MODEL, and not inferable from the export: an ONNX graph does not carry the
    # pooling config sentence-transformers reads. Every e5 model to date (small, base,
    # large, and large-instruct) publishes pooling_mode_mean_tokens=true in its
    # 1_Pooling/config.json, so "mean" carries over unchanged across those swaps — but
    # CONFIRM this against multilingual-e5-large-instruct's own config.json before
    # shipping rather than assuming the family convention holds; the wrong choice (e.g.
    # "cls", right for bge-*) produces perfectly valid vectors in the wrong space,
    # degrading retrieval with no error.
    EMBEDDING_ONNX_POOLING: str = "mean"
    # 2, matched to the worker's 2048 CPU units. This was 1, justified by a comment in
    # local_onnx.py about "a 0.5 vCPU task" — a sizing the worker has not had for some
    # time, and the number was never revisited when it grew. One thread on two cores
    # leaves half the task idle through the whole embedding pass.
    #
    # It is a CEILING tied to the task size, not a free dial: more threads than cores
    # spends the difference on scheduling, which is what the original comment was right
    # about. Raise this and the `cpu` in infra/live/*/main.tf together or not at all.
    EMBEDDING_ONNX_THREADS: int = 2
    # multilingual-e5-large-instruct shares e5-small/e5-base's XLM-R tokenizer window:
    # sentence_bert_config.json max_seq_length 512. Unchanged from the e5-small default,
    # so this value does NOT need to move for the large-instruct swap — but still confirm
    # against the model's own sentence_bert_config.json before shipping (the 8192 value
    # would be bge-m3's window; the tokenizer silently truncates rather than erroring on
    # a mismatch, so a wrong number here degrades long inputs with nothing to notice).
    #
    # This is also why chunker.py sizes retrieval children at up to 250 characters —
    # raise both together or neither.
    EMBEDDING_MAX_TOKENS: int = 512
    EMBEDDING_BATCH_SIZE: int = 32
    # Hosted-embedding retries. Rate limits are the EXPECTED condition for a hosted
    # embedding API during bulk ingestion, not an exceptional one: without a retry a
    # single 429 fails the index for a whole article. Only 429 and 5xx are retried — a
    # 400 means the request is wrong and repeating it just spends the quota again.
    EMBEDDING_HTTP_MAX_ATTEMPTS: int = 4
    EMBEDDING_HTTP_BACKOFF_SECONDS: float = 1.0

    # Stamped per chunk (indexing.py), not read back anywhere to trigger anything
    # automatically -- a version bump alone does not re-index existing content. Forcing
    # already-published articles onto the new version needs its own migration that
    # re-queues them (see 20260828_64/20260912_77 for the same pattern on
    # EMBEDDING_VERSION). Bumped for contextual chunk headers (src/rag/contextual_header.py):
    # the embedding INPUT changed (a section-level header is now prepended before a child
    # is embedded), even though chunk boundaries and stored chunk_text did not.
    CHUNKING_VERSION: str = "v3-contextual-headers"
    EMBEDDING_DIMENSION: int | None = None
    # Opt-in for the realign migrations (20260810_51 / 20260828_64 / 20260912_77).
    #
    # Those revisions refuse to run when the pgvector column is at the wrong width AND
    # embeddings are present, because vectors of different widths are not comparable --
    # a 1024-dimension vector is not an extension of a 384-dimension one, so there is no
    # in-place conversion, only a rebuild. Refusing is the right default: the alternative
    # is a migration that silently destroys every embedding in the database.
    #
    # The refusal left no supported way forward, though. The docstring said to delete the
    # chunks by hand and re-run, which means the recovery path for a blocked deploy was an
    # ad-hoc DELETE typed against a live database, outside migration history and outside
    # review. This flag makes that step explicit, reviewable and logged instead: set it and
    # the migration does the wipe itself, in the same transaction as the ALTER, then requeues
    # every published article so the API's startup sweep re-embeds at the new width.
    #
    # It stays False by default, so nothing changes for anyone who does not set it. The
    # guard still fires, and its message names this flag rather than prescribing manual SQL.
    EMBEDDING_REALIGN_DISCARD_VECTORS: bool = False
    LLM_MODEL: str = "gemma-4-26b-a4b-it"
    RESTRUCTURE_ENABLED: bool = True
    RESTRUCTURE_MODEL: str | None = None
    # GLM-5 has a 200K context window shared by prompt and generation. 120K
    # characters leaves practical headroom for the system prompt and a 64K
    # lossless Markdown response, even for token-dense source languages.
    RESTRUCTURE_MAX_CHARS: int = 120000
    # Restructuring regenerates the WHOLE document, so wall clock is bound by output
    # tokens and grows with length. A 26,633-character upload measured 304 s as one
    # call — against RESTRUCTURE_TIMEOUT_SECONDS of 300, which it cleared by a margin
    # too thin to rely on. Past that edge the result is not an error but a silent
    # fallback to unformatted text.
    #
    # Sectioning existed already but only above RESTRUCTURE_MAX_CHARS, which no real
    # document reached, and it ran the sections SEQUENTIALLY — so the one path that did
    # split was the slowest of all.
    #
    # 8000 splits that upload into 4 sections; run together they finish in about the
    # time of one, and each sits far from the timeout. Smaller is faster and choppier:
    # the seams between sections are where headings lose their thread, because each
    # section is formatted without sight of its neighbours.
    RESTRUCTURE_SECTION_CHARS: int = 8000
    # Concurrent sections. The ceiling is the provider's rate limit, not ours — GLM
    # refuses a burst rather than queueing it, and a refused section falls back to
    # unformatted text for that part of the document.
    RESTRUCTURE_MAX_CONCURRENCY: int = 4
    # Formatting is an optional enhancement. Keep review responsive and use
    # the lossless local fallback when the configured provider is slow.
    RESTRUCTURE_TIMEOUT_SECONDS: float = 300.0
    # A lossless reading view can approach the source length. Keep enough
    # space for long documents while retaining a finite cost/time safeguard.
    RESTRUCTURE_MAX_OUTPUT_TOKENS: int = 65536
    RESTRUCTURE_NUMERIC_COVERAGE_THRESHOLD: float = 0.90
    AI_RATE_LIMIT_PER_MINUTE: int = 30
    VECTOR_DISTANCE_THRESHOLD: float = 0.45
    RAG_MIN_RELEVANCE_SCORE: float = 0.12
    RAG_MIN_CONTEXT_SCORE: float = 0.35
    # Above RAG_MIN_CONTEXT_SCORE (the refusal line) but below this, an answer is
    # generated normally yet marked `"confidence": "low"` in the response instead of
    # presented as certain -- see ai_service.py. This is NOT self-consistency sampling
    # (asking twice and comparing): `complete()` hardcodes temperature=0.0 at every call
    # site in this file, deliberately, so the KB's answers are reproducible, and asking
    # the same deterministic prompt twice would mostly return the same text and prove
    # nothing. The retrieval score is a signal that already exists at zero extra LLM
    # cost, so it is what this reads instead.
    RAG_LOW_CONFIDENCE_SCORE: float = 0.5
    # Off by default, same posture as CROSS_ENCODER_RERANKER_ENABLED/GRAPH_RETRIEVAL_ENABLED
    # above: a new code path with no production traffic behind it yet, and this one adds a
    # real LLM call (rag/llm_judge.py::judge_entailment, per flagged sentence) to the live
    # /ai/ask latency, unlike those two which are pure retrieval-side additions. Flip once
    # measured against real answers.
    CLAIM_VERIFICATION_ENABLED: bool = False
    # Bounds the worst case: a long, heavily-cited answer must not turn into an unbounded
    # number of judge calls before the response can return.
    CLAIM_VERIFICATION_MAX_SENTENCES: int = 5
    # Off by default, deliberately: this is a new code path with no production traffic
    # behind it yet. Flip it once it has been proven against a real corpus and the `ml`
    # poetry group (sentence-transformers) is actually installed in the target image —
    # same shape as RESTRUCTURE_ENABLED above, and the same reasoning as
    # EMBEDDING_RUNTIME defaulting to onnx: the default must be satisfiable by what ships,
    # not by what a developer happens to have installed locally.
    CROSS_ENCODER_RERANKER_ENABLED: bool = False
    CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # Off by default for the same reason: a change to what the reader sees on the live
    # answer path, with no production traffic behind it yet to confirm the ambiguity
    # heuristic (rag/query_router.py::detect_ambiguous_departments) actually fires on
    # real questions rather than on ordinary ones with two loosely-relevant departments.
    CLARIFICATION_ON_AMBIGUOUS_DEPARTMENTS_ENABLED: bool = False
    # How many hybrid_search candidates are re-scored. A cross-encoder pays for every
    # (query, passage) pair it scores, unlike a bi-encoder embedding computed once per
    # passage at index time — so this is a cost/latency dial, not a recall ceiling like
    # RAG_CANDIDATE_POOL_SIZE below.
    CROSS_ENCODER_CANDIDATE_POOL: int = 30
    # Off by default, same posture as CROSS_ENCODER_RERANKER_ENABLED above: a new
    # retrieval leg with no production traffic behind it yet. It adds a third
    # candidate list to hybrid_search's existing RRF fusion (src/repositories/chunk.py) --
    # chunks from articles whose entity-graph mentions overlap with entities named in the
    # query, one hop out (see ChunkRepository._graph_augmented_candidates). Flip once
    # measured against a real corpus with a populated graph, same discipline as every
    # other retrieval change in this file.
    GRAPH_RETRIEVAL_ENABLED: bool = False
    # Article-level, not chunk-level (ArticleEntityMention records "this article mentions
    # this entity", not which chunk), so this leg is coarser than the vector/keyword legs
    # by construction -- kept smaller than RAG_CANDIDATE_POOL_SIZE so a broad entity match
    # cannot dominate the fused pool.
    GRAPH_RETRIEVAL_POOL_SIZE: int = 30
    # Off by default, same posture as CROSS_ENCODER_RERANKER_ENABLED/GRAPH_RETRIEVAL_ENABLED
    # above: a new code path with no production traffic behind it yet, and this one adds a
    # real LLM call (rag/multi_query.py) to the live search path. When on: a non-comparison
    # query (a comparison already gets its own free, deterministic decomposition just above
    # -- this does not layer on top of that one) is expanded into up to
    # MULTI_QUERY_MAX_SUBQUERIES additional targeted search queries, run in parallel with
    # the original and merged the same way comparison subjects already are. Flip once
    # measured against real traffic, same discipline as every other retrieval change here.
    MULTI_QUERY_RETRIEVAL_ENABLED: bool = False
    MULTI_QUERY_MAX_SUBQUERIES: int = 3
    # When the reranked pool's top score is still below RAG_MIN_CONTEXT_SCORE -- the same
    # line ai_service.py uses to decide "not enough to answer confidently" -- one further,
    # LLM-proposed query is tried and merged in before that decision is made. Bounded (not
    # a while-true) for the same reason MAX_NEIGHBOR_DEPTH bounds graph traversal: an
    # adaptive loop must still have a worst-case cost. 0 disables the follow-up round
    # without disabling the sub-query expansion above.
    MULTI_QUERY_MAX_FOLLOWUPS: int = 1
    # 48. Each retrieval leg truncates here BEFORE RRF fusion, so it is a recall
    # ceiling rather than a performance dial — and it is measurably too shallow for
    # Vietnamese: on failed MLQA-vi questions the sparse leg had already found the
    # gold document 96.7% of the time, at median rank 172, and this cut it.
    #
    # Raising it to 256 was measured both ways (experiments/results.jsonl, iteration 1)
    # and only pays off with a retrieval-trained encoder: with multilingual-e5-small it
    # moved ViQuAD R@10 67.4 -> 74.5 and MLQA-vi 61.0 -> 63.0, but with this MiniLM
    # default it gave MLQA-en +2.0 R@10 while MLQA-vi went 56.5 -> 56.0, for about
    # +400ms retrieval p50. Left at 48 until the encoder that justifies it is adopted.
    RAG_CANDIDATE_POOL_SIZE: int = 240
    # Must stay >= RAG_CANDIDATE_POOL_SIZE, and higher, because pgvector filters AFTER
    # the index scan: the permission and published-status predicates consume
    # candidates the index already committed to. pgvector's default is 40, i.e. below
    # the pool we ask for. Measured recall on a 128-dim/1M set: 40 -> 95.4%, 200 ->
    # 99.8%, at 1.19ms -> 4.60ms p99 (jkatz05.com/post/postgres/pgvector-scalar-binary-quantization).
    # Valid range is 1..1000; raising it costs latency, so it is a setting, not a literal.
    #
    # Raise this WITH RAG_CANDIDATE_POOL_SIZE or the pool increase buys nothing: 256
    # needs roughly 400 here.
    HNSW_EF_SEARCH: int = 1000
    RAG_RERANK_LIMIT: int = 80
    RAG_MAX_CONTEXT_PARENTS: int = 40
    RAG_CONTEXT_MAX_CHARS: int = 70000
    RAG_CONTEXT_MAX_TOKENS: int = 17500
    RAG_PARENT_CONTEXT_CHARS: int = 2400
    RAG_MAX_PARENTS_PER_ARTICLE: int = 15
    PROMPT_VERSION: str = "v2.1-query-language-grounded-extended-sections"
    # Part of the ai_cache key (src/domain/ai_service.py:917-924), so this MUST move
    # whenever retrieval behaviour changes, or a cached answer from the old pipeline is
    # served for six hours as if it came from the new one. Bumped for the
    # e5-large-instruct encoder swap (see EMBEDDING_MODEL) -- a cached answer keyed to
    # v4's e5-small-derived retrieval must not be served once the vector space changes.
    RETRIEVAL_VERSION: str = "v5-e5-large-instruct"
    RERANKER_VERSION: str = "v1.2-definition-aware"
    RAG_ENABLE_EXTENDED_SECTION: bool = True
    RAG_CACHE_EXTENDED_SECTION: bool = False
    # False by default: an intentional product choice, not an oversight, so changing it
    # is a product decision for whoever owns that tradeoff -- not something to flip here
    # as a side effect of one local bug report. See docker-compose.yml (dev override) for
    # what turning it on actually does and why it was tried there instead.
    RAG_ALLOW_EXTENDED_ON_REFUSAL: bool = False
    # Character budget for conversation history injected into the prompt.
    # Bounds total prompt size (history + RAG context + system prompt) so long
    # conversations cannot overflow smaller models' context windows.
    RAG_HISTORY_MAX_CHARS: int = 9000
    # Output cap for the main RAG generation path. Without it, provider-side
    # output length is unbounded (cost/latency exposure); only Gemini enforced
    # its own cap before this setting existed.
    #
    # NOW UNCAPPED BY DEFAULT (None). 2048 cut replies off mid-word: ONE generation
    # produces both the grounded answer and the EXTENDED section, split afterwards on
    # the sentinel, so the cap was shared between them and the extended half is what ran
    # out. Vietnamese also costs roughly twice the tokens per character that English
    # does, so 2048 bought about half the answer it appeared to.
    #
    # None means the `max_tokens` field is simply not sent and the model stops when the
    # answer is finished, which is the behaviour people expect. The runaway-generation
    # exposure the cap guarded against is small in practice — a grounded answer ends
    # when it ends — and it is now much smaller still, because the reasoning tokens that
    # were consuming the budget are disabled on this path.
    #
    # Set an integer to put the bound back; the value is passed straight through to
    # every provider, Gemini included — its branch now omits `maxOutputTokens` on None
    # the same way the OpenAI-compatible branch omits `max_tokens`, so there is no
    # separate Gemini-only ceiling to also change. One thing to know if you do set it: a
    # very long answer becomes part of the next turn's history, which RAG_HISTORY_MAX_CHARS
    # trims at 9,000 characters, so a rambling reply crowds out the conversation before it
    # costs anything else.
    RAG_MAX_ANSWER_TOKENS: int | None = None
    OIDC_ISSUER_URL: str | None = None
    OIDC_CLIENT_ID: str | None = None
    OIDC_CLIENT_SECRET: str | None = None
    OIDC_REDIRECT_URI: str | None = None
    OIDC_SCOPES: str = "openid profile email"
    MICROSOFT_CLIENT_ID: str | None = None
    MICROSOFT_CLIENT_SECRET: str | None = None
    MICROSOFT_TENANT_ID: str = "common"
    MICROSOFT_REDIRECT_URI: str | None = None
    MICROSOFT_LOGIN_REDIRECT_URI: str | None = None
    # Entra is an authentication provider, not an account-provisioning path.
    # A user must have a matching, unexpired invitation.
    ENTRA_AUTO_PROVISION_DOMAIN: str = ""
    MICROSOFT_GRAPH_SENDER: str | None = None
    MICROSOFT_GRAPH_SCOPE: str = "https://graph.microsoft.com/.default"
    # "graph" keeps the existing Microsoft Graph transport (and its development
    # fallback to FakeEmailSender below). Set to "ses" to send through AWS SES
    # instead — useful locally when you want invitations to actually arrive
    # without an Entra app registration — or "fake" to force the in-memory
    # sender regardless of environment. get_email_sender() in src/services/email.py
    # is what reads this.
    EMAIL_PROVIDER: str = "graph"
    # Required when EMAIL_PROVIDER=ses; SesEmailSender raises without it.
    MAIL_FROM_EMAIL: str | None = None
    MAIL_FROM_NAME: str = "QNSC Knowledge Base"
    # ``delegated`` keeps the interactive OAuth flow. ``application`` uses the
    # tenant-approved client-credentials flow for unattended 24/7 ingestion.
    MICROSOFT_CONNECTOR_AUTH_MODE: str = "delegated"
    MICROSOFT_GRAPH_SUBSCRIPTION_MINUTES: int = 1440
    # Comma-separated Entra object IDs. Application mode deliberately avoids
    # tenant-wide site/user discovery so it works with Sites.Selected and a
    # least-privilege OneDrive allowlist.
    MICROSOFT_SHAREPOINT_SITE_IDS: str = ""
    MICROSOFT_ONEDRIVE_USER_IDS: str = ""
    CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS: int = 30
    CONNECTOR_RECONCILE_INTERVAL_MINUTES: int = 360
    CONNECTOR_AUTO_PUBLISH_MODE: str = "governed"
    # Department routing asks the LLM first and falls back to keyword ranking. The
    # fallback is not a degraded mode: it is what runs whenever no provider is
    # configured, and it is what every existing deployment already had.
    DEPARTMENT_ROUTING_LLM_ENABLED: bool = True
    # Short on purpose. This runs inside document ingest, once per document, and a slow
    # provider must cost a suggestion rather than the import.
    DEPARTMENT_ROUTING_LLM_TIMEOUT: float = 20.0
    # Builds the tenant knowledge graph (domain/entity_extraction.py, graph_service.py)
    # from every published article. On by default, same reasoning as
    # DEPARTMENT_ROUTING_LLM_ENABLED: it degrades to "no entities extracted this publish"
    # whenever no provider is configured, never to a broken publish, so there is no
    # unsatisfiable-dependency reason to default it off the way CROSS_ENCODER_RERANKER_
    # ENABLED is (that one needs an extra dependency group actually installed).
    GRAPH_EXTRACTION_ENABLED: bool = True
    # Longer than DEPARTMENT_ROUTING_LLM_TIMEOUT: this runs once per publish rather than
    # once per document import, and the prompt asks for a whole entity+relationship
    # graph rather than one classification, so it is allowed more time before falling
    # back to "no entities extracted this publish".
    GRAPH_EXTRACTION_TIMEOUT_SECONDS: float = 45.0
    # The approval agent applies an administrator's written rule to pending drafts. It
    # is off unless a rule exists AND that rule was explicitly granted authority, so this
    # switch exists to stop it entirely without deleting anyone's rules.
    APPROVAL_AGENT_ENABLED: bool = True
    APPROVAL_AGENT_TIMEOUT: float = 30.0
    # Drafts examined per run. A connector can drop hundreds into the queue at once, and
    # a run that decides a few hundred documents unattended should be a deliberate act
    # repeated, not one call with no upper bound.
    APPROVAL_AGENT_BATCH_LIMIT: int = 50
    # Comma-separated, same convention as CORS_ORIGINS (see cors_origin_list below) --
    # departments an operator has explicitly decided count as "high risk" for approval
    # scoping (ApprovalRule.risk_tiers). Empty by default: this is a hook an operator
    # opts into, not a judgment the code makes on their behalf about which departments
    # are sensitive. A draft's risk tier is otherwise "standard" (see
    # approval_agent.compute_draft_risk_tier), so an empty list changes nothing.
    APPROVAL_AGENT_HIGH_RISK_DEPARTMENTS: str = ""
    # How often the beat schedule calls approval_agent.run() for every company that has
    # an active rule. Without this the agent only ever ran when something called
    # POST /governance/approval-agent/run by hand -- rules existed and were correctly
    # scoped, but nothing applied them to a draft that arrived after the last manual run.
    APPROVAL_AGENT_RUN_INTERVAL_SECONDS: int = 300
    # Factory reset: who may erase every non-identity table, and whether the
    # endpoint exists at all.
    #
    # Both are REQUIRED. The allowlist alone would leave a permanently loaded gun
    # in every deployment that happens to have that account, so the capability is
    # off unless an operator turns it on in the API environment — somewhere a
    # compromised session cannot reach.
    #
    # Comma-separated so a second operator can be added without a code change.
    # Matched case-insensitively against the CURRENT verified account, never
    # against a token claim.
    FACTORY_RESET_ENABLED: bool = False
    FACTORY_RESET_ALLOWED_EMAILS: str = ""
    SYSTEM_DATA_OWNER_EMAIL: str | None = None
    DEFAULT_LANGUAGE: str = "vi"
    REVIEW_SLA_DAYS: int = 3
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GOOGLE_REDIRECT_URI: str | None = None
    CONNECTOR_WEBHOOK_BASE_URL: str | None = None
    ALLOWED_EMAIL_DOMAINS: str = ""
    ALLOW_SELF_REGISTRATION: bool = True

    # The first global administrator, created at API startup when the deployment has none
    # — see src/domain/admin_bootstrap.py. A migrated database has no users and there is
    # no signup route, so without this a fresh deployment is unreachable by anyone.
    #
    # BOOTSTRAP_ADMIN_PASSWORD keeps its development default only outside production;
    # validate_production refuses to start with it, so a public deployment supplies its
    # own or does not come up.
    BOOTSTRAP_ADMIN_ENABLED: bool = True
    BOOTSTRAP_ADMIN_EMAIL: str = "admin@qnsc.vn"
    BOOTSTRAP_ADMIN_NAME: str = "Admin"
    BOOTSTRAP_ADMIN_PASSWORD: str = DEVELOPMENT_DEFAULT_PASSWORD
    PADDLEOCR_LANG: str = "en"
    MARKITDOWN_ENABLED: bool = True
    SOURCE_STORAGE_PATH: str = "/app/storage/sources"
    # MVP-1 stores originals in a private Cloudflare R2 bucket.  There is no
    # local-disk fallback because a local fallback would make deployment
    # topology part of the authorization boundary.
    SOURCE_STORAGE_BACKEND: str = "r2"
    SOURCE_STORAGE_BUCKET: str | None = None
    SOURCE_STORAGE_PREFIX: str = "qnsc-sources"
    SOURCE_ORPHAN_GRACE_HOURS: int = 24
    AWS_REGION: str | None = None
    S3_ENDPOINT_URL: str | None = None
    R2_ACCOUNT_ID: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None
    CONNECTOR_ROOT_PATH: str = "/app/storage/connectors"
    MAX_SOURCE_UPLOAD_BYTES: int = 25 * 1024 * 1024
    MAX_CONNECTOR_API_RESPONSE_BYTES: int = 5 * 1024 * 1024
    MAX_CONNECTOR_FILES: int = 2_000
    METRICS_RETENTION_DAYS: int = 30
    MAX_SOURCE_PAGES: int = 500
    MAX_SOURCE_TEXT_CHARS: int = 2_000_000
    MAX_SOURCE_IMAGE_PIXELS: int = 40_000_000

    # Near-duplicate detection at upload, and both numbers are cost controls rather
    # than tuning knobs — the SCORE is insensitive to them.
    #
    # Comparing whole documents was never viable. SequenceMatcher is O(n*m): measured
    # on a developer machine, one 20,000-character pair costs 8.25 s and the cost rises
    # roughly 8x per doubling, and MAX_SOURCE_TEXT_CHARS admits 2,000,000. That ran once
    # PER EXISTING ARTICLE, on the event loop.
    #
    # 2000 characters is where the cost curve turns: 0.073 s per comparison against
    # 0.234 s at 3000 and 1.0 s at 6000, while a 95%-identical pair scores 0.974 here
    # and 0.968 at 6000. Nothing is bought by comparing more.
    SIMILARITY_COMPARE_CHARS: int = 2_000
    # Token overlap is O(n) and ranks candidates cheaply, so the O(n*m) comparison is
    # spent only on the most plausible ones. Without this the total still grows with the
    # corpus and the timeout returns once the knowledge base is large enough; 50 is a
    # wide margin over the 5 matches ever returned.
    SIMILARITY_MAX_SEQUENCE_COMPARISONS: int = 50
    MAX_SOURCE_UNCOMPRESSED_BYTES: int = 100_000_000
    MAX_SOURCE_ARCHIVE_FILES: int = 2_000
    MALWARE_SCAN_ENABLED: bool = False
    MALWARE_SCANNER_COMMAND: str = "clamdscan"
    MALWARE_SCANNER_HOST: str | None = None
    MALWARE_SCANNER_PORT: int = 3310
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    def _compose_dsn(self, user: str, password: str) -> str:
        """Build a SQLAlchemy asyncpg URL from the connection parts.

        The password is percent-encoded: a generated credential routinely contains
        ``@``, ``/`` or ``:``, each of which terminates a different component of a URL,
        so pasting one in raw yields a URL that parses cleanly into the wrong host,
        port or database.
        """
        return (
            f"postgresql+asyncpg://{quote(user, safe='')}:{quote(password, safe='')}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    def model_post_init(self, __context: Any) -> None:
        # Compose the database URLs from parts when a host is supplied and no explicit
        # URL was given. `model_fields_set` is what makes "explicit" mean explicit:
        # DATABASE_URL has a non-empty default, so testing its truthiness would treat
        # the localhost default as a deliberate choice and ignore the injected parts.
        if self.DATABASE_HOST:
            if (
                "DATABASE_URL" not in self.model_fields_set
                and self.DATABASE_USER
                and self.DATABASE_PASSWORD
            ):
                self.DATABASE_URL = self._compose_dsn(
                    self.DATABASE_USER, self.DATABASE_PASSWORD
                )
            if (
                not self.MIGRATION_DATABASE_URL
                and self.MIGRATION_DATABASE_USER
                and self.MIGRATION_DATABASE_PASSWORD
            ):
                self.MIGRATION_DATABASE_URL = self._compose_dsn(
                    self.MIGRATION_DATABASE_USER, self.MIGRATION_DATABASE_PASSWORD
                )

        if self.EMBEDDING_DIMENSION is None:
            model = self.EMBEDDING_MODEL.lower()
            if "bge-m3" in model:
                self.EMBEDDING_DIMENSION = 1024
            elif "minilm" in model:
                self.EMBEDDING_DIMENSION = 384
            elif "e5-small" in model:
                # multilingual-e5-small is 384-wide, same as MiniLM — so switching
                # between them needs no pgvector column or HNSW rebuild.
                self.EMBEDDING_DIMENSION = 384
            elif "e5-base" in model:
                self.EMBEDDING_DIMENSION = 768
            elif "e5-large" in model:
                # multilingual-e5-large(-instruct) is XLM-R-large-based, 1024-wide —
                # NOT 768 like e5-base. Collapsing the two here previously produced a
                # silently-wrong derived width for e5-large; base.py's finalise() width
                # check would catch the mismatch at runtime, but don't rely on that.
                self.EMBEDDING_DIMENSION = 1024
            elif "text-embedding-3-small" in model or "ada-002" in model:
                self.EMBEDDING_DIMENSION = 1536
            elif "gemini-embedding" in model or "text-embedding-004" in model:
                # 768, NOT the provider's 3072 default: pgvector's HNSW index refuses to
                # build above 2000 dimensions, and the adapter asks for this width
                # explicitly via outputDimensionality.
                self.EMBEDDING_DIMENSION = 768
            else:
                self.EMBEDDING_DIMENSION = 1024

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    @property
    def approval_agent_high_risk_department_list(self) -> list[str]:
        return [
            dept.strip().lower()
            for dept in self.APPROVAL_AGENT_HIGH_RISK_DEPARTMENTS.split(",")
            if dept.strip()
        ]

    @property
    def microsoft_connector_auth_mode(self) -> str:
        """The Microsoft connector auth mode, normalised in ONE place.

        validate_production compared this stripped while every runtime site compared it
        with `.lower()` alone, so a value carrying stray whitespace — which is what an
        environment variable set from a Terraform join or a copied YAML line looks like —
        passed the boot check and then behaved as "delegated" everywhere afterwards: the
        app-only worker asked for a delegated refresh token it never had and every sync
        failed with "not authorized".
        """
        return (self.MICROSOFT_CONNECTOR_AUTH_MODE or "").strip().lower()

    def validate_production(self) -> None:
        environment = self.ENVIRONMENT.lower()
        if environment in {"development", "dev", "local"}:
            return
        # Secret-material checks apply to ANY non-development environment. The
        # committed default SECRET_KEY signs access/refresh JWTs and (without a
        # DEK) encrypts stored connector tokens; a staging/uat/demo deployment
        # with it is a production incident waiting to happen, so the gate must
        # not key off the exact string "production".
        if (
            self.SECRET_KEY in {"", "super-secret-key-change-in-production"}
            or len(self.SECRET_KEY) < 32
        ):
            raise RuntimeError(
                "SECRET_KEY must be a strong, externally supplied value outside development"
            )
        if not self.DATA_ENCRYPTION_KEY or len(self.DATA_ENCRYPTION_KEY) < 32:
            raise RuntimeError(
                "DATA_ENCRYPTION_KEY must be a separate, strong externally supplied value outside development"
            )
        if any(
            len(key.strip()) < 32
            for key in self.PREVIOUS_DATA_ENCRYPTION_KEYS.split(",")
            if key.strip()
        ):
            raise RuntimeError(
                "PREVIOUS_DATA_ENCRYPTION_KEYS entries must each be at least 32 characters"
            )
        if environment not in {"production", "prod"}:
            return
        if self.AUTO_CREATE_SCHEMA:
            raise RuntimeError(
                "AUTO_CREATE_SCHEMA must be false in production; use Alembic migrations"
            )
        if self.ALLOW_SELF_REGISTRATION:
            raise RuntimeError("ALLOW_SELF_REGISTRATION must be false in production")
        if self.ENABLE_API_DOCS:
            raise RuntimeError("ENABLE_API_DOCS must be false in production")
        cors_origins = self.cors_origin_list
        if not cors_origins or "*" in cors_origins:
            raise RuntimeError(
                "CORS_ORIGINS must contain explicit frontend origins in production"
            )

        def normalized_https_origin(value: str, setting_name: str) -> str:
            try:
                parsed = urlparse(value)
                port = parsed.port
            except ValueError as exc:
                raise RuntimeError(f"{setting_name} contains an invalid port") from exc
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
                or parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
            ):
                raise RuntimeError(
                    f"{setting_name} must contain explicit HTTPS application origins in production"
                )
            host = parsed.hostname.lower()
            return f"https://{host}" + (f":{port}" if port and port != 443 else "")

        frontend_origin = normalized_https_origin(self.FRONTEND_URL, "FRONTEND_URL")
        if not frontend_origin:
            raise RuntimeError(
                "FRONTEND_URL must be an explicit HTTPS application origin in production"
            )
        normalized_cors = {
            normalized_https_origin(origin, "CORS_ORIGINS") for origin in cors_origins
        }
        if frontend_origin not in normalized_cors:
            raise RuntimeError("CORS_ORIGINS must include FRONTEND_URL in production")

        def validated_https_url(value: str, setting_name: str) -> None:
            try:
                parsed = urlparse(value)
            except ValueError as exc:
                raise RuntimeError(f"{setting_name} contains an invalid URL") from exc
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
            ):
                raise RuntimeError(
                    f"{setting_name} must use a public HTTPS URL in production"
                )

        for setting_name, value in (
            ("MICROSOFT_REDIRECT_URI", self.MICROSOFT_REDIRECT_URI),
            ("MICROSOFT_LOGIN_REDIRECT_URI", self.MICROSOFT_LOGIN_REDIRECT_URI),
            ("GOOGLE_REDIRECT_URI", self.GOOGLE_REDIRECT_URI),
            ("CONNECTOR_WEBHOOK_BASE_URL", self.CONNECTOR_WEBHOOK_BASE_URL),
        ):
            if value:
                validated_https_url(value, setting_name)
        connector_auth_mode = self.microsoft_connector_auth_mode
        if connector_auth_mode not in {"delegated", "application"}:
            raise RuntimeError(
                "MICROSOFT_CONNECTOR_AUTH_MODE must be delegated or application"
            )
        if connector_auth_mode == "application" and not self.CONNECTOR_WEBHOOK_BASE_URL:
            raise RuntimeError(
                "CONNECTOR_WEBHOOK_BASE_URL is required for Microsoft application mode"
            )
        storage_backend = (self.SOURCE_STORAGE_BACKEND or "").strip().lower()
        storage_bucket = (self.SOURCE_STORAGE_BUCKET or "").strip()
        storage_endpoint = (self.S3_ENDPOINT_URL or "").strip()
        storage_account = (self.R2_ACCOUNT_ID or "").strip()
        storage_access_key = (self.R2_ACCESS_KEY_ID or "").strip()
        storage_secret_key = (self.R2_SECRET_ACCESS_KEY or "").strip()
        if storage_backend not in {"r2", "cloudflare_r2"}:
            raise RuntimeError(
                "SOURCE_STORAGE_BACKEND must be Cloudflare R2 in production"
            )
        if not storage_bucket:
            raise RuntimeError("SOURCE_STORAGE_BUCKET is required for Cloudflare R2")
        if not (storage_endpoint or storage_account):
            raise RuntimeError(
                "R2_ACCOUNT_ID or S3_ENDPOINT_URL is required for Cloudflare R2"
            )
        if not (storage_access_key and storage_secret_key):
            raise RuntimeError(
                "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required for Cloudflare R2"
            )
        if storage_endpoint:
            validated_https_url(storage_endpoint, "S3_ENDPOINT_URL")
            if not is_cloudflare_r2_endpoint(storage_endpoint):
                raise RuntimeError("S3_ENDPOINT_URL must be a Cloudflare R2 endpoint")
        if (
            storage_account
            and storage_account.lower().startswith(("http://", "https://"))
            and not is_cloudflare_r2_endpoint(storage_account)
        ):
            raise RuntimeError(
                "R2_ACCOUNT_ID endpoint must be a Cloudflare R2 endpoint"
            )
        if not self.MALWARE_SCAN_ENABLED:
            raise RuntimeError("MALWARE_SCAN_ENABLED must be true in production")
        if not self.MALWARE_SCANNER_HOST:
            raise RuntimeError("MALWARE_SCANNER_HOST is required in production")
        microsoft_required = (
            self.MICROSOFT_CLIENT_ID,
            self.MICROSOFT_CLIENT_SECRET,
            self.MICROSOFT_TENANT_ID,
            self.MICROSOFT_LOGIN_REDIRECT_URI,
        )
        if self.microsoft_connector_auth_mode == "delegated":
            microsoft_required = (*microsoft_required, self.MICROSOFT_REDIRECT_URI)
        if not all(value and value.strip() for value in microsoft_required):
            raise RuntimeError(
                "Microsoft Entra client, tenant, login redirect, and delegated connector redirect settings are required in production"
            )
        if not re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            self.MICROSOFT_TENANT_ID.strip(),
        ):
            raise RuntimeError(
                "MICROSOFT_TENANT_ID must be a specific Entra tenant GUID in production"
            )

        # LAST on purpose. This is the newest check, and putting it earlier made it
        # shadow every one above: a settings object left deliberately broken to exercise
        # the CORS or R2 rule would fail on the bootstrap password instead, and the rule
        # actually under test would go unverified.
        if self.BOOTSTRAP_ADMIN_ENABLED:
            # The account this creates is a GLOBAL administrator: it bypasses tenant RLS
            # and can read every company's data. A password that ships in this repository
            # would make that reachable by anyone who can read GitHub and find the host.
            if self.BOOTSTRAP_ADMIN_PASSWORD == DEVELOPMENT_DEFAULT_PASSWORD:
                raise RuntimeError(
                    "BOOTSTRAP_ADMIN_PASSWORD must be set in production; it is still the "
                    "development default. Set BOOTSTRAP_ADMIN_ENABLED=false if this "
                    "deployment provisions its administrator some other way."
                )
            if len(self.BOOTSTRAP_ADMIN_PASSWORD) < 12:
                raise RuntimeError(
                    "BOOTSTRAP_ADMIN_PASSWORD must be at least 12 characters in production"
                )
            if "@" not in self.BOOTSTRAP_ADMIN_EMAIL:
                raise RuntimeError(
                    "BOOTSTRAP_ADMIN_EMAIL must be an email address in production"
                )


settings = Settings()
