from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any
from urllib.parse import urlparse
import re


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
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173"
    FRONTEND_URL: str = "http://localhost:5173"
    # Schema lifecycle is owned exclusively by Alembic migrations.
    AUTO_CREATE_SCHEMA: bool = False
    JOB_MODE: str = "inline"
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
    GEMINI_MAX_OUTPUT_TOKENS: int = 8192
    LLM_TIMEOUT_SECONDS: float = 90.0
    # Local, in-process, and deliberately so: no embedding text leaves the deployment and
    # no third-party key gates indexing or search.
    #
    # It is paid for in image size and memory. The API embeds the search QUERY on every
    # search, so the fp32 ONNX export lives in the api image as well as the worker's —
    # ~2.2 GB each, and no torch anywhere (PR #54): the `ml` group is a local-checkout
    # reference, not a shipped dependency. Choosing a hosted model instead
    # (gemini-embedding-001, text-embedding-3-small) is a one-line change here plus the
    # migration below, and src/lib/embeddings.py already carries that path.
    #
    # This value fixes EMBEDDING_DIMENSION, which fixes the pgvector column width and the
    # HNSW index AT MIGRATION TIME. Changing it later needs a migration and a full
    # re-embed: a query and a chunk embedded by different models are points in unrelated
    # spaces, and their distance is meaningless rather than merely wrong.
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_VERSION: str = "bge-m3-v1"
    CHUNKING_VERSION: str = "v2-structure-aware"
    EMBEDDING_DIMENSION: int | None = None
    LLM_MODEL: str = "gemma-4-26b-a4b-it"
    RESTRUCTURE_ENABLED: bool = True
    RESTRUCTURE_MODEL: str | None = None
    RESTRUCTURE_MAX_CHARS: int = 60000
    # Formatting is an optional enhancement. Keep review responsive and use
    # the lossless local fallback when the configured provider is slow.
    RESTRUCTURE_TIMEOUT_SECONDS: float = 120.0
    RESTRUCTURE_MAX_OUTPUT_TOKENS: int = 16384
    RESTRUCTURE_NUMERIC_COVERAGE_THRESHOLD: float = 0.90
    AI_RATE_LIMIT_PER_MINUTE: int = 30
    VECTOR_DISTANCE_THRESHOLD: float = 0.45
    RAG_MIN_RELEVANCE_SCORE: float = 0.12
    RAG_MIN_CONTEXT_SCORE: float = 0.35
    RAG_CANDIDATE_POOL_SIZE: int = 48
    RAG_RERANK_LIMIT: int = 16
    RAG_MAX_CONTEXT_PARENTS: int = 8
    RAG_CONTEXT_MAX_CHARS: int = 14000
    RAG_CONTEXT_MAX_TOKENS: int = 3500
    RAG_PARENT_CONTEXT_CHARS: int = 2400
    RAG_MAX_PARENTS_PER_ARTICLE: int = 3
    PROMPT_VERSION: str = "v2.0-grounded-extended-sections"
    RETRIEVAL_VERSION: str = "v2-parent-budget-confidence"
    RERANKER_VERSION: str = "v1.2-definition-aware"
    RAG_ENABLE_EXTENDED_SECTION: bool = True
    RAG_CACHE_EXTENDED_SECTION: bool = False
    RAG_ALLOW_EXTENDED_ON_REFUSAL: bool = False
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
            if "DATABASE_URL" not in self.model_fields_set and self.DATABASE_USER and self.DATABASE_PASSWORD:
                self.DATABASE_URL = self._compose_dsn(self.DATABASE_USER, self.DATABASE_PASSWORD)
            if not self.MIGRATION_DATABASE_URL and self.MIGRATION_DATABASE_USER and self.MIGRATION_DATABASE_PASSWORD:
                self.MIGRATION_DATABASE_URL = self._compose_dsn(
                    self.MIGRATION_DATABASE_USER, self.MIGRATION_DATABASE_PASSWORD
                )

        if self.EMBEDDING_DIMENSION is None:
            model = self.EMBEDDING_MODEL.lower()
            if "bge-m3" in model:
                self.EMBEDDING_DIMENSION = 1024
            elif "minilm" in model:
                self.EMBEDDING_DIMENSION = 384
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

    def validate_production(self) -> None:
        if self.ENVIRONMENT.lower() not in {"production", "prod"}:
            return
        if (
            self.SECRET_KEY in {"", "super-secret-key-change-in-production"}
            or len(self.SECRET_KEY) < 32
        ):
            raise RuntimeError(
                "SECRET_KEY must be a strong, externally supplied value in production"
            )
        if not self.DATA_ENCRYPTION_KEY or len(self.DATA_ENCRYPTION_KEY) < 32:
            raise RuntimeError(
                "DATA_ENCRYPTION_KEY must be a separate, strong externally supplied value in production"
            )
        if any(
            len(key.strip()) < 32
            for key in self.PREVIOUS_DATA_ENCRYPTION_KEYS.split(",")
            if key.strip()
        ):
            raise RuntimeError(
                "PREVIOUS_DATA_ENCRYPTION_KEYS entries must each be at least 32 characters"
            )
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
        if not all(
            value and value.strip()
            for value in (
                self.MICROSOFT_CLIENT_ID,
                self.MICROSOFT_CLIENT_SECRET,
                self.MICROSOFT_TENANT_ID,
                self.MICROSOFT_REDIRECT_URI,
                self.MICROSOFT_LOGIN_REDIRECT_URI,
            )
        ):
            raise RuntimeError(
                "Microsoft Entra client, tenant, and redirect settings are required in production"
            )
        if not re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            self.MICROSOFT_TENANT_ID.strip(),
        ):
            raise RuntimeError(
                "MICROSOFT_TENANT_ID must be a specific Entra tenant GUID in production"
            )


settings = Settings()
