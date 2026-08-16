from contextlib import asynccontextmanager
import asyncio
import re
import time
import uuid
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from src.core.config import settings
from src.api.routers import (
    auth,
    articles,
    search,
    ai,
    interactions,
    governance,
    meta,
    connectors,
    knowledge,
    llm,
    notifications,
)
from src.api.deps import SessionLocal, engine, init_db, set_database_context
from src.domain.events import event_bus
from src.models.article import Article
from src.models.chunk import ArticleChunk
from src.models.ops import ApiRequestMetric
from src.core.metrics import record_request, prometheus_text
from src.core.tracing import configure_tracing, get_tracer, trace
from src.lib.embeddings import OnnxEmbeddingModelSingleton
import structlog

logger = structlog.get_logger()
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


async def _preload_embedding_model() -> None:
    """Warm ONNX weights without preventing the API from becoming healthy."""
    try:
        await asyncio.to_thread(OnnxEmbeddingModelSingleton.get_model)
    except Exception as exc:
        logger.warning(
            "Embedding model preload failed; keyword search remains available",
            error=str(exc),
        )


def _request_id_from(request) -> str:
    supplied = request.headers.get("X-Request-ID", "")
    return supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())


def _metric_path_for(request) -> str:
    """Use the resolved route template to keep metrics cardinality bounded."""
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else "/unmatched"


async def record_request_metric(
    request_id: str, method: str, path: str, status_code: int, duration_ms: float
) -> None:
    record_request(method, path, status_code, duration_ms)
    try:
        async with SessionLocal() as db:
            db.add(
                ApiRequestMetric(
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
            )
            await db.commit()
    except Exception as exc:
        logger.warning(
            "Could not persist API request metric",
            request_id=request_id,
            error=str(exc),
        )


async def reconcile_published_indexes() -> None:
    """Repair index state from before persisted indexing status was introduced."""
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        result = await db.execute(
            select(
                Article.id,
                Article.index_status,
                func.count(ArticleChunk.id).label("chunk_count"),
            )
            .outerjoin(ArticleChunk, ArticleChunk.article_id == Article.id)
            .where(Article.status == "published")
            .group_by(Article.id, Article.index_status)
        )
        rows = result.all()
        ready_ids = [
            article_id
            for article_id, index_status, chunk_count in rows
            if index_status == "pending" and chunk_count > 0
        ]
        missing_ids = [
            article_id for article_id, _, chunk_count in rows if chunk_count == 0
        ]

        for article_id in ready_ids:
            await db.execute(
                Article.__table__.update()
                .where(Article.id == article_id)
                .values(index_status="ready", index_error=None)
            )
        if ready_ids:
            await db.commit()

    for article_id in missing_ids:
        await event_bus.publish("ArticlePublished", {"article_id": str(article_id)})

    if ready_ids or missing_ids:
        logger.info(
            "Published index reconciliation completed",
            marked_ready=len(ready_ids),
            queued_for_indexing=len(missing_ids),
        )


async def initialize_resources() -> None:
    logger.info("Starting up and initializing database...")
    settings.validate_production()
    configure_tracing()
    for attempt in range(1, 4):
        try:
            # PostgreSQL can need a few seconds to finish crash recovery after
            # Docker restarts. Migrations run in the deployment entrypoint;
            # application startup only verifies/bootstraps runtime data.
            await asyncio.wait_for(init_db(), timeout=60)
            logger.info("Database initialized successfully", attempt=attempt)
            await reconcile_published_indexes()
            break
        except Exception as exc:
            if attempt == 3:
                logger.exception(
                    "Failed to initialize database", error=str(exc), attempts=attempt
                )
                raise
            logger.warning(
                "Database initialization failed; retrying",
                error=str(exc),
                attempt=attempt,
                retry_in_seconds=attempt * 2,
            )
            # Drop connections that may have been interrupted during recovery
            # before the next attempt obtains a fresh asyncpg connection.
            await engine.dispose()
            await asyncio.sleep(attempt * 2)

    # Do not make API readiness depend on a multi-second ONNX session load.
    # The singleton still avoids repeated loading when the first embedding is
    # requested, while the background warmup normally completes before then.
    if settings.OPENAI_API_KEY != "mock" and settings.EMBEDDING_MODEL != "mock":
        asyncio.create_task(_preload_embedding_model())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Database migrations/readiness must complete before serving traffic.
    await initialize_resources()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=(
        f"{settings.API_V1_STR}/openapi.json" if settings.ENABLE_API_DOCS else None
    ),
    docs_url="/docs" if settings.ENABLE_API_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
    lifespan=lifespan,
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    started = time.perf_counter()
    request_id = _request_id_from(request)
    request.state.request_id = request_id
    logger.info(
        "API request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    tracer = get_tracer()
    span_context = tracer.start_as_current_span("http.request") if tracer else None
    try:
        if span_context:
            span_context.__enter__()
        if tracer:
            current_span = trace.get_current_span() if trace else None
            if current_span:
                current_span.set_attribute("http.method", request.method)
                current_span.set_attribute("http.route", request.url.path)
        response = await call_next(request)
        logger.info(
            "API request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            request_id=request_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        await record_request_metric(
            request_id,
            request.method,
            _metric_path_for(request),
            response.status_code,
            round((time.perf_counter() - started) * 1000, 2),
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        if request.url.path.startswith(settings.API_V1_STR + "/"):
            # API responses can contain private article and AI content.  Do
            # not let browsers or shared proxies retain them after logout.
            response.headers["Cache-Control"] = "no-store, private"
        if settings.ENVIRONMENT.lower() in {"production", "prod"}:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        logger.error(
            "API request failed",
            method=request.method,
            path=request.url.path,
            request_id=request_id,
            error=str(exc),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        await record_request_metric(
            request_id,
            request.method,
            _metric_path_for(request),
            500,
            round((time.perf_counter() - started) * 1000, 2),
        )
        raise
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


# Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(
    articles.router, prefix=f"{settings.API_V1_STR}/articles", tags=["articles"]
)
app.include_router(
    search.router, prefix=f"{settings.API_V1_STR}/search", tags=["search"]
)
app.include_router(ai.router, prefix=f"{settings.API_V1_STR}/ai", tags=["ai"])
app.include_router(
    interactions.router,
    prefix=f"{settings.API_V1_STR}/interactions",
    tags=["interactions"],
)
app.include_router(
    governance.router, prefix=f"{settings.API_V1_STR}/governance", tags=["governance"]
)
app.include_router(meta.router, prefix=f"{settings.API_V1_STR}/meta", tags=["meta"])
app.include_router(
    connectors.router, prefix=f"{settings.API_V1_STR}/connectors", tags=["connectors"]
)
app.include_router(
    knowledge.router, prefix=f"{settings.API_V1_STR}/knowledge", tags=["knowledge"]
)
app.include_router(llm.router, prefix=f"{settings.API_V1_STR}/admin/llm", tags=["llm"])
app.include_router(
    notifications.router,
    prefix=f"{settings.API_V1_STR}/notifications",
    tags=["notifications"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to QNSC Knowledge Base API", "docs": "/docs"}


@app.get("/health/live", tags=["system"])
async def health_live():
    return {"status": "alive"}


@app.get("/metrics", include_in_schema=False, response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(prometheus_text(), media_type="text/plain; version=0.0.4")


@app.get("/health/ready", tags=["system"])
async def health_ready():
    from sqlalchemy import text
    from src.api.deps import engine

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        from redis.asyncio import Redis

        redis = Redis.from_url(
            settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2
        )
        try:
            await redis.ping()
        finally:
            await redis.aclose()
        return {
            "status": "ready",
            "database": "ok",
            "redis": "ok",
            "job_mode": settings.JOB_MODE,
        }
    except Exception as exc:
        logger.error("Readiness check failed", error=str(exc))
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503, detail={"status": "not_ready", "database": "unavailable"}
        )
