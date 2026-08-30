from contextlib import asynccontextmanager
import asyncio
import re
import time
import traceback
import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import func, select, text
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
from src.lib.embeddings import warm_up as warm_up_embeddings
import structlog

logger = structlog.get_logger()
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


async def _preload_embedding_model() -> None:
    """Warm the configured embedding backend without blocking API readiness."""
    try:
        await asyncio.to_thread(warm_up_embeddings)
    except Exception as exc:
        logger.warning(
            "Embedding model preload failed; keyword search remains available",
            error=str(exc),
        )


async def verify_embedding_column_width() -> None:
    """Report a pgvector column that no longer matches EMBEDDING_DIMENSION.

    The column width is DERIVED from EMBEDDING_MODEL but enforced by a one-shot Alembic
    revision, and Alembic never re-runs an applied revision. So changing the model in
    infra silently leaves the old width behind, and the only symptom is every article
    turning "Search index: failed" one at a time, with the real cause — a DataError deep
    in the worker — visible nowhere near the setting that caused it. That has now
    happened twice: bge-m3 against a 768 column, and MiniLM against a 1024 one.

    Logged, not raised. A width mismatch breaks indexing, not serving, and taking the
    whole API down over it would turn degraded search into an outage. The fix is a new
    realign revision; this exists so the next person reads one line instead of a
    traceback.
    """
    from sqlalchemy import text

    from src.api.deps import engine

    try:
        async with engine.connect() as connection:
            current = (
                await connection.execute(
                    text(
                        "SELECT a.atttypmod FROM pg_attribute a "
                        "JOIN pg_class c ON c.oid = a.attrelid "
                        "JOIN pg_type t ON t.oid = a.atttypid "
                        "WHERE c.relname = 'article_chunks' "
                        "AND a.attname = 'embedding' AND t.typname = 'vector'"
                    )
                )
            ).scalar()
    except Exception as exc:  # pragma: no cover - diagnostics must never block boot
        logger.warning("Could not verify embedding column width", error=str(exc))
        return

    if current is None or current == settings.EMBEDDING_DIMENSION:
        return
    logger.error(
        "Embedding column width does not match EMBEDDING_DIMENSION; indexing will fail "
        "on every article until a realign migration runs",
        column_dimension=current,
        expected_dimension=settings.EMBEDDING_DIMENSION,
        embedding_model=settings.EMBEDDING_MODEL,
    )


async def verify_rls_policies() -> None:
    """Fail fast when production RLS policies are missing.

    The tenant-RLS migrations only create policies when ENABLE_RLS is set in
    the environment *at migration time*. If an operator runs Alembic without
    it, the revision is recorded as applied with no policies created and
    re-running does nothing. The application-level SQL predicates still
    enforce isolation, but RLS is the second layer production relies on, so
    a production startup must refuse to serve without it.
    """
    if settings.ENVIRONMENT.lower() not in {"production", "prod"}:
        return
    async with SessionLocal() as db:
        await set_database_context(db, None, True)
        count = (
            await db.execute(
                text(
                    "SELECT count(*) FROM pg_policies "
                    "WHERE schemaname = 'public' "
                    "AND tablename = 'articles' AND policyname = 'tenant_articles'"
                )
            )
        ).scalar_one()
    if count == 0:
        raise RuntimeError(
            "Tenant RLS policies are missing. The RLS migrations only apply when "
            "ENABLE_RLS=true is set in the environment when Alembic runs; the "
            "revision was likely recorded without them. Re-run migrations with "
            "ENABLE_RLS=true against a database where the RLS revisions have not "
            "yet been applied, or restore from a properly migrated backup."
        )


def _request_id_from(request) -> str:
    supplied = request.headers.get("X-Request-ID", "")
    return supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())


def _metric_path_for(request) -> str:
    """Use the resolved route template to keep metrics cardinality bounded."""
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else "/unmatched"


#: A traceback is unbounded and this row is written on the request path, so the stored
#: detail is capped. The innermost frames are the ones that name the actual fault.
ERROR_DETAIL_MAX_CHARS = 4000


async def record_request_metric(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    exc: BaseException | None = None,
) -> None:
    record_request(method, path, status_code, duration_ms)
    error_type: str | None = None
    error_detail: str | None = None
    if exc is not None:
        error_type = type(exc).__name__
        # The message first, so it survives truncation even when the traceback is deep.
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        error_detail = f"{exc}\n\n{formatted}"[:ERROR_DETAIL_MAX_CHARS]
    try:
        async with SessionLocal() as db:
            db.add(
                ApiRequestMetric(
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    error_type=error_type,
                    error_detail=error_detail,
                )
            )
            await db.commit()
    except Exception as persist_error:
        # Named distinctly from the `exc` parameter above: shadowing it here would mean a
        # storage failure silently overwrote the exception we were trying to record.
        logger.warning(
            "Could not persist API request metric",
            request_id=request_id,
            error=str(persist_error),
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
    # Structured JSON logs to stdout (the production-hardening doc's contract
    # with the log aggregation pipeline).
    from src.lib.observability import setup_logging

    setup_logging()
    for attempt in range(1, 4):
        try:
            # PostgreSQL can need a few seconds to finish crash recovery after
            # Docker restarts. Migrations run in the deployment entrypoint;
            # application startup only verifies/bootstraps runtime data.
            await asyncio.wait_for(init_db(), timeout=60)
            logger.info("Database initialized successfully", attempt=attempt)
            await verify_rls_policies()
            await verify_embedding_column_width()
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


async def _inline_outbox_recovery_loop() -> None:
    """Inline-mode replacement for the Celery beat replay task.

    Retries outbox events that failed or were orphaned by a process restart.
    Without it, inline deployments never replay events and the outbox grows
    forever.
    """
    while True:
        await asyncio.sleep(settings.OUTBOX_RECOVERY_INTERVAL_SECONDS)
        try:
            recovered = await event_bus.recover_outbox_once()
            if recovered:
                logger.info("Inline outbox recovery pass", recovered=recovered)
            # Terminal outbox rows have no replay value; bound table growth
            # the same way prune_operational_metrics bounds telemetry.
            from datetime import datetime, timedelta
            from sqlalchemy import delete as sa_delete

            from src.api.deps import SessionLocal
            from src.models.ops import OutboxEvent

            cutoff = datetime.utcnow() - timedelta(
                days=settings.METRICS_RETENTION_DAYS
            )
            async with SessionLocal() as db:
                result = await db.execute(
                    sa_delete(OutboxEvent).where(
                        OutboxEvent.status.in_(["completed", "dead"]),
                        OutboxEvent.created_at < cutoff,
                    )
                )
                await db.commit()
            if result.rowcount:
                logger.info(
                    "Pruned terminal outbox events", pruned=result.rowcount
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inline outbox recovery pass failed")


async def _inline_connector_sync_dispatch_loop() -> None:
    """Durably dispatch queued connector work when Celery is not enabled."""

    from src.domain.sync_queue import claim_sync_request, recover_stale_sync_requests
    from src.domain.sync_queue import enqueue_connector_sync
    from src.models.connectors import SourceScope, SyncCursor, SyncRequest
    from src.models.ops import Connector

    while True:
        await asyncio.sleep(settings.CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS)
        try:
            async with SessionLocal() as db:
                await set_database_context(db, None, True)
                await recover_stale_sync_requests(db)
                # Compared as datetimes. `utcnow().timestamp()` reads a naive UTC value
                # as local time, so the arithmetic shifted by the UTC offset and moved
                # again across a DST boundary.
                reconcile_cutoff = datetime.utcnow() - timedelta(
                    minutes=settings.CONNECTOR_RECONCILE_INTERVAL_MINUTES
                )
                connectors = (
                    await db.execute(
                        select(Connector).where(
                            Connector.system.in_(["sharepoint", "google_drive"]),
                            Connector.status.in_(["active", "error"]),
                        )
                    )
                ).scalars().all()
                for connector in connectors:
                    scopes = (
                        await db.execute(
                            select(SourceScope).where(
                                SourceScope.connector_id == connector.id,
                                SourceScope.selected.is_(True),
                            )
                        )
                    ).scalars().all()
                    for scope in scopes:
                        cursor = (
                            await db.execute(
                                select(SyncCursor).where(
                                    SyncCursor.connector_id == connector.id,
                                    SyncCursor.scope_id == scope.id,
                                )
                            )
                        ).scalar_one_or_none()
                        if cursor and cursor.last_reconcile_at and cursor.last_reconcile_at > reconcile_cutoff:
                            continue
                        if cursor is None:
                            cursor = SyncCursor(
                                connector_id=connector.id,
                                scope_id=scope.id,
                                cursor_type="delta" if connector.system == "sharepoint" else "changes",
                            )
                            db.add(cursor)
                        cursor.full_sync_required = True
                        cursor.status = "reconcile"
                        await enqueue_connector_sync(
                            db,
                            connector.id,
                            scope_id=scope.id,
                            reason="reconcile",
                            requested_by=connector.created_by,
                            priority=40,
                        )
                await db.commit()
                request_ids = (
                    await db.execute(
                        select(SyncRequest.id)
                        .where(
                            SyncRequest.status == "queued",
                            SyncRequest.available_at <= datetime.utcnow(),
                        )
                        .order_by(SyncRequest.priority.asc(), SyncRequest.created_at.asc())
                        .limit(25)
                    )
                ).scalars().all()
            from src.api.routers.connectors import _run_cloud_sync_inline
            for request_id in request_ids:
                async with SessionLocal() as claim_db:
                    await set_database_context(claim_db, None, True)
                    request = await claim_sync_request(claim_db, request_id)
                if request and request.job_id:
                    asyncio.create_task(
                        _run_cloud_sync_inline(
                            request.connector_id,
                            request.job_id,
                            request.id,
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inline connector dispatch pass failed")


# Ten minutes, matching the Celery beat entry. Both modes must maintain subscriptions
# on the same cadence: a deployment that switches to JOB_MODE=inline to avoid running
# Redis would otherwise keep every webhook only until its first expiry.
WEBHOOK_MAINTENANCE_INTERVAL_SECONDS = 600


async def _inline_webhook_maintenance_loop() -> None:
    """Renew and repair provider push subscriptions when Celery is not enabled."""

    from src.domain.webhook_subscriptions import (
        renew_due_subscriptions,
        repair_webhook_subscriptions,
    )

    async def restore_context(session) -> None:
        await set_database_context(session, None, True)

    while True:
        await asyncio.sleep(WEBHOOK_MAINTENANCE_INTERVAL_SECONDS)
        try:
            async with SessionLocal() as db:
                await set_database_context(db, None, True)
                await renew_due_subscriptions(db)
                await repair_webhook_subscriptions(db, set_context=restore_context)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inline webhook maintenance pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Database migrations/readiness must complete before serving traffic.
    await initialize_resources()
    recovery_task: asyncio.Task | None = None
    connector_dispatch_task: asyncio.Task | None = None
    webhook_maintenance_task: asyncio.Task | None = None
    if settings.JOB_MODE.lower() != "celery":
        recovery_task = asyncio.create_task(_inline_outbox_recovery_loop())
        connector_dispatch_task = asyncio.create_task(_inline_connector_sync_dispatch_loop())
        webhook_maintenance_task = asyncio.create_task(_inline_webhook_maintenance_loop())
    try:
        yield
    finally:
        if recovery_task:
            recovery_task.cancel()
            try:
                await recovery_task
            except asyncio.CancelledError:
                pass
        if connector_dispatch_task:
            connector_dispatch_task.cancel()
            try:
                await connector_dispatch_task
            except asyncio.CancelledError:
                pass
        if webhook_maintenance_task:
            webhook_maintenance_task.cancel()
            try:
                await webhook_maintenance_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=(
        f"{settings.API_V1_STR}/openapi.json" if settings.ENABLE_API_DOCS else None
    ),
    docs_url="/docs" if settings.ENABLE_API_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
    lifespan=lifespan,
)


async def unhandled_error_boundary(request, call_next):
    """Turn an unhandled exception into a real response, INSIDE CORSMiddleware.

    Starlette converts an escaped exception to a 500 in ServerErrorMiddleware, which
    wraps everything — so that response never passes back through CORSMiddleware and
    carries no `access-control-allow-origin`. The browser then discards it before any
    JavaScript sees it, and axios reports a bare "Network Error" with no status: a real
    server-side bug is indistinguishable from the API being unreachable. A source upload
    failed exactly this way and cost two rounds of misdiagnosis against the tunnel and
    the ClamAV sidecar, neither of which was involved.

    Registered BEFORE CORSMiddleware so it sits INSIDE it: `add_middleware` inserts at
    the front of the stack, so the last one registered is the outermost. Returning here
    rather than re-raising means the 500 travels back out through CORS like any other
    response and reaches the client with its headers intact.

    HTTPException never arrives here — ExceptionMiddleware is further in and has already
    turned it into a response. Only genuinely unhandled errors reach this.
    """
    try:
        return await call_next(request)
    except Exception:
        try:
            # exception(), not error(): without the traceback the log names the
            # exception and nothing about where it came from, which is what made the
            # upload failure unreadable in CloudWatch as well as in the browser.
            logger.exception(
                "Unhandled application error",
                request_id=getattr(request.state, "request_id", None),
                method=request.method,
                path=request.url.path,
            )
        except Exception:
            # A boundary that its own logging can defeat is not a boundary. Rendering a
            # traceback can itself raise — a non-UTF-8 stream turns one non-ASCII
            # character into UnicodeEncodeError — and that escaped exception would
            # restore precisely the CORS-less 500 this exists to prevent. Caught while
            # testing this function, not hypothesised.
            pass
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


app.add_middleware(BaseHTTPMiddleware, dispatch=unhandled_error_boundary)

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
        # setdefault, not assignment: this used to overwrite whatever the endpoint had
        # chosen, so `/articles/{id}/source` could not opt in to being framed by our own
        # source viewer no matter what it set — DENY blocks an iframe even same-origin,
        # and the citation preview was blocked in the browser. Everything else still
        # gets DENY, because nothing else has any business being framed.
        response.headers.setdefault("X-Frame-Options", "DENY")
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
            exc,
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
