"""The retention sweep deletes an eligible row and never touches audit logs.

The sweep sets `app.global_admin = true` for its own session: audit and telemetry span every
tenant, and a policy-scoped session would silently delete nothing. This test therefore also
fails when a row is eligible but zero rows are removed — the symptom of losing that GUC or
of an RLS policy change (it runs as the restricted role in the CI RLS step).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.deps import set_database_context
from src.core.config import settings
from src.models.governance import AuditLog
from src.models.ops import SearchLog
from src.workers.tasks import prune_operational_metrics

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs the migrated PostgreSQL test database",
)


def test_an_eligible_row_is_deleted_and_audit_logs_are_kept(monkeypatch):
    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    search_id = uuid.uuid4()
    audit_id = uuid.uuid4()
    old = datetime(2020, 1, 1)

    async def setup() -> None:
        async with factory() as db:
            await set_database_context(db, None, True)
            await db.execute(
                insert(SearchLog).values(
                    id=search_id,
                    query="retention test",
                    result_count=0,
                    latency_ms=1,
                    created_at=old,
                    updated_at=old,
                )
            )
            await db.execute(
                insert(AuditLog).values(
                    id=audit_id,
                    action="retention.test",
                    target_type="test",
                    target_id=str(audit_id),
                    outcome="success",
                    detail_json={},
                    created_at=old,
                    updated_at=old,
                )
            )
            await db.commit()

    async def check() -> tuple[object, object]:
        async with factory() as db:
            await set_database_context(db, None, True)
            gone = await db.scalar(
                select(SearchLog.id).where(SearchLog.id == search_id)
            )
            kept = await db.scalar(select(AuditLog.id).where(AuditLog.id == audit_id))
            return gone, kept

    async def cleanup() -> None:
        async with factory() as db:
            await set_database_context(db, None, True)
            await db.execute(delete(SearchLog).where(SearchLog.id == search_id))
            await db.execute(delete(AuditLog).where(AuditLog.id == audit_id))
            await db.commit()
        await engine.dispose()

    try:
        asyncio.run(setup())
        monkeypatch.setattr(settings, "RETENTION_SWEEP_ENABLED", True)
        monkeypatch.setattr(settings, "RETENTION_DRY_RUN", False)
        # The task is synchronous at the Celery boundary; running it here is the same call
        # beat makes. A 0-row delete with an eligible row is the failure this pins.
        prune_operational_metrics.run()
        gone, kept = asyncio.run(check())
        assert gone is None, "the eligible search_logs row was not deleted"
        assert kept == audit_id, "audit_logs must never be capped"
    finally:
        asyncio.run(cleanup())
