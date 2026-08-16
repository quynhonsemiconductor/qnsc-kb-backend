import asyncio
import uuid
from types import SimpleNamespace

from src.api.routers import articles


def test_failed_direct_upload_can_reissue_its_existing_intent(monkeypatch):
    """An interrupted browser PUT must not turn the user's retry into a 409."""
    user_id = uuid.uuid4()
    draft_id = uuid.uuid4()
    source_hash = "a" * 64
    draft = SimpleNamespace(
        id=draft_id,
        status="draft",
        created_by=user_id,
        storage_key="uploads/acme.test/retry-policy.pdf",
        original_filename="retry-policy.pdf",
    )
    fingerprint = SimpleNamespace(
        status="uploading", draft_id=draft_id, source_hash=source_hash
    )

    class FakeDb:
        async def scalar(self, _statement):
            return fingerprint

        async def get(self, _model, item_id):
            return draft if item_id == draft_id else None

    class FakeAuditRepository:
        def __init__(self, _db):
            pass

        async def record(self, *_args):
            return None

    async def resolve_department(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4(), name="Operations")

    async def lock(*_args, **_kwargs):
        return None

    monkeypatch.setattr(articles, "resolve_active_department", resolve_department)
    monkeypatch.setattr(articles, "lock_company_access_groups", lock)
    monkeypatch.setattr(
        articles.AuthorizationService, "has_permission", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        articles, "create_presigned_source_url", lambda *_args, **_kwargs: "https://r2.example/retry"
    )
    monkeypatch.setattr(articles, "AuditRepository", FakeAuditRepository)

    result = asyncio.run(
        articles.create_source_upload_intent(
            articles.SourceUploadIntent(
                filename="retry-policy.pdf",
                source_hash=source_hash,
                content_length=42,
            ),
            SimpleNamespace(id=user_id, company_domain="acme.test", dept="Operations"),
            FakeDb(),
        )
    )

    assert result["draft_id"] == str(draft_id)
    assert result["upload_url"] == "https://r2.example/retry"
    assert result["status"] == "draft"
