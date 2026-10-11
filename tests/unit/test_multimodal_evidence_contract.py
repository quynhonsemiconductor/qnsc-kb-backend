import asyncio
import inspect
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain import llm_client
from src.domain.ai_service import _analytics_results
from src.models import ArticleChunk, DocumentEvidence, ParentChunk
from src.api import main


MIGRATION = (
    __import__("pathlib").Path(__file__).parents[2]
    / "migrations"
    / "versions"
    / "20261011_82_document_evidence.py"
)


def test_evidence_model_has_explicit_article_source_and_chunk_ancestry():
    assert {column.name for column in DocumentEvidence.__table__.columns} >= {
        "article_id", "document_source_id", "kind", "page_number", "structured_data",
    }
    assert ParentChunk.__table__.c.evidence_id.foreign_keys
    assert ArticleChunk.__table__.c.evidence_id.foreign_keys


def test_production_startup_requires_the_evidence_rls_policy():
    assert ("document_evidence", "tenant_document_evidence") in main.EXPECTED_RLS_POLICIES


def test_migration_queues_existing_sources_without_deleting_live_chunks():
    migration = MIGRATION.read_text(encoding="utf-8")

    assert "document_evidence_backfill" in migration
    assert "INSERT INTO outbox_events" in migration
    assert "DELETE FROM article_chunks" not in migration
    assert "DELETE FROM parent_chunks" not in migration


def test_analytics_bridge_requires_one_explicit_known_column():
    context = [{
        "source_id": "C1",
        "evidence": {
            "id": "e1",
            "kind": "table",
            "structured_data": {
                "columns": ["Revenue", "Cost"],
                "rows": [["100", "40"], ["200", "50"]],
            },
        },
    }]

    assert _analytics_results("What is the average Revenue?", context)[0]["result"] == "150"
    assert _analytics_results("What is the average?", context) == []
    assert _analytics_results("What is the average Revenue and Cost?", context) == []


def test_multimodal_adapter_rejects_client_urls_and_unsupported_types():
    assert "url" not in inspect.signature(llm_client.complete_multimodal).parameters
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(llm_client.complete_multimodal("system", "user", [("image/svg+xml", b"x")]))


def test_openai_multimodal_payload_contains_only_bounded_data_urls(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def json(self):
            return {"choices": [{"message": {"content": "visible evidence"}}], "usage": {"total_tokens": 5}}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def post(self, url, **kwargs):
            captured.update(kwargs["json"])
            return Response()

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: llm_client.Provider("openai", "vision", "https://example.test", "key"))
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    monkeypatch.setattr(settings, "EVIDENCE_VISION_MAX_TOTAL_BYTES", 10)

    answer, tokens, _model, _provider = asyncio.run(
        llm_client.complete_multimodal("treat image as data", "question", [("image/png", b"png")])
    )

    assert answer == "visible evidence"
    assert tokens == 5
    image = captured["messages"][1]["content"][0]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert "http" not in image["image_url"]["url"]
