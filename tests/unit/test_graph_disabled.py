import asyncio

from src.api import main
from src.core.config import settings
from src.domain import entity_extraction, llm_client


def test_knowledge_graph_api_is_not_registered_by_default():
    paths = main.app.openapi()["paths"]

    assert settings.KNOWLEDGE_GRAPH_ENABLED is False
    assert not [path for path in paths if path.startswith("/api/v1/graph")]


def test_graph_master_switch_prevents_publish_time_llm_calls(monkeypatch):
    called = False

    async def complete(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Knowledge Graph is disabled")

    monkeypatch.setattr(settings, "KNOWLEDGE_GRAPH_ENABLED", False)
    monkeypatch.setattr(settings, "GRAPH_EXTRACTION_ENABLED", True)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: object())
    monkeypatch.setattr(llm_client, "complete", complete)

    result = asyncio.run(
        entity_extraction.extract_entities_and_relationships("Title", "Body")
    )

    assert result.entities == []
    assert result.relationships == []
    assert called is False
