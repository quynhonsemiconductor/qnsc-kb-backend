"""Entity/relationship extraction must never be the reason a publish fails.

Every failure mode (no provider, a failed call, an unparseable reply, a malformed
entity/relationship) has to degrade to an empty result rather than raise -- this runs
inline in governance.py::approve_draft, right after the publish it must never undo.
"""
from __future__ import annotations

import asyncio

import pytest

from src.domain.entity_extraction import (
    ExtractionResult,
    extract_entities_and_relationships,
    normalize_entity_name,
)


@pytest.fixture
def llm(monkeypatch):
    from src.domain import llm_client
    from src.core.config import settings

    state: dict = {
        "reply": (
            '{"entities": ['
            '{"name": "Database Team", "type": "organization", "description": "Owns backups"},'
            '{"name": "Backup Policy", "type": "policy", "description": "Daily backups"}'
            '], "relationships": ['
            '{"source": "Database Team", "relation": "owns", "target": "Backup Policy", '
            '"description": "The team maintains the policy"}'
            ']}'
        ),
        "raises": None,
        "provider": object(),
    }

    async def fake_complete(messages, **kwargs):
        state["messages"] = messages
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 10, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: state["provider"])
    monkeypatch.setattr(settings, "GRAPH_EXTRACTION_ENABLED", True)
    return state


def _extract(llm, title="Doc", body="content") -> ExtractionResult:
    return asyncio.run(extract_entities_and_relationships(title, body))


def test_returns_well_formed_entities_and_relationships(llm):
    result = _extract(llm)
    assert [entity.name for entity in result.entities] == ["Database Team", "Backup Policy"]
    assert result.entities[0].type == "organization"
    edge = result.relationships[0]
    assert (edge.source, edge.relation, edge.target) == (
        "Database Team",
        "owns",
        "Backup Policy",
    )


def test_disabled_by_flag_returns_empty_without_calling(llm, monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "GRAPH_EXTRACTION_ENABLED", False)
    result = _extract(llm)
    assert result == ExtractionResult([], [])
    assert "messages" not in llm


def test_no_provider_configured_returns_empty(llm):
    llm["provider"] = None
    result = _extract(llm)
    assert result.entities == []
    assert result.relationships == []


def test_provider_failure_returns_empty_not_raises(llm):
    llm["raises"] = RuntimeError("provider is down")
    result = _extract(llm)
    assert result.entities == []


@pytest.mark.parametrize("reply", ["not json", "", "[1,2,3]", '"just a string"'])
def test_unparseable_reply_returns_empty(llm, reply):
    llm["reply"] = reply
    result = _extract(llm)
    assert result.entities == [] and result.relationships == []


def test_code_fence_is_stripped_before_parsing(llm):
    llm["reply"] = '```json\n{"entities": [{"name": "X", "type": "concept", "description": ""}], "relationships": []}\n```'
    result = _extract(llm)
    assert [entity.name for entity in result.entities] == ["X"]


def test_entity_with_invalid_type_is_dropped(llm):
    llm["reply"] = (
        '{"entities": ['
        '{"name": "Valid", "type": "concept", "description": ""},'
        '{"name": "Invalid", "type": "not-a-real-type", "description": ""}'
        '], "relationships": []}'
    )
    result = _extract(llm)
    assert [entity.name for entity in result.entities] == ["Valid"]


def test_duplicate_entity_names_are_deduplicated_accent_insensitively(llm):
    llm["reply"] = (
        '{"entities": ['
        '{"name": "an toan", "type": "concept", "description": ""},'
        '{"name": "an toàn", "type": "concept", "description": "duplicate"}'
        '], "relationships": []}'
    )
    result = _extract(llm)
    assert len(result.entities) == 1
    assert result.entities[0].name == "an toan"


def test_relationship_naming_an_unknown_entity_is_dropped(llm):
    llm["reply"] = (
        '{"entities": [{"name": "A", "type": "concept", "description": ""}], '
        '"relationships": [{"source": "A", "relation": "relates_to", "target": "B", "description": ""}]}'
    )
    result = _extract(llm)
    assert result.relationships == []


def test_self_referential_relationship_is_dropped(llm):
    llm["reply"] = (
        '{"entities": [{"name": "A", "type": "concept", "description": ""}], '
        '"relationships": [{"source": "A", "relation": "relates_to", "target": "A", "description": ""}]}'
    )
    result = _extract(llm)
    assert result.relationships == []


def test_relation_type_is_normalized_to_snake_case(llm):
    llm["reply"] = (
        '{"entities": ['
        '{"name": "A", "type": "concept", "description": ""},'
        '{"name": "B", "type": "concept", "description": ""}'
        '], "relationships": [{"source": "A", "relation": "Depends On!", "target": "B", "description": ""}]}'
    )
    result = _extract(llm)
    assert result.relationships[0].relation == "depends_on"


def test_result_is_capped_at_max_entities_and_relationships(llm):
    entities = ", ".join(f'{{"name": "E{i}", "type": "concept", "description": ""}}' for i in range(30))
    relationships = ", ".join(
        f'{{"source": "E{i}", "relation": "rel", "target": "E{i + 1}", "description": ""}}'
        for i in range(29)
    )
    llm["reply"] = f'{{"entities": [{entities}], "relationships": [{relationships}]}}'
    result = _extract(llm)
    assert len(result.entities) == 15
    assert len(result.relationships) <= 20


def test_non_dict_json_payload_returns_empty(llm):
    llm["reply"] = "[1, 2, 3]"
    result = _extract(llm)
    assert result.entities == []


def test_missing_keys_return_empty(llm):
    llm["reply"] = "{}"
    result = _extract(llm)
    assert result.entities == [] and result.relationships == []


def test_normalize_entity_name_folds_accents_and_case():
    # NFKD decomposes accented vowels (à -> a + combining grave) but NOT "đ", which is
    # its own Unicode letter rather than "d" plus a combining mark -- the same known
    # limitation auto_tagging.py's catalogue matching and articles.py's tag comparisons
    # already accept, so this only asserts what NFKD folding actually does.
    assert normalize_entity_name("Toàn Bộ") == normalize_entity_name("toan bo")
