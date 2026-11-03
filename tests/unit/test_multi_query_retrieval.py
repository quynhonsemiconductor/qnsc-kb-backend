"""Agentic/adaptive multi-query retrieval, exercised through the real `SearchService.search`.

MULTI_QUERY_RETRIEVAL_ENABLED defaults to False (core/config.py), same posture as every
other retrieval-side addition in this codebase -- the first group of tests below pins
that the feature is a true no-op at the default: `generate_subqueries` is never even
called, and exactly the one hybrid_search call that ran before this feature existed still
runs. The rest exercise the feature once explicitly turned on: parallel sub-query
expansion merged/de-duplicated by chunk id, the adaptive follow-up round firing only when
the pool is genuinely weak, and comparison queries (which already get their own free
decomposition) not also paying for an LLM call.

The reranker and cross-encoder are real; only the embedder (`embed_query`, which would
otherwise load the actual ONNX model) and the lexical scorer (`rerank_chunks_with_scores`,
whose real scores are not worth hand-crafting text to hit exact thresholds) are
monkeypatched, so every chunk's score is exactly the `fake_score` this file sets on it.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.domain.search_service import SearchService, _dedup_preserve_order, _merge_candidates
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User
from src.rag.reranker import normalize_query

COMPANY = "acme.test"


def _admin_user() -> User:
    user = User(
        id=uuid.uuid4(),
        email="admin@acme.test",
        name="Admin",
        company_domain=COMPANY,
        role="Admin",
        active=True,
    )
    role = Role(name="Admin", company_domain=COMPANY, active=True)
    role.permissions.append(
        RolePermission(permission=Permission(key="article.read", name="Read"), scope="company")
    )
    user.roles.append(role)
    return user


def _article() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        title="Test Article",
        dept="Engineering",
        domain=None,
        type="document",
        sensitivity="internal",
        status="published",
        lifecycle_status="active",
        visibility="company",
        company_domain=COMPANY,
        owner_id=uuid.uuid4(),
        sources=[],
        user_permissions=[],
        departments=[],
        owner=None,
        last_reviewed=None,
    )


def _chunk(article: SimpleNamespace, fake_score: float, text: str = "chunk text") -> SimpleNamespace:
    chunk = SimpleNamespace(
        id=uuid.uuid4(),
        parent_chunk_id=uuid.uuid4(),
        article=article,
        parent_chunk=SimpleNamespace(
            text=text, section_ref="S1", heading="Heading", page_number=None, child_chunks=[]
        ),
        chunk_text=text,
        chunk_type="text",
        chunking_version=None,
        page_number=None,
        fake_score=fake_score,
    )
    return chunk


class _FakeChunkRepo:
    """Records every hybrid_search call's query and returns configured results for it."""

    def __init__(self, results_by_query: dict[str, list]):
        self.results_by_query = results_by_query
        self.calls: list[str] = []
        self.db = None

    async def hybrid_search(self, *, query, query_embedding, user, limit, filters):
        self.calls.append(query)
        return list(self.results_by_query.get(query, []))


@pytest.fixture
def patched(monkeypatch):
    """Deterministic stand-ins for the embedder and lexical reranker, so every test
    controls scores directly via `fake_score` instead of depending on real text scoring."""
    from src.domain import search_service as module

    async def fake_embed_query(text):
        return [0.1, 0.2, 0.3]

    def fake_rerank(query, chunks, limit=5):
        ranked = sorted(chunks, key=lambda c: c.fake_score, reverse=True)
        return [(chunk, chunk.fake_score) for chunk in ranked[:limit]]

    monkeypatch.setattr(module, "embed_query", fake_embed_query)
    monkeypatch.setattr(module, "rerank_chunks_with_scores", fake_rerank)
    return module


def _search(chunk_repo, query: str) -> list[dict]:
    service = SearchService(chunk_repo=chunk_repo, gov_repo=None, feature_flags=None)
    return asyncio.run(service.search(_admin_user(), query, limit=10))


# --- default (flag off): true no-op -------------------------------------------------


def test_default_disabled_runs_exactly_the_one_search_it_always_ran(patched):
    article = _article()
    query = "unsupervised learning methods"
    retrieval_query = normalize_query(query)
    repo = _FakeChunkRepo({retrieval_query: [_chunk(article, 0.9)]})

    results = _search(repo, query)

    assert repo.calls == [retrieval_query]
    assert len(results) == 1


def test_default_disabled_never_calls_generate_subqueries(patched, monkeypatch):
    from src.domain import search_service as module

    def _boom(*_args, **_kwargs):
        raise AssertionError("generate_subqueries must not run when the feature is off")

    monkeypatch.setattr(module, "generate_subqueries", _boom)
    article = _article()
    query = "unsupervised learning methods"
    repo = _FakeChunkRepo({normalize_query(query): [_chunk(article, 0.9)]})

    _search(repo, query)  # would raise via the monkeypatched spy if this regressed


# --- enabled: parallel sub-query expansion, merged and de-duplicated ---------------


def test_enabled_expands_runs_each_subquery_and_merges_by_chunk_id(patched, monkeypatch):
    from src.core.config import settings
    from src.domain import search_service as module

    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_FOLLOWUPS", 0)

    article = _article()
    query = "unsupervised learning methods"
    retrieval_query = normalize_query(query)
    shared = _chunk(article, 0.5, text="shared passage")
    only_in_variant = _chunk(article, 0.6, text="clustering passage")

    async def fake_generate_subqueries(_question):
        return ["clustering algorithms"]

    monkeypatch.setattr(module, "generate_subqueries", fake_generate_subqueries)

    variant_query = normalize_query("clustering algorithms")
    repo = _FakeChunkRepo(
        {
            retrieval_query: [shared],
            variant_query: [shared, only_in_variant],
        }
    )

    results = _search(repo, query)

    assert set(repo.calls) == {retrieval_query, variant_query}
    result_ids = {item["chunk_id"] for item in results}
    assert result_ids == {str(shared.id), str(only_in_variant.id)}, (
        "the chunk found only via the generated sub-query must be in the merged pool, "
        "and the chunk both queries found must not be duplicated"
    )


def test_comparison_query_does_not_also_pay_for_llm_subqueries(patched, monkeypatch):
    from src.core.config import settings
    from src.domain import search_service as module

    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_FOLLOWUPS", 0)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "a comparison query already gets its own decomposition; it must not also "
            "trigger the LLM-based multi-query path"
        )

    monkeypatch.setattr(module, "generate_subqueries", _boom)

    article = _article()
    query = "so sánh SOP-114 và SOP-118"
    repo = _FakeChunkRepo(
        {
            normalize_query("SOP-114"): [_chunk(article, 0.5)],
            normalize_query("SOP-118"): [_chunk(article, 0.5)],
        }
    )

    _search(repo, query)  # would raise via the monkeypatched spy if this regressed


# --- adaptive follow-up --------------------------------------------------------------


def test_followup_fires_on_weak_evidence_and_the_stronger_result_wins(patched, monkeypatch):
    from src.core.config import settings
    from src.domain import search_service as module

    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_FOLLOWUPS", 1)
    monkeypatch.setattr(settings, "RAG_MIN_CONTEXT_SCORE", 0.35)
    monkeypatch.setattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)

    article = _article()
    query = "unsupervised learning methods"
    retrieval_query = normalize_query(query)
    weak = _chunk(article, 0.2, text="weakly related passage")
    strong = _chunk(article, 0.9, text="the actual answer")

    async def fake_generate_subqueries(_question):
        return []  # isolate this test to the follow-up mechanism alone

    calls = {"followup": 0}

    async def fake_generate_followup(question, tried_queries, weak_titles):
        calls["followup"] += 1
        assert retrieval_query in tried_queries
        assert weak_titles == ["Test Article"]
        return "a sharper query"

    monkeypatch.setattr(module, "generate_subqueries", fake_generate_subqueries)
    monkeypatch.setattr(module, "generate_followup_query", fake_generate_followup)

    followup_query = normalize_query("a sharper query")
    repo = _FakeChunkRepo({retrieval_query: [weak], followup_query: [strong]})

    results = _search(repo, query)

    assert calls["followup"] == 1
    assert followup_query in repo.calls
    assert results[0]["chunk_id"] == str(strong.id)
    assert {item["chunk_id"] for item in results} == {str(weak.id), str(strong.id)}


def test_followup_does_not_fire_when_evidence_is_already_strong(patched, monkeypatch):
    from src.core.config import settings
    from src.domain import search_service as module

    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_FOLLOWUPS", 1)
    monkeypatch.setattr(settings, "RAG_MIN_CONTEXT_SCORE", 0.35)

    def _boom(*_args, **_kwargs):
        raise AssertionError("a strong top score must not trigger the follow-up round")

    monkeypatch.setattr(module, "generate_subqueries", _no_subqueries)
    monkeypatch.setattr(module, "generate_followup_query", _boom)

    article = _article()
    query = "unsupervised learning methods"
    retrieval_query = normalize_query(query)
    repo = _FakeChunkRepo({retrieval_query: [_chunk(article, 0.9)]})

    _search(repo, query)  # would raise via the monkeypatched spy if this regressed


def test_followup_disabled_by_max_followups_zero(patched, monkeypatch):
    from src.core.config import settings
    from src.domain import search_service as module

    monkeypatch.setattr(settings, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_MAX_FOLLOWUPS", 0)
    monkeypatch.setattr(settings, "RAG_MIN_CONTEXT_SCORE", 0.35)

    def _boom(*_args, **_kwargs):
        raise AssertionError("MULTI_QUERY_MAX_FOLLOWUPS=0 must disable the follow-up round")

    monkeypatch.setattr(module, "generate_subqueries", _no_subqueries)
    monkeypatch.setattr(module, "generate_followup_query", _boom)

    article = _article()
    query = "unsupervised learning methods"
    retrieval_query = normalize_query(query)
    repo = _FakeChunkRepo({retrieval_query: [_chunk(article, 0.1)]})

    _search(repo, query)  # would raise via the monkeypatched spy if this regressed


async def _no_subqueries(_question):
    return []


# --- pure helpers --------------------------------------------------------------------


def test_dedup_preserve_order_keeps_first_occurrence_order():
    assert _dedup_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedup_preserve_order_empty_input():
    assert _dedup_preserve_order([]) == []


def test_merge_candidates_skips_already_seen_ids():
    a = SimpleNamespace(id=1)
    b = SimpleNamespace(id=2)
    b_again = SimpleNamespace(id=2)
    c = SimpleNamespace(id=3)
    seen = {1}

    merged = _merge_candidates([a], [b_again, b, c], seen)

    assert merged == [a, b_again, c]
    assert seen == {1, 2, 3}


def test_merge_candidates_mutates_seen_ids_in_place_across_calls():
    seen: set = set()
    first = _merge_candidates([], [SimpleNamespace(id=1)], seen)
    second = _merge_candidates(first, [SimpleNamespace(id=1), SimpleNamespace(id=2)], seen)

    assert [c.id for c in second] == [1, 2]
