"""The cross-encoder inside the real `SearchService.search`.

Only the embedder, the lexical scorer and the model call are replaced (same approach as
test_search_bounds_the_cross_encoder.py). What is pinned here is what the search hands the model and what it
does with the answer.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain import search_service as module
from src.domain.search_service import SearchService
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User

COMPANY = "acme.test"
QUESTION = "How is the watchdog counter clocked and how long is one tick?"


def _admin() -> User:
    user = User(
        id=uuid.uuid4(),
        email="a@acme.test",
        name="A",
        company_domain=COMPANY,
        role="Admin",
        active=True,
    )
    role = Role(name="Admin", company_domain=COMPANY, active=True)
    role.permissions.append(
        RolePermission(
            permission=Permission(key="article.read", name="Read"), scope="company"
        )
    )
    user.roles.append(role)
    return user


def _article():
    return SimpleNamespace(
        id=uuid.uuid4(),
        title="T",
        dept="Eng",
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


def _chunk(article, fake_score: float, label: str):
    return SimpleNamespace(
        id=uuid.uuid4(),
        parent_chunk_id=uuid.uuid4(),
        article=article,
        chunk_text=label,
        heading=label,
        chunk_type="text",
        chunking_version=None,
        page_number=None,
        fake_score=fake_score,
        label=label,
        parent_chunk=SimpleNamespace(
            text=label, section_ref="S1", heading="H", page_number=None, child_chunks=[]
        ),
    )


class _Repo:
    def __init__(self, chunks, vector_ranks_by_label=None):
        self.chunks = chunks
        self.db = None
        self.vector_ranks: dict = {}
        self._by_label = vector_ranks_by_label or {}

    async def hybrid_search(
        self, *, query, query_embedding, user, limit, filters, identifiers=None
    ):
        for chunk in self.chunks:
            if chunk.label in self._by_label:
                self.vector_ranks[chunk.id] = self._by_label[chunk.label]
        return list(self.chunks)


@pytest.fixture
def on(monkeypatch):
    async def fake_embed(text):
        return [0.1, 0.2, 0.3]

    def fake_lexical(query, chunks, limit=5, identifiers=None, vector_ranks=None):
        ranked = sorted(chunks, key=lambda c: c.fake_score, reverse=True)
        return [(c, c.fake_score) for c in ranked[:limit]]

    monkeypatch.setattr(module, "embed_query", fake_embed)
    monkeypatch.setattr(module, "rerank_chunks_with_scores", fake_lexical)
    monkeypatch.setattr(settings, "CROSS_ENCODER_RERANKER_ENABLED", True)
    monkeypatch.setattr(settings, "CROSS_ENCODER_CANDIDATE_POOL", 3)
    monkeypatch.setattr(settings, "CROSS_ENCODER_VECTOR_CANDIDATES", 1)
    monkeypatch.setattr(settings, "CROSS_ENCODER_ONLY_DEEP_QUERIES", False)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)


def _search(repo, limit=10, query=QUESTION):
    service = SearchService(chunk_repo=repo, gov_repo=None, feature_flags=None)
    return asyncio.run(service.search(_admin(), query, limit=limit))


def _pool(n=10):
    article = _article()
    return [_chunk(article, round(0.9 - 0.05 * i, 2), f"c{i}") for i in range(n)]


def test_the_model_reads_the_question_as_asked_and_the_lexical_and_vector_candidates(
    on, monkeypatch
):
    chunks = _pool()
    seen = {}

    def capture(query, passages):
        seen["query"], seen["passages"] = query, passages
        return [0.5] * len(passages)

    monkeypatch.setattr(module, "cross_encoder_score", capture)
    _search(_Repo(chunks, {"c9": 1}))

    assert (
        seen["query"] == QUESTION
    ), "the original question, not the lower-cased retrieval query"
    heads = [p.split("\n")[1] for p in seen["passages"]]
    assert heads == [
        "c0",
        "c1",
        "c2",
        "c9",
    ], "the first 3 lexical, and the vector's best wherever it sat"


def test_a_chunk_far_down_the_lexical_order_that_the_model_picks_comes_first(
    on, monkeypatch
):
    chunks = _pool()
    monkeypatch.setattr(
        module, "cross_encoder_score", lambda q, passages: [0.1, 0.1, 0.1, 0.9]
    )
    out = _search(_Repo(chunks, {"c9": 1}))
    assert out[0]["chunk_id"] == str(chunks[9].id)


def test_the_search_returns_as_many_results_as_it_would_without_the_model(
    on, monkeypatch
):
    chunks = _pool(10)
    monkeypatch.setattr(
        module, "cross_encoder_score", lambda q, passages: [0.5] * len(passages)
    )
    assert len(_search(_Repo(chunks, {"c9": 1}), limit=8)) == 8


def test_a_passage_the_model_is_sure_of_survives_a_gate_that_only_counts_shared_words(
    on, monkeypatch
):
    """The point of the assist: lexical score 0.05, the model says 0.9, the gate is 0.3."""
    article = _article()
    buried = _chunk(article, 0.05, "answer")
    others = [_chunk(article, 0.9 - 0.1 * i, f"o{i}") for i in range(3)]
    monkeypatch.setattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.3)
    monkeypatch.setattr(settings, "CROSS_ENCODER_CANDIDATE_POOL", 4)
    monkeypatch.setattr(settings, "CROSS_ENCODER_ASSIST_TOP_K", 1)
    monkeypatch.setattr(
        module, "cross_encoder_score", lambda q, passages: [0.1, 0.1, 0.1, 0.95]
    )
    with_model = [r["chunk_id"] for r in _search(_Repo([*others, buried]))]
    monkeypatch.setattr(settings, "CROSS_ENCODER_RERANKER_ENABLED", False)
    without_model = [r["chunk_id"] for r in _search(_Repo([*others, buried]))]
    assert str(buried.id) in with_model
    assert str(buried.id) not in without_model


def test_a_model_that_fails_leaves_the_lexical_order(on, monkeypatch):
    from src.rag.cross_encoder import CrossEncoderUnavailable

    def broken(q, passages):
        raise CrossEncoderUnavailable("no model baked")

    monkeypatch.setattr(module, "cross_encoder_score", broken)
    chunks = _pool(5)
    out = _search(_Repo(chunks))
    assert [r["chunk_id"] for r in out] == [str(c.id) for c in chunks]


def test_it_can_be_limited_to_complex_questions(on, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module, "cross_encoder_score", lambda q, p: calls.append(1) or [0.5] * len(p)
    )
    monkeypatch.setattr(settings, "CROSS_ENCODER_ONLY_DEEP_QUERIES", True)
    _search(_Repo(_pool(5)), query="what is the leave policy")
    assert calls == []
    _search(_Repo(_pool(5)), query="what is the leave policy? and who approves it?")
    assert calls == [1]


def test_by_default_it_runs_on_an_ordinary_question(on, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module, "cross_encoder_score", lambda q, p: calls.append(1) or [0.5] * len(p)
    )
    _search(_Repo(_pool(5)), query="what is the leave policy")
    assert calls == [1]
