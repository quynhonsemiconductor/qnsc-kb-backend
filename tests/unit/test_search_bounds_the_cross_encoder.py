"""A slow cross-encoder must cost a search at most CROSS_ENCODER_TIMEOUT_SECONDS.

Measured (scripts/bench_inference.py): bge-reranker-v2-m3 scores ~2 passages per second on 2 vCPU,
so the default pool of 30 takes ~15 s there. The feature is off by default and only fires for
comparison / multi-question queries, but when it fires on a task that small the search used to
wait for all of it. It now gives up and returns the lexical order it already has.

Driven through the real `SearchService.search`; only the embedder, the lexical scorer and the
model call are replaced (same approach as test_multi_query_retrieval.py).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain import search_service as module
from src.domain.search_service import SearchService
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User
from src.rag.reranker import normalize_query

COMPANY = "acme.test"
# Two question marks: the cheapest way to satisfy needs_deep_retrieval().
QUERY = "what is the leave policy? and who approves it?"


def _admin() -> User:
    user = User(id=uuid.uuid4(), email="a@acme.test", name="A", company_domain=COMPANY, role="Admin", active=True)
    role = Role(name="Admin", company_domain=COMPANY, active=True)
    role.permissions.append(RolePermission(permission=Permission(key="article.read", name="Read"), scope="company"))
    user.roles.append(role)
    return user


def _chunk(article, fake_score: float, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), parent_chunk_id=uuid.uuid4(), article=article, chunk_text=label, chunk_type="text",
        chunking_version=None, page_number=None, fake_score=fake_score, label=label,
        parent_chunk=SimpleNamespace(text=label, section_ref="S1", heading="H", page_number=None, child_chunks=[]),
    )


def _article():
    return SimpleNamespace(
        id=uuid.uuid4(), title="T", dept="Eng", domain=None, type="document", sensitivity="internal",
        status="published", lifecycle_status="active", visibility="company", company_domain=COMPANY,
        owner_id=uuid.uuid4(), sources=[], user_permissions=[], departments=[], owner=None, last_reviewed=None,
    )


class _Repo:
    def __init__(self, chunks):
        self.chunks, self.db = chunks, None

    async def hybrid_search(self, *, query, query_embedding, user, limit, filters):
        return list(self.chunks)


@pytest.fixture
def cross_encoder_on(monkeypatch):
    async def fake_embed(text):
        return [0.1, 0.2, 0.3]

    def fake_lexical(query, chunks, limit=5):
        ranked = sorted(chunks, key=lambda c: c.fake_score, reverse=True)
        return [(c, c.fake_score) for c in ranked[:limit]]

    monkeypatch.setattr(module, "embed_query", fake_embed)
    monkeypatch.setattr(module, "rerank_chunks_with_scores", fake_lexical)
    monkeypatch.setattr(settings, "CROSS_ENCODER_RERANKER_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)


def _run(chunks):
    service = SearchService(chunk_repo=_Repo(chunks), gov_repo=None, feature_flags=None)

    # Timed INSIDE the loop: asyncio.run() joins the default executor on exit, so it would also
    # wait for the abandoned scoring thread, which a running server never does.
    async def timed():
        started = time.monotonic()
        out = await service.search(_admin(), QUERY, limit=10)
        return out, time.monotonic() - started

    out, elapsed = asyncio.run(timed())
    return [item["chunk_id"] for item in out], elapsed


def test_a_slow_cross_encoder_is_abandoned_and_the_lexical_order_is_kept(cross_encoder_on, monkeypatch):
    article = _article()
    high, low = _chunk(article, 0.9, "lexically first"), _chunk(article, 0.5, "lexically second")
    release = threading.Event()

    def slow_scores(query, passages):
        release.wait(5)
        return [0.0, 1.0]  # would put `low` first, if anyone waited for it

    monkeypatch.setattr(module, "cross_encoder_score", slow_scores)
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 0.1)
    release_later = threading.Timer(0.5, release.set)  # frees the thread so the loop can close
    release_later.start()
    try:
        ids, elapsed = _run([high, low])
    finally:
        release.set()

    assert ids == [str(high.id), str(low.id)], "the search must fall back to the lexical order"
    assert elapsed < 2.0, f"the search waited {elapsed:.1f}s for a reranker it had a 0.1s budget for"


def test_a_cross_encoder_inside_the_budget_still_reorders(cross_encoder_on, monkeypatch):
    """Negative control: the bound must not have switched the feature off."""
    article = _article()
    high, low = _chunk(article, 0.9, "lexically first"), _chunk(article, 0.5, "lexically second")

    monkeypatch.setattr(module, "cross_encoder_score", lambda query, passages: [0.0, 1.0])
    monkeypatch.setattr(settings, "CROSS_ENCODER_TIMEOUT_SECONDS", 5.0)

    ids, _ = _run([high, low])

    assert ids == [str(low.id), str(high.id)]
