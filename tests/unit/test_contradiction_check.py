"""The proactive contradiction check must find real conflicts and touch nothing else.

Three properties matter here, in order of how badly getting them wrong would hurt:

1. It must never block or fail the approval it runs alongside -- every failure mode
   (similarity search breaks, no candidates, no conflicts) returns quietly.
2. It must not re-flag a fact someone already marked resolved, or double-flag one that
   is already open -- both are read straight off ai_service's own dedup query.
3. It must actually find the conflict when one exists, using the same explicit-fact
   detector the reactive (answer-time) path already relies on.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.domain.contradiction_check import detect_contradictions_for_draft


class _Article:
    def __init__(self, id_, title, body_md):
        self.id = id_
        self.title = title
        self.body_md = body_md


class _FakeDB:
    """Enough of an AsyncSession for this module's two calls: `get` and `scalar`."""

    def __init__(self, articles: dict[str, _Article], existing_open_facts: set[str] | None = None):
        self._articles = articles
        self._existing_open_facts = existing_open_facts or set()
        self.added: list = []

    async def get(self, _model, article_id):
        return self._articles.get(str(article_id))

    async def scalar(self, statement):
        # Only ever called with a SELECT ... WHERE fact == X AND status == 'open'. Bound
        # values are read back by VALUE rather than by parameter name -- SQLAlchemy's
        # anonymous bind-parameter naming (e.g. "fact_1") is an implementation detail
        # this test has no business depending on.
        compiled = statement.compile()
        bound_values = set(compiled.params.values())
        if bound_values & self._existing_open_facts:
            return object()
        return None

    def add(self, obj):
        self.added.append(obj)


class _User:
    company_domain = "qnsc.vn"


def _run(db, matches, draft_body, exclude_id=None, monkeypatch=None):
    import src.domain.contradiction_check as module

    async def fake_find_similar_documents(_db, _user, _content):
        return matches

    monkeypatch.setattr(module, "find_similar_documents", fake_find_similar_documents)
    return asyncio.run(
        detect_contradictions_for_draft(db, _User(), draft_body, exclude_article_id=exclude_id)
    )


def test_no_candidates_returns_nothing(monkeypatch):
    db = _FakeDB({})
    result = _run(db, [], "Deadline: Friday", monkeypatch=monkeypatch)
    assert result == []
    assert db.added == []


def test_low_similarity_candidates_are_ignored(monkeypatch):
    other = _Article(uuid.uuid4(), "Other SOP", "Deadline: Monday")
    db = _FakeDB({str(other.id): other})
    matches = [{"article_id": str(other.id), "score": 0.05}]
    result = _run(db, matches, "Deadline: Friday", monkeypatch=monkeypatch)
    assert result == []
    assert db.added == []


def test_the_article_being_updated_is_excluded_from_its_own_comparison(monkeypatch):
    article_id = str(uuid.uuid4())
    db = _FakeDB({})
    matches = [{"article_id": article_id, "score": 0.9}]
    result = _run(db, matches, "Deadline: Friday", exclude_id=article_id, monkeypatch=monkeypatch)
    assert result == []
    assert db.added == []


def test_a_real_conflict_is_detected_and_recorded(monkeypatch):
    other = _Article(uuid.uuid4(), "Other SOP", "Deadline: Monday.")
    db = _FakeDB({str(other.id): other})
    matches = [{"article_id": str(other.id), "score": 0.5}]

    conflicts = _run(db, matches, "Deadline: Friday.", monkeypatch=monkeypatch)

    assert len(conflicts) == 1
    assert conflicts[0]["fact"] == "deadline"
    assert len(db.added) == 1
    record = db.added[0]
    assert record.company_domain == "qnsc.vn"
    assert record.fact == "deadline"
    assert record.contradiction_type == "date"
    assert set(record.article_ids) == {"draft", str(other.id)}


def test_an_ownership_conflict_is_classified_correctly(monkeypatch):
    other = _Article(uuid.uuid4(), "Other SOP", "Owner: Finance team.")
    db = _FakeDB({str(other.id): other})
    matches = [{"article_id": str(other.id), "score": 0.5}]

    _run(db, matches, "Owner: HR team.", monkeypatch=monkeypatch)

    assert db.added[0].contradiction_type == "ownership"


def test_no_shared_labelled_fact_means_no_conflict(monkeypatch):
    other = _Article(uuid.uuid4(), "Other SOP", "Owner: Finance team.")
    db = _FakeDB({str(other.id): other})
    matches = [{"article_id": str(other.id), "score": 0.5}]

    result = _run(db, matches, "This document has no labelled facts at all.", monkeypatch=monkeypatch)
    assert result == []
    assert db.added == []


def test_matching_values_are_not_a_conflict(monkeypatch):
    """Same fact, same value across two articles is agreement, not a contradiction."""
    other = _Article(uuid.uuid4(), "Other SOP", "Deadline: Friday.")
    db = _FakeDB({str(other.id): other})
    matches = [{"article_id": str(other.id), "score": 0.5}]

    result = _run(db, matches, "Deadline: Friday.", monkeypatch=monkeypatch)
    assert result == []
    assert db.added == []


def test_an_already_open_conflict_is_not_recorded_twice(monkeypatch):
    other = _Article(uuid.uuid4(), "Other SOP", "Deadline: Monday.")
    db = _FakeDB({str(other.id): other}, existing_open_facts={"deadline"})
    matches = [{"article_id": str(other.id), "score": 0.5}]

    conflicts = _run(db, matches, "Deadline: Friday.", monkeypatch=monkeypatch)

    # Still reported to the caller -- only the persisted record is deduplicated.
    assert len(conflicts) == 1
    assert db.added == []


def test_similarity_search_failure_is_swallowed(monkeypatch):
    import src.domain.contradiction_check as module

    async def broken_find_similar_documents(_db, _user, _content):
        raise RuntimeError("pgvector is down")

    monkeypatch.setattr(module, "find_similar_documents", broken_find_similar_documents)
    db = _FakeDB({})
    result = asyncio.run(
        detect_contradictions_for_draft(db, _User(), "Deadline: Friday.")
    )
    assert result == []
    assert db.added == []


def test_candidate_limit_is_respected(monkeypatch):
    """Only the top CANDIDATE_LIMIT matches are ever fetched, regardless of pool size."""
    import src.domain.contradiction_check as module

    fetched_ids: list[str] = []
    articles = {}
    matches = []
    for i in range(10):
        article = _Article(uuid.uuid4(), f"SOP {i}", "Owner: Team A.")
        articles[str(article.id)] = article
        matches.append({"article_id": str(article.id), "score": 0.9 - i * 0.01})

    class _CountingDB(_FakeDB):
        async def get(self, model, article_id):
            fetched_ids.append(str(article_id))
            return await super().get(model, article_id)

    db = _CountingDB(articles)
    _run(db, matches, "Owner: Team B.", monkeypatch=monkeypatch)
    assert len(fetched_ids) == module.CANDIDATE_LIMIT
