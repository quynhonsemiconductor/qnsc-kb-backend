"""The identifier leg of hybrid_search: a share of the pool for each name in the question.

Same approach as test_graph_augmented_retrieval.py: a fake DB answers by the compiled statement, since the full-text
expressions are PostgreSQL-specific. The leg exists because the keyword leg ranks by occurrences: for
"Thanh ghi PADOUTSET của GPIO nằm ở offset nào?" none of the chunks naming PADOUTSET was among the ~380
candidates the reranker scores, while 200 that repeat GPIO were.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from src.core.config import settings
from src.models.user import Department, User
from src.repositories.chunk import ChunkRepository


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self


class _FakeDB:
    """Answers an identifier statement with the rows registered for the term it searches for."""

    def __init__(self, by_term=None):
        self.by_term = by_term or {}
        self.statements: list = []

    @staticmethod
    def _terms(statement) -> set[str]:
        compiled = statement.compile(dialect=postgresql.dialect())
        return {str(v) for v in compiled.params.values() if isinstance(v, str)}

    async def execute(self, statement, params=None, *args, **kwargs):
        self.statements.append(statement)
        if _is_identifier_statement(statement):
            for term, rows in self.by_term.items():
                if term in self._terms(statement):
                    return _Rows(rows)
        return _Rows([])

    async def scalar(self, statement):
        return 0

    def identifier_statements(self) -> list:
        return [s for s in self.statements if _is_identifier_statement(s)]


def _is_identifier_statement(statement) -> bool:
    """The leg's per-name queries are capped at the identifier pool; the keyword and title queries at the much
    larger candidate pool."""
    limit = getattr(statement, "_limit", None)
    return limit is not None and limit <= settings.RAG_IDENTIFIER_POOL_SIZE


def _user() -> User:
    user = User(
        id=uuid.uuid4(), role="Staff", company_domain="acme.test", dept="Engineering"
    )
    user.departments = [Department(id=uuid.uuid4(), name="Engineering", active=True)]
    return user


def _chunk(label: str = ""):
    return SimpleNamespace(id=uuid.uuid4(), label=label)


def _candidates(db, question, identifiers=None):
    return asyncio.run(
        ChunkRepository(db)._identifier_candidates(question, [], identifiers)
    )


def test_each_name_in_the_question_gets_its_own_query():
    db = _FakeDB()
    _candidates(db, "Thanh ghi PADOUTSET của GPIO nằm ở offset nào?")
    assert len(db.statements) == 2


def test_each_query_is_limited_to_its_share_of_the_pool(monkeypatch):
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_POOL_SIZE", 10)
    db = _FakeDB()
    _candidates(db, "PADOUTSET GPIO")
    assert [s._limit for s in db.statements] == [5, 5]


def test_an_ordinary_question_does_not_query_at_all():
    db = _FakeDB()
    assert _candidates(db, "Nhân viên được nghỉ phép bao nhiêu ngày?") == []
    assert db.statements == []


def test_identifiers_given_by_the_caller_are_used_as_they_are():
    """The caller holds the original question; `query` is the lower-cased one, where no name can be told."""
    db = _FakeDB()
    _candidates(db, "thanh ghi padoutset gpio offset", identifiers=["padoutset"])
    assert len(db.statements) == 1


def test_a_rare_name_keeps_all_its_chunks_beside_a_common_one():
    """What a single ranked list got wrong: the common name's chunks filled the pool."""
    rare = [_chunk(f"padoutset-{n}") for n in range(3)]
    common = [_chunk(f"gpio-{n}") for n in range(24)]
    db = _FakeDB({"padoutset": rare, "gpio": common})
    found = _candidates(db, "PADOUTSET GPIO")
    assert all(chunk in found for chunk in rare)


def test_a_chunk_named_by_both_comes_first_and_only_once():
    both = _chunk("both")
    only_rare = _chunk("rare")
    only_common = _chunk("common")
    db = _FakeDB({"padoutset": [only_rare, both], "gpio": [only_common, both]})
    found = _candidates(db, "PADOUTSET GPIO")
    assert found[0] is both
    assert len(found) == len({chunk.id for chunk in found}) == 3


def test_the_pool_is_bounded(monkeypatch):
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_POOL_SIZE", 4)
    db = _FakeDB(
        {
            "padoutset": [_chunk(f"a{n}") for n in range(10)],
            "gpio": [_chunk(f"b{n}") for n in range(10)],
        }
    )
    assert len(_candidates(db, "PADOUTSET GPIO")) == 4


def _search(monkeypatch, *, enabled: bool, question: str, db=None) -> _FakeDB:
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_LEG_ENABLED", enabled)
    db = db or _FakeDB()
    asyncio.run(
        ChunkRepository(db).hybrid_search(
            question, None, _user(), limit=5, identifiers=_names(question)
        )
    )
    return db


def _names(question):
    from src.rag.identifier_terms import identifier_terms

    return identifier_terms(question)


def test_the_leg_runs_inside_hybrid_search(monkeypatch):
    db = _search(monkeypatch, enabled=True, question="Offset của PADOUTSET là gì?")
    assert len(db.identifier_statements()) == 1


def test_it_can_be_switched_off(monkeypatch):
    db = _search(monkeypatch, enabled=False, question="Offset của PADOUTSET là gì?")
    assert db.identifier_statements() == []


def test_a_question_without_a_name_costs_nothing_extra(monkeypatch):
    on = _search(monkeypatch, enabled=True, question="nghỉ phép bao nhiêu ngày")
    off = _search(monkeypatch, enabled=False, question="nghỉ phép bao nhiêu ngày")
    assert len(on.statements) == len(off.statements)


def test_a_chunk_found_by_the_leg_joins_the_fused_pool(monkeypatch):
    found = _chunk("register")
    monkeypatch.setattr(settings, "RAG_IDENTIFIER_LEG_ENABLED", True)
    db = _FakeDB({"padoutset": [found]})
    fused = asyncio.run(
        ChunkRepository(db).hybrid_search(
            "Offset của PADOUTSET là gì?",
            None,
            _user(),
            limit=5,
            identifiers=["padoutset"],
        )
    )
    assert [chunk.id for chunk in fused] == [found.id]
