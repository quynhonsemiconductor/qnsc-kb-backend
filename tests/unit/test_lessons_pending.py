"""A lesson an assistant saved is not an article until a reviewer approves it, so it used to be invisible to that
assistant the next day: it said nothing was recorded, or recorded the same thing again (a second draft for the
reviewer). The author's own pending drafts are now listed (GET /knowledge/my-drafts, MCP `list_my_drafts`) and
compared with a new lesson (src/domain/lessons.py)."""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import src.domain.llm_client as llm_client
from src.api.routers import knowledge
from src.domain import lessons
from src.domain.search_service import SearchService


def _run(coro):
    return asyncio.run(coro)


def _draft(title, body, status="pending", **extra):
    return SimpleNamespace(
        id=uuid.uuid4(),
        title=title,
        summary=body,
        restructured_body_md=None,
        status=status,
        dept="Engineering",
        assigned_approver_id=extra.get("assigned_approver_id"),
        review_note=extra.get("review_note"),
        tags=["session-lesson"],
        created_at=datetime(2026, 11, 4, 9, 0, 0),
    )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Db:
    """Returns the given drafts for any select and remembers the statement."""

    def __init__(self, drafts):
        self.drafts, self.statements = drafts, []

    async def execute(self, statement):
        self.statements.append(
            str(statement.compile(compile_kwargs={"literal_binds": False}))
        )
        return _Result(self.drafts)

    async def get(
        self, _model, _article_id
    ):  # never needed: the drafts carry their text
        return None


def _nothing_else(monkeypatch):
    async def no_matches(_db, _user, _content):
        return []

    async def no_search(self, user, query, filters=None, limit=5):
        return []

    monkeypatch.setattr(lessons, "find_similar_documents", no_matches)
    monkeypatch.setattr(SearchService, "search", no_search)


def _model(monkeypatch, reply):
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(_messages, **_k):
        return reply, 1, "m", "p"

    monkeypatch.setattr(llm_client, "complete", fake)


USER = SimpleNamespace(id=uuid.uuid4(), company_domain="acme.test")
BODY = "## Lesson\n\nBoard bring-up console is UART0 at 115200 baud because UART1 shares its pins with SPI."


def test_a_reworded_lesson_matching_the_authors_pending_draft_is_a_skip(monkeypatch):
    _nothing_else(monkeypatch)
    draft = _draft("Board bring-up console", BODY)
    _model(monkeypatch, '{"decision": "skip", "target": 1, "reason": "same decision"}')

    out = _run(
        lessons.decide_lesson(
            _Db([draft]),
            USER,
            "Console debug default",
            "## Lesson\n\nFor board bring-up the team uses UART0 at 115200 baud as the console; UART1 shares pins with SPI.",
        )
    )

    assert out.action == "skip" and out.pending is True
    assert out.article_id == str(draft.id)


def test_a_pending_draft_cannot_be_updated_so_an_update_is_reported_as_a_skip(
    monkeypatch,
):
    _nothing_else(monkeypatch)
    draft = _draft("Board bring-up console", BODY)
    _model(
        monkeypatch, '{"decision": "update", "target": 1, "reason": "adds the baud"}'
    )

    out = _run(lessons.decide_lesson(_Db([draft]), USER, "Console", BODY))

    assert out.action == "skip" and out.pending is True


def test_new_information_next_to_a_pending_draft_is_still_added(monkeypatch):
    _nothing_else(monkeypatch)
    _model(
        monkeypatch, '{"decision": "add", "target": null, "reason": "different fact"}'
    )

    out = _run(
        lessons.decide_lesson(
            _Db([_draft("Board bring-up console", BODY)]), USER, "Console", BODY
        )
    )

    assert out.action == "add" and out.pending is False


def test_without_a_model_a_close_copy_of_a_pending_draft_is_a_skip(monkeypatch):
    _nothing_else(monkeypatch)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)

    out = _run(
        lessons.decide_lesson(
            _Db([_draft("Board bring-up console", BODY)]),
            USER,
            "Board bring-up console",
            BODY,
        )
    )

    assert out.action == "skip" and out.pending is True


def test_an_unrelated_pending_draft_is_not_a_candidate(monkeypatch):
    _nothing_else(monkeypatch)
    _model(monkeypatch, '{"decision": "skip", "target": 1, "reason": "x"}')

    out = _run(
        lessons.decide_lesson(
            _Db(
                [
                    _draft(
                        "Expense policy",
                        "Travel expenses are paid within thirty days of the receipt being filed.",
                    )
                ]
            ),
            USER,
            "Console",
            BODY,
        )
    )

    assert out.action == "add"


def test_only_the_authors_unpublished_drafts_are_read(monkeypatch):
    _nothing_else(monkeypatch)
    db = _Db([])
    _run(lessons.decide_lesson(db, USER, "Console", BODY))

    sql = " ".join(db.statements)
    assert "created_by" in sql and "status IN" in sql
    assert lessons.OWN_DRAFT_STATUSES == (
        "draft",
        "pending",
    ), "published and rejected drafts are not duplicates"


def test_a_failing_draft_lookup_never_blocks_saving(monkeypatch):
    _nothing_else(monkeypatch)

    class _Broken:
        async def execute(self, _statement):
            raise RuntimeError("db down")

    out = _run(lessons.decide_lesson(_Broken(), USER, "Console", BODY))

    assert out.action == "add"


# -- the listing -----------------------------------------------------------------------------


def test_my_drafts_lists_the_callers_submissions_with_the_reviewers_note_when_rejected():
    waiting = _draft("Waiting", "body " * 200)
    rejected = _draft("Rejected", "r", status="rejected", review_note="Too vague")
    db = _Db([waiting, rejected])

    out = _run(knowledge.my_drafts(limit=20, current_user=USER, db=db))

    assert [item["status"] for item in out] == ["waiting_for_review", "rejected"]
    assert out[0]["review_note"] is None and out[1]["review_note"] == "Too vague"
    assert out[0]["has_approver"] is False
    assert len(out[0]["excerpt"]) == 400
    sql = db.statements[0]
    assert "created_by" in sql and "company_domain" in sql


def test_submitting_a_lesson_that_is_already_pending_saves_nothing_and_says_where_to_look(
    monkeypatch,
):
    draft_id = str(uuid.uuid4())

    async def decided(_db, _user, _title, _body):
        return lessons.LessonDecision(
            "skip", draft_id, "Board bring-up console", "same", pending=True
        )

    monkeypatch.setattr(lessons, "decide_lesson", decided)

    out = _run(
        knowledge.submit_lesson(
            knowledge.LessonRequest(title="t", lesson="l"),
            current_user=USER,
            db=_Db([]),
        )
    )

    assert (
        out["decision"] == "skip"
        and out["pending"] is True
        and out["article_id"] == draft_id
    )
    assert "list_my_drafts" in out["next"] and "waiting for review" in out["next"]


# -- the duplicate check does not spend the cross-encoder ---------------------------------------


def test_the_duplicate_check_search_skips_the_cross_encoder(monkeypatch):
    seen = []

    async def fake_search(self, user, query, filters=None, limit=5):
        seen.append(self.use_cross_encoder)
        return []

    monkeypatch.setattr(SearchService, "search", fake_search)

    _run(lessons._retrieved_candidates(SimpleNamespace(), USER, "q"))

    assert seen == [False]
    assert SearchService.use_cross_encoder is True, "ordinary searches keep it"


from src.domain import search_service as module  # noqa: E402
from tests.unit.test_search_bounds_the_cross_encoder import (  # noqa: E402,F401 - fixture and helpers
    _article,
    _chunk,
    _run as _search_run,
    cross_encoder_on,
)


def test_a_search_that_opts_out_never_calls_the_cross_encoder(
    cross_encoder_on, monkeypatch
):
    article = _article()
    high, low = _chunk(article, 0.9, "lexically first"), _chunk(
        article, 0.5, "lexically second"
    )
    calls = []
    monkeypatch.setattr(
        module,
        "cross_encoder_score",
        lambda query, passages: calls.append(1) or [0.0, 1.0],
    )
    monkeypatch.setattr(module.SearchService, "use_cross_encoder", False)

    ids, _ = _search_run([high, low])

    assert calls == [] and ids == [str(high.id), str(low.id)]
