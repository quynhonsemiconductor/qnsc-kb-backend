"""Conflict decisions, model-found conflicts, stalled-job hints, the background gate and the
session-lesson tool: the pieces borrowed from how cognee treats conflicts, jobs and sessions.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.domain import background_gate, conflict_llm, mcp_tools
from src.domain.conflict_decisions import (
    apply_conflict_decisions,
    drop_superseded,
    superseded_article_ids,
)
from src.domain.job_stall import stall_info
from src.domain.mcp_tools import LESSON_TAG, Call, ToolInputError


def _run(coro):
    return asyncio.run(coro)


# --- decisions ------------------------------------------------------------------------


def test_the_losing_article_is_dropped_when_the_winner_is_present():
    decisions = [(["a", "b"], "a")]

    assert superseded_article_ids(decisions, {"a", "b", "c"}) == {"b"}


def test_nothing_is_dropped_when_the_winner_is_not_in_the_context():
    # b lost one disagreement but may be the only source for what this question asks.
    assert superseded_article_ids([(["a", "b"], "a")], {"b", "c"}) == set()


def test_a_decision_naming_no_present_article_changes_nothing():
    assert superseded_article_ids([(["x", "y"], "x")], {"a", "b"}) == set()


def test_several_decisions_combine():
    decisions = [(["a", "b"], "a"), (["a", "c"], "a")]

    assert superseded_article_ids(decisions, {"a", "b", "c"}) == {"b", "c"}


def test_dropping_removes_only_the_superseded_articles_passages():
    results = [
        {"article_id": "a", "chunk_id": 1},
        {"article_id": "b", "chunk_id": 2},
        {"article_id": "b", "chunk_id": 3},
    ]

    kept = drop_superseded(results, {"b"})

    assert [item["chunk_id"] for item in kept] == [1]
    assert drop_superseded(results, set()) is results


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Db:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, 0

    async def execute(self, _stmt):
        self.calls += 1
        if self.error:
            raise self.error
        return _Rows(self.rows)


def test_a_recorded_decision_is_applied_to_the_context():
    db = _Db([(["a", "b"], "a")])
    results = [{"article_id": "a"}, {"article_id": "b"}]

    out = _run(apply_conflict_decisions(db, "qnsc.vn", results))

    assert out == [{"article_id": "a"}]


def test_one_article_in_the_context_needs_no_lookup():
    db = _Db([(["a", "b"], "a")])
    results = [{"article_id": "a"}, {"article_id": "a"}]

    assert _run(apply_conflict_decisions(db, "qnsc.vn", results)) is results
    assert db.calls == 0


def test_unreadable_decisions_leave_the_context_unchanged():
    db = _Db(error=RuntimeError("db down"))
    results = [{"article_id": "a"}, {"article_id": "b"}]

    assert _run(apply_conflict_decisions(db, "qnsc.vn", results)) is results


# --- model-found conflicts ------------------------------------------------------------


def _reply(*items):
    return json.dumps({"conflicts": list(items)})


def _item(**overrides):
    base = {
        "other": 1,
        "topic": "Approval time",
        "new_statement": "Approval takes 3 working days.",
        "other_statement": "Approval takes 5 working days.",
        "kind": "numerical",
    }
    return {**base, **overrides}


def test_a_well_formed_reply_becomes_conflicts():
    found = conflict_llm.parse_conflicts(_reply(_item()), 2)

    assert len(found) == 1
    assert found[0].other_index == 1 and found[0].kind == "numerical"
    assert found[0].topic == "Approval time"


def test_code_fences_around_the_json_are_tolerated():
    fenced = "```json\n" + _reply(_item()) + "\n```"

    assert len(conflict_llm.parse_conflicts(fenced, 1)) == 1


@pytest.mark.parametrize(
    "reply",
    ["", "not json", "[]", '{"conflicts": "x"}', '{"other": 1}'],
)
def test_a_malformed_reply_yields_nothing(reply):
    assert conflict_llm.parse_conflicts(reply, 2) == []


def test_items_that_cannot_be_trusted_are_dropped():
    reply = _reply(
        _item(other=9),  # no such article
        _item(other="x"),  # not a number
        _item(topic=""),
        _item(new_statement="Same claim.", other_statement="same claim."),
        _item(other_statement=""),
        "not an object",
    )

    assert conflict_llm.parse_conflicts(reply, 2) == []


def test_an_unknown_kind_falls_back_to_other_and_repeats_are_dropped():
    reply = _reply(_item(kind="weird"), _item(kind="weird"))

    found = conflict_llm.parse_conflicts(reply, 2)

    assert [item.kind for item in found] == ["other"]


def test_no_more_than_the_cap_is_reported():
    items = [_item(topic=f"topic {n}") for n in range(conflict_llm.MAX_CONFLICTS + 4)]

    found = conflict_llm.parse_conflicts(_reply(*items), 2)

    assert len(found) == conflict_llm.MAX_CONFLICTS


def test_detection_needs_a_provider(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)

    async def boom(*_a, **_k):
        raise AssertionError("must not call the model without a provider")

    monkeypatch.setattr(llm_client, "complete", boom)

    assert _run(conflict_llm.detect_llm_conflicts("T", "body", [{"title": "O"}])) == []


def test_a_failing_call_yields_nothing(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fail(*_a, **_k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(llm_client, "complete", fail)

    assert _run(conflict_llm.detect_llm_conflicts("T", "b", [{"title": "O"}])) == []


def test_detection_sends_both_articles_and_parses_the_answer(monkeypatch):
    import src.domain.llm_client as llm_client

    seen = {}
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def fake(messages, **kwargs):
        seen["user"] = messages[1]["content"]
        seen["kwargs"] = kwargs
        return _reply(_item()), 10, "m", "p"

    monkeypatch.setattr(llm_client, "complete", fake)

    found = _run(
        conflict_llm.detect_llm_conflicts(
            "New title", "new body", [{"title": "Old title", "body_md": "old body"}]
        )
    )

    assert len(found) == 1
    assert "New title" in seen["user"] and "Old title" in seen["user"]
    assert seen["kwargs"]["background"] is True and seen["kwargs"]["thinking"] is False


def test_the_setting_can_turn_detection_off(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "CONFLICT_LLM_DETECTION_ENABLED", False)

    assert _run(conflict_llm.detect_llm_conflicts("T", "b", [{"title": "O"}])) == []


# --- stalled jobs ---------------------------------------------------------------------


def test_a_queued_job_past_the_threshold_is_stalled():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    info = stall_info(
        "queued", now - timedelta(minutes=40), threshold_minutes=15, now=now
    )

    assert info == {"stalled": True, "queued_minutes": 40}


def test_a_recent_queued_job_is_just_waiting():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

    assert stall_info(
        "queued", now - timedelta(minutes=2), threshold_minutes=15, now=now
    ) == {"stalled": False, "queued_minutes": 2}


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
def test_only_queued_jobs_can_be_stalled(status):
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

    assert stall_info(
        status, now - timedelta(days=3), threshold_minutes=15, now=now
    ) == {"stalled": False, "queued_minutes": None}


def test_a_naive_timestamp_is_read_as_utc():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

    info = stall_info(
        "queued", datetime(2026, 10, 30, 11, 0), threshold_minutes=15, now=now
    )

    assert info["stalled"] and info["queued_minutes"] == 60


# --- background gate ------------------------------------------------------------------


class _AuditDb:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


def test_steps_run_when_a_provider_exists(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    assert background_gate.background_ai_skip_reason() is None
    assert not _run(
        background_gate.should_skip_background_step(
            _AuditDb(), type("U", (), {"id": 1}), "a1", "entity_extraction"
        )
    )


def test_a_missing_provider_skips_and_says_why(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)
    recorded = []

    class _Audit:
        def __init__(self, _db):
            pass

        async def record(self, *args, **kwargs):
            recorded.append((args, kwargs))

    monkeypatch.setattr("src.repositories.audit.AuditRepository", _Audit)

    skipped = _run(
        background_gate.should_skip_background_step(
            _AuditDb(), type("U", (), {"id": 7}), "a1", "structured_metadata"
        )
    )

    assert skipped is True
    args, kwargs = recorded[0]
    assert args[:2] == (7, "background_ai_skipped")
    assert kwargs["detail"] == {
        "step": "structured_metadata",
        "reason": "no_llm_provider",
    }


def test_a_broken_audit_write_still_reports_the_skip(monkeypatch):
    import src.domain.llm_client as llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)

    class _Broken:
        def __init__(self, _db):
            raise RuntimeError("audit down")

    monkeypatch.setattr("src.repositories.audit.AuditRepository", _Broken)

    assert _run(
        background_gate.should_skip_background_step(
            _AuditDb(), type("U", (), {"id": 1}), "a1", "entity_extraction"
        )
    )


# --- record_lesson --------------------------------------------------------------------


def test_the_lesson_tool_is_a_write_tool_hidden_from_read_only_tokens():
    tool = mcp_tools.TOOLS_BY_NAME["record_lesson"]

    assert tool.write
    assert "record_lesson" not in [
        t.name for t in mcp_tools.visible_tools(can_write=False)
    ]
    assert "record_lesson" in [t.name for t in mcp_tools.visible_tools(can_write=True)]


def test_a_lesson_becomes_a_tagged_department_article():
    call = mcp_tools.TOOLS_BY_NAME["record_lesson"].build(
        {
            "title": "Bus ID width",
            "lesson": "S_BUS IDs are 6 bits.",
            "context": "After the xbar rework.",
            "tags": ["chip", LESSON_TAG],
        }
    )

    assert isinstance(call, Call) and call.method == "POST"
    assert call.path == "/articles/"
    body = call.body
    assert body["title"] == "Bus ID width"
    assert body["body_md"] == (
        "## Lesson\n\nS_BUS IDs are 6 bits.\n\n## Context\n\nAfter the xbar rework."
    )
    assert body["tags"] == ["chip", LESSON_TAG]
    assert body["visibility"] == "department"


def test_a_lesson_without_context_has_only_the_lesson_section():
    call = mcp_tools.TOOLS_BY_NAME["record_lesson"].build(
        {"title": "T", "lesson": "Do X."}
    )

    assert call.body["body_md"] == "## Lesson\n\nDo X."
    assert call.body["tags"] == [LESSON_TAG]


@pytest.mark.parametrize("missing", ["title", "lesson"])
def test_a_lesson_needs_a_title_and_text(missing):
    args = {"title": "T", "lesson": "Do X."}
    del args[missing]

    with pytest.raises(ToolInputError):
        mcp_tools.TOOLS_BY_NAME["record_lesson"].build(args)


# --- recording model-found conflicts --------------------------------------------------


class _ScalarRows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _RecordDb:
    def __init__(self, known=None, title="New article"):
        self.known = known or []
        self.title = title
        self.added = []

    async def get(self, _model, _id):
        return type("A", (), {"title": self.title})()

    async def execute(self, _stmt):
        return _ScalarRows(self.known)

    def add(self, obj):
        self.added.append(obj)


_OTHERS = [{"article_id": "b-id", "title": "Old article", "context_text": "old"}]
_USER = type("U", (), {"company_domain": "qnsc.vn"})()


def _found():
    return [
        conflict_llm.LlmConflict(
            other_index=1,
            topic="Approval time",
            new_statement="3 days",
            other_statement="5 days",
            kind="numerical",
        )
    ]


def _patch_detection(monkeypatch, found):
    async def fake(*_a, **_k):
        return found

    monkeypatch.setattr(conflict_llm, "detect_llm_conflicts", fake)


def _record(db, exclude="a-id", others=None):
    from src.domain.contradiction_check import _record_llm_conflicts

    return _run(
        _record_llm_conflicts(db, _USER, "new body", exclude, others or _OTHERS)
    )


def test_a_model_found_conflict_is_recorded_with_both_statements(monkeypatch):
    _patch_detection(monkeypatch, _found())
    db = _RecordDb()

    recorded = _record(db)

    assert len(recorded) == 1 and len(db.added) == 1
    row = db.added[0]
    assert row.detection_method == "llm" and row.contradiction_type == "numerical"
    assert row.article_ids == ["a-id", "b-id"]
    assert [e["value"] for e in row.evidence] == ["3 days", "5 days"]
    assert row.company_domain == "qnsc.vn" and row.fact == "Approval time"


def test_a_conflict_already_seen_for_the_same_articles_is_not_recorded_again(
    monkeypatch,
):
    _patch_detection(monkeypatch, _found())
    # open or resolved, article order irrelevant
    db = _RecordDb(known=[("Approval time", ["b-id", "a-id"])])

    assert _record(db) == [] and db.added == []


def test_a_reworded_topic_for_the_same_articles_is_still_the_same_conflict(
    monkeypatch,
):
    _patch_detection(monkeypatch, _found())
    db = _RecordDb(known=[("Approval duration time", ["a-id", "b-id"])])

    assert _record(db) == [] and db.added == []


def test_an_unrelated_topic_for_the_same_articles_is_a_new_conflict(monkeypatch):
    _patch_detection(monkeypatch, _found())
    db = _RecordDb(known=[("Parental leave weeks", ["a-id", "b-id"])])

    assert len(_record(db)) == 1


def test_topic_overlap_ignores_case_and_short_words():
    from src.domain.contradiction_check import same_topic

    assert same_topic("Approval TIME", "approval time")
    assert same_topic("Approval time", "The approval time")
    assert not same_topic("Approval time", "Parental leave")
    assert same_topic("", "")


def test_the_same_topic_for_other_articles_is_a_new_conflict(monkeypatch):
    _patch_detection(monkeypatch, _found())
    db = _RecordDb(known=[("Approval time", ["x-id", "y-id"])])

    assert len(_record(db)) == 1


def test_a_draft_with_no_stored_article_records_nothing(monkeypatch):
    _patch_detection(monkeypatch, _found())
    db = _RecordDb()

    assert _record(db, exclude=None) == [] and db.added == []


# --- resolving with a winner ----------------------------------------------------------


class _ResolveDb:
    def __init__(self, item):
        self.item, self.commits = item, 0

    async def scalar(self, _stmt):
        return self.item

    async def commit(self):
        self.commits += 1


def _conflict():
    from src.models.governance import ConflictRecord

    return ConflictRecord(
        company_domain="qnsc.vn",
        fact="Approval time",
        article_ids=["a-id", "b-id"],
        status="open",
    )


def _resolve(item, user, **body):
    from src.api.routers import knowledge

    request = knowledge.ConflictResolutionRequest(**body)
    return _run(
        knowledge.resolve_conflict(
            "11111111-1111-1111-1111-111111111111", request, user, _ResolveDb(item)
        )
    )


def _user(role):
    return type(
        "U",
        (),
        {"id": "u1", "role": role, "roles": [], "company_domain": "qnsc.vn"},
    )()


def test_an_admin_can_name_the_article_that_is_right(monkeypatch):
    from src.domain import events

    published = []

    async def fake_publish(name, payload):
        published.append((name, payload["article_id"]))

    monkeypatch.setattr(events.event_bus, "publish", fake_publish)

    class _Audit:
        def __init__(self, _db):
            pass

        async def record(self, *a, **k):
            return None

    monkeypatch.setattr("src.repositories.audit.AuditRepository", _Audit)
    item = _conflict()

    result = _resolve(item, _user("Admin"), keep_article_id="a-id")

    assert result["status"] == "resolved" and result["kept_article_id"] == "a-id"
    assert item.kept_article_id == "a-id" and item.resolved_at is not None
    assert published == [("ConflictResolved", "a-id"), ("ConflictResolved", "b-id")]


def test_the_winner_must_be_one_of_the_conflicting_articles():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        _resolve(_conflict(), _user("Admin"), keep_article_id="zzz")

    assert caught.value.status_code == 422


def test_only_an_admin_or_ceo_can_pick_a_winner():
    from fastapi import HTTPException

    item = _conflict()
    with pytest.raises(HTTPException) as caught:
        _resolve(item, _user("Manager"), keep_article_id="a-id")

    assert caught.value.status_code == 403 and item.status == "open"


def test_resolving_without_a_winner_still_needs_a_note():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        _resolve(_conflict(), _user("Manager"))

    assert caught.value.status_code == 422


# --- reopening, and the order of audit and events -------------------------------------


def _reopen(item, user):
    from src.api.routers import knowledge

    return _run(
        knowledge.reopen_conflict(
            "11111111-1111-1111-1111-111111111111", user, _ResolveDb(item)
        )
    )


def _decided(kept="a-id"):
    item = _conflict()
    item.status = "resolved"
    item.kept_article_id = kept
    item.resolution_note = "chosen"
    return item


def _quiet_side_effects(monkeypatch, order=None):
    from src.domain import events

    async def fake_publish(name, payload):
        if order is not None:
            order.append("event")

    monkeypatch.setattr(events.event_bus, "publish", fake_publish)

    class _Audit:
        def __init__(self, _db):
            pass

        async def record(self, *a, **k):
            if order is not None:
                order.append(("audit", k.get("commit")))

    monkeypatch.setattr("src.repositories.audit.AuditRepository", _Audit)


def test_only_an_admin_or_ceo_can_undo_a_chosen_winner(monkeypatch):
    from fastapi import HTTPException

    _quiet_side_effects(monkeypatch)
    item = _decided()

    with pytest.raises(HTTPException) as caught:
        _reopen(item, _user("Manager"))

    assert caught.value.status_code == 403
    assert item.status == "resolved" and item.kept_article_id == "a-id"


def test_an_admin_reopening_clears_the_choice(monkeypatch):
    _quiet_side_effects(monkeypatch)
    item = _decided()

    result = _reopen(item, _user("Admin"))

    assert result["status"] == "open" and item.kept_article_id is None
    assert item.resolved_at is None and item.resolution_note is None


def test_anyone_with_access_can_reopen_a_conflict_closed_with_only_a_note(monkeypatch):
    _quiet_side_effects(monkeypatch)
    item = _decided(kept=None)

    assert _reopen(item, _user("Manager"))["status"] == "open"


def test_the_audit_row_is_written_before_the_commit_and_events_after(monkeypatch):
    order: list = []
    _quiet_side_effects(monkeypatch, order)

    class _OrderedDb(_ResolveDb):
        async def commit(self):
            order.append("commit")

    from src.api.routers import knowledge

    _run(
        knowledge.resolve_conflict(
            "11111111-1111-1111-1111-111111111111",
            knowledge.ConflictResolutionRequest(keep_article_id="a-id"),
            _user("Admin"),
            _OrderedDb(_conflict()),
        )
    )

    assert order == [("audit", False), "commit", "event", "event"]


def test_a_failing_cache_invalidation_does_not_fail_a_stored_decision(monkeypatch):
    from src.domain import events

    async def broken(*_a, **_k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(events.event_bus, "publish", broken)

    class _Audit:
        def __init__(self, _db):
            pass

        async def record(self, *a, **k):
            return None

    monkeypatch.setattr("src.repositories.audit.AuditRepository", _Audit)
    item = _conflict()

    result = _resolve(item, _user("Admin"), keep_article_id="a-id")

    assert result["status"] == "resolved"


def test_the_governance_repository_exposes_the_session_conflict_decisions_use():
    # ai_service reads the session off the repository; a rename must fail here, not
    # silently switch conflict decisions off.
    from src.repositories.governance import GovernanceRepository

    sentinel = object()

    assert GovernanceRepository(sentinel).db is sentinel
