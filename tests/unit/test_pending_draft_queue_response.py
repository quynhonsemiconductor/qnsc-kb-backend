from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import uuid

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from src.api.routers.governance import (
    _draft_response,
    _queue_draft_restructure,
    _visible_restructure_state,
)
from src.models.governance import PendingDraft
from src.repositories.governance import GovernanceRepository


class _Service:
    @staticmethod
    def _can_review_draft(_user, _draft):
        return True


def _draft(**overrides):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    values = {
        "id": uuid.uuid4(),
        "title": "Draft",
        "company_domain": "example.test",
        "dept": "Engineering",
        "source_ref": "upload.pdf",
        "source_hash": "a" * 64,
        "summary": "original body",
        "restructured_body_md": "# Prepared",
        "restructure_candidate_md": "# Candidate",
        "restructure_decision": "pending_review",
        "restructure_status": "llm",
        "restructure_model": "model",
        "restructure_error": None,
        "status": "pending",
        "created_by": uuid.uuid4(),
        "assigned_approver_id": None,
        "assigned_by": None,
        "assigned_at": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "created_at": now,
        "updated_at": now,
        "similarity_level": None,
        "similarity_matches": None,
        "requires_update_confirmation": False,
        "related_article_ids": None,
        "tags": None,
        "content_metadata": {},
        "external_document_id": None,
        "candidates": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stale_queue_and_processing_jobs_become_actionable():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    queued = _draft(restructure_status="queued", updated_at=now - timedelta(minutes=6))
    processing = _draft(restructure_status="processing", updated_at=now - timedelta(minutes=66))

    assert _visible_restructure_state(queued, now)[0] == "stalled"
    assert _visible_restructure_state(processing, now)[0] == "stalled"


def test_recent_queue_and_processing_jobs_remain_in_flight():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    queued = _draft(restructure_status="queued", updated_at=now - timedelta(minutes=4))
    processing = _draft(restructure_status="processing", updated_at=now - timedelta(minutes=64))

    assert _visible_restructure_state(queued, now)[0] == "queued"
    assert _visible_restructure_state(processing, now)[0] == "processing"


def test_retrying_a_stalled_queued_row_refreshes_its_heartbeat():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    old = now - timedelta(minutes=10)
    draft = _draft(
        restructure_status="queued",
        restructure_error="stale",
        updated_at=old,
    )

    _queue_draft_restructure(draft, now)

    assert draft.updated_at == now
    assert draft.restructure_status == "queued"
    assert draft.restructure_error is None
    assert _visible_restructure_state(draft, now)[0] == "queued"


def test_list_response_omits_document_bodies_but_keeps_layout_signal():
    draft = _draft()
    user = SimpleNamespace(id=uuid.uuid4())

    result = _draft_response(draft, _Service(), user, include_content=False)

    assert result["has_restructure_candidate"] is True
    assert "summary" not in result
    assert "restructured_body_md" not in result
    assert "restructure_candidate_md" not in result
    assert "restructure_report" not in result


def test_detail_response_includes_authorized_document_bodies():
    draft = _draft()
    user = SimpleNamespace(id=uuid.uuid4())

    result = _draft_response(draft, _Service(), user, include_content=True)

    assert result["summary"] == "original body"
    assert result["restructured_body_md"] == "# Prepared"
    assert result["restructure_candidate_md"] == "# Candidate"
    assert result["restructure_report"] is not None


def test_inbox_projection_excludes_large_and_sensitive_document_columns():
    statement = select(PendingDraft).options(
        *GovernanceRepository._draft_inbox_options()
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))

    for column in (
        "summary",
        "restructured_body_md",
        "restructure_candidate_md",
        "page_texts",
        "storage_key",
        "review_note",
    ):
        assert f"pending_drafts.{column}" not in sql
