"""Deleting an article must let its document be uploaded again.

Every article was deleted, and re-uploading the same PDFs still returned

    409 {"code": "duplicate_document", "message": "This document already exists."}

while a file never seen before uploaded fine. The documents were unrecoverably stuck:
nothing in the product could clear them.

`IngestionFingerprint` reserves a (tenant, source_hash) pair to close the concurrent
upload race. It is created `pending` at upload and flipped to `approved` with an
`article_id` at approval — and then **nothing ever deletes it**. Not
`soft_delete_article`, not rejection. Article deletion is a SOFT delete, so the
reservation outlives the article and refuses the file forever.

The duplicate check sitting immediately below it in the same function already gets this
right, and says so:

    # Article deletion is soft-delete. Ignore source rows belonging to deleted
    # or inactive articles so a document can be uploaded again after removal.

Two gates on the same decision, one soft-delete-aware and one not — the same shape as the
department bug in #78.

The reservation must still block a LIVE draft or article. That is its actual job, and
these tests pin both directions.
"""
from __future__ import annotations

import asyncio
import uuid

from src.api.routers import articles as articles_router


class _Article:
    def __init__(self, status="published", lifecycle_status="active"):
        self.id = uuid.uuid4()
        self.status = status
        self.lifecycle_status = lifecycle_status


class _Draft:
    def __init__(self, status="draft"):
        self.id = uuid.uuid4()
        self.status = status


class _Fingerprint:
    def __init__(self, status, article=None, draft=None):
        self.status = status
        self.article_id = article.id if article else None
        self.draft_id = draft.id if draft else None


class _DB:
    """Resolves db.get(Model, id) from the objects it was built with."""

    def __init__(self, *objects):
        self._by_id = {obj.id: obj for obj in objects}

    async def get(self, _model, identifier):
        return self._by_id.get(identifier)


def _guards(db, fingerprint) -> bool:
    return asyncio.run(
        articles_router._reservation_guards_live_content(db, fingerprint)
    )


# ── the reported bug ──────────────────────────────────────────────────────────


def test_a_soft_deleted_article_releases_its_reservation():
    """The exact case: delete the article, re-upload the same file."""
    article = _Article(status="deleted")
    fingerprint = _Fingerprint("approved", article=article)
    assert _guards(_DB(article), fingerprint) is False


def test_an_inactive_article_releases_its_reservation():
    article = _Article(lifecycle_status="archived")
    fingerprint = _Fingerprint("approved", article=article)
    assert _guards(_DB(article), fingerprint) is False


def test_a_vanished_article_releases_its_reservation():
    """ondelete=SET NULL and hard deletes both leave the row pointing at nothing."""
    fingerprint = _Fingerprint("approved", article=_Article())
    assert _guards(_DB(), fingerprint) is False  # article not in the database


def test_an_approved_reservation_without_an_article_is_orphaned():
    fingerprint = _Fingerprint("approved")
    assert _guards(_DB(), fingerprint) is False


def test_a_rejected_draft_releases_its_reservation():
    draft = _Draft(status="rejected")
    fingerprint = _Fingerprint("pending", draft=draft)
    assert _guards(_DB(draft), fingerprint) is False


# ── what must STILL block ─────────────────────────────────────────────────────


def test_a_live_article_still_blocks():
    """The reservation's real job: refuse a genuine re-upload."""
    article = _Article()
    fingerprint = _Fingerprint("approved", article=article)
    assert _guards(_DB(article), fingerprint) is True


def test_a_draft_awaiting_review_still_blocks():
    """The document is in the queue; uploading it again would duplicate the review."""
    draft = _Draft(status="draft")
    fingerprint = _Fingerprint("pending", draft=draft)
    assert _guards(_DB(draft), fingerprint) is True


def test_a_submitted_draft_still_blocks():
    draft = _Draft(status="pending")
    fingerprint = _Fingerprint("pending", draft=draft)
    assert _guards(_DB(draft), fingerprint) is True


def test_an_uploading_reservation_is_handled_by_its_own_branch():
    """`uploading` has separate handling that predates this and must not be captured
    here, or an abandoned browser-to-R2 attempt changes behaviour."""
    assert _guards(_DB(), _Fingerprint("uploading")) is False
