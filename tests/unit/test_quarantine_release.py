"""A quarantine must be releasable when the failure was ours, not the file's.

A document that fails to ingest three times on one revision stops being retried. Only a
new revision at the provider clears that, and `_is_quarantined` explains why: replacing
the broken file at the source is how a person fixes it, and making them find an admin
screen as well would be worse.

That reasoning holds when the file is the problem. It does not hold when the failure was
a server-side defect: it quarantines every document it touches, and once it is fixed
there is nothing at the provider for anyone to repair -- the files were always fine.

Exactly that happened. A greenlet error in the connector's draft actor failed every
document that reached ingest, three times each, so a 241-file sync reported "241 checked,
skipped" with an empty corpus behind it. Fixing the defect could not bring them back:
`_is_quarantined` still matched the recorded revision, so the next sync skipped them all
again, and the run read like a problem with the documents.

These tests pin the release, and pin that it does not quietly widen into "retry
everything" -- a document that has NOT exhausted its attempts must keep them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.cloud_sync import (
    _MAX_ITEM_ATTEMPTS,
    _clear_ingest_failure,
    _is_quarantined,
)


@dataclass
class _Document:
    metadata_json: dict | None = field(default_factory=dict)


@dataclass
class _Change:
    revision: str | None


def _failed(revision: str, attempts: int) -> _Document:
    return _Document(
        {
            "ingest_failure": {
                "revision": revision,
                "attempts": attempts,
                "reason": "greenlet_spawn has not been called",
            },
            "provider_acl_present": True,
        }
    )


def test_an_exhausted_document_is_skipped():
    """The state the corpus was left in."""
    assert _is_quarantined(_failed("rev-1", _MAX_ITEM_ATTEMPTS), _Change("rev-1"))


def test_releasing_makes_the_next_sync_try_it_again():
    """The point of the endpoint: no revision changed at the provider, and it still runs."""
    document = _failed("rev-1", _MAX_ITEM_ATTEMPTS)
    _clear_ingest_failure(document)
    assert not _is_quarantined(document, _Change("rev-1"))


def test_releasing_keeps_the_rest_of_the_metadata():
    """metadata_json also carries ACL state; dropping it would re-open a document's
    permissions as a side effect of retrying its content."""
    document = _failed("rev-1", _MAX_ITEM_ATTEMPTS)
    _clear_ingest_failure(document)
    assert document.metadata_json == {"provider_acl_present": True}


def test_releasing_a_document_that_never_failed_changes_nothing():
    document = _Document({"provider_acl_present": True})
    _clear_ingest_failure(document)
    assert document.metadata_json == {"provider_acl_present": True}


def test_a_document_with_attempts_left_is_still_tried():
    """Release must not be the only thing keeping retries alive."""
    assert not _is_quarantined(_failed("rev-1", _MAX_ITEM_ATTEMPTS - 1), _Change("rev-1"))


def test_a_new_revision_still_clears_it_by_itself():
    """The original escape hatch keeps working: fixing the file at the source is enough."""
    assert not _is_quarantined(_failed("rev-1", _MAX_ITEM_ATTEMPTS), _Change("rev-2"))


def test_the_release_endpoint_is_registered():
    """It is the only way back for a corpus quarantined by a defect.

    Asserted on the connectors router rather than the assembled app. The app's route
    table depends on import order across the whole test session -- read from there this
    same assertion passed locally and failed in CI, seeing an app carrying only
    FastAPI's built-in routes. The router is the thing this change actually adds to,
    and reading it directly is both stable and closer to the edit.
    """
    from src.api.routers.connectors import router

    routes = {
        getattr(route, "path", ""): getattr(route, "methods", set()) for route in router.routes
    }
    assert "POST" in routes.get("/{connector_id}/retry-failed", set()), sorted(routes)
