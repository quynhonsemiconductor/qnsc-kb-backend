"""A reviewer must be able to read the document itself, not only its extracted text.

The review screen rendered Markdown and nothing else, so an upload that is mostly
screenshots lost almost everything it showed and was approved unseen. This endpoint
streams the stored upload for exactly the reviewers who may already see the draft.

A pending draft is NOT an Article, so none of the article authorization applies here:
visibility comes from `get_draft_for_user`, and a reviewer who cannot see the draft must
get 404 rather than its bytes. Most of what follows pins that, plus the one failure mode
that is reached by an ordinary action — rejecting a manual upload deletes its object from
the bucket while the draft row keeps the key, so the file can be legitimately absent.
"""
from __future__ import annotations

import inspect

from src.api.routers import governance
from src.domain.source_storage import SOURCE_VIEW_SECURITY_HEADERS


def _endpoint_source() -> str:
    return inspect.getsource(governance.view_pending_draft_source)


def test_the_route_is_registered_without_shadowing_a_literal_path():
    """`/pending-drafts/{id}/source`, never a bare `/{id}`.

    A bare parameterised route in this router would swallow literal siblings such as
    `/bulk-decide`, which is why the detail endpoint is spelled `/{id}/detail`.
    """
    paths = {
        route.path: sorted(route.methods)
        for route in governance.router.routes
        if "pending-drafts" in getattr(route, "path", "")
    }
    assert paths.get("/pending-drafts/{id}/source") == ["GET"]
    assert "/pending-drafts/bulk-decide" in paths


def test_visibility_comes_from_the_draft_rule_not_the_article_rule():
    """The draft's own visibility check, the same one the detail endpoint uses."""
    source = _endpoint_source()
    assert "get_draft_for_user" in source
    assert 'require_permission("governance.read")' in inspect.getsource(governance)


def test_an_invisible_draft_is_not_found_rather_than_forbidden():
    """404, so the endpoint never confirms that a draft the caller may not see exists."""
    source = _endpoint_source()
    missing_draft = source.index("if not draft:")
    assert "status_code=404" in source[missing_draft : missing_draft + 200]


def test_a_draft_with_no_stored_upload_is_not_found():
    """Manually authored drafts have no file at all; that is not an error state."""
    source = _endpoint_source()
    no_key = source.index("if not draft.storage_key:")
    assert "status_code=404" in source[no_key : no_key + 200]


def test_a_deleted_object_is_reported_as_missing_not_as_a_server_fault():
    """The ordinary path into this: rejecting a manual upload deletes its object.

    `load_source` now translates the bucket's NoSuchKey into FileNotFoundError, which
    this endpoint already turns into a 404. Before that translation the same request
    produced a 500 and the UI read it as "no original file is stored", which was wrong
    — the draft had a key, the bytes were simply gone.
    """
    source = _endpoint_source()
    assert "except FileNotFoundError" in source
    storage = inspect.getsource(__import__("src.domain.source_storage", fromlist=["x"]).load_source)
    assert "NoSuchKey" in storage
    assert "FileNotFoundError" in storage


def test_the_uploaded_type_is_never_echoed_back():
    """The allow-list decides the response type; the uploader's header never does.

    An HTML content type would execute in this application's origin when a reviewer
    opens the file.
    """
    source = _endpoint_source()
    assert "safe_source_media_type" in source
    assert "draft.mime_type" not in source


def test_the_response_carries_the_shared_source_view_headers():
    source = _endpoint_source()
    assert "SOURCE_VIEW_SECURITY_HEADERS" in source
    assert "allow-same-origin" not in SOURCE_VIEW_SECURITY_HEADERS["Content-Security-Policy"]


def test_the_filename_is_sanitised_before_it_reaches_a_header():
    """The uploader chose this name; it must not be able to inject header syntax."""
    source = _endpoint_source()
    assert 're.sub(r"[^A-Za-z0-9._-]+"' in source


def test_one_page_can_be_requested_as_text():
    """The citation path: a single extracted page, not the whole binary."""
    source = _endpoint_source()
    assert 'media_type="text/plain"' in source
    assert "page_number" in source
    page_missing = source.index("if selected_page is None:")
    assert "status_code=404" in source[page_missing : page_missing + 200]


def test_the_security_headers_have_exactly_one_definition():
    """Two copies of a security allow-list eventually disagree, and that is the hole."""
    from src.api.routers import articles

    assert articles.SOURCE_VIEW_SECURITY_HEADERS is SOURCE_VIEW_SECURITY_HEADERS
