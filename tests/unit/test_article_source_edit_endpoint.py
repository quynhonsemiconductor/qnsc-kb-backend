"""The endpoint that corrects text inside a stored original.

What matters here is not the file editing (test_source_edit.py) but what surrounds it: an edit
must go through the same permission and approval path as any other, must never store a file
for a change that did not apply, and must clean up the file it stored if the draft could not be
created. Plain-text sources are used so these tests need no document libraries.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routers import articles as router
from src.api.routers.articles import (
    SourceEditItem,
    SourceEditRequest,
    edit_article_source,
    preview_source_edit,
)

ARTICLE_ID = uuid.uuid4()
FILE = b"Muc luong 1234 dong. Hom nay 1234."


class _Result:
    def __init__(self, source):
        self._source = source

    def scalars(self):
        return self

    def first(self):
        return self._source


class _DB:
    def __init__(self, source):
        self.source = source

    async def execute(self, _statement):
        return _Result(self.source)


def _article(body="Muc luong 1234 dong.", version=3):
    return SimpleNamespace(
        id=ARTICLE_ID,
        body_md=body,
        version=version,
        company_domain="acme.test",
        title="Quy dinh",
    )


def _source(filename="quy-dinh.txt"):
    return SimpleNamespace(
        storage_key="s3://bucket/key/quy-dinh.txt", original_filename=filename
    )


def _body(*pairs, version=None):
    return SourceEditRequest(
        replacements=[SourceEditItem(find=a, replace=b) for a, b in pairs],
        version=version,
    )


@pytest.fixture
def world(monkeypatch):
    state = SimpleNamespace(
        article=_article(),
        can_edit=True,
        stored=[],
        deleted=[],
        submitted=[],
        save_error=None,
        submit_error=None,
        loaded=FILE,
    )

    class _Service:
        def __init__(self, *args, **kwargs):
            pass

        async def get_article(self, user, article_id):
            return state.article

    monkeypatch.setattr(router, "ArticleService", _Service)
    monkeypatch.setattr(router, "ArticleRepository", lambda db: None)
    monkeypatch.setattr(router, "UserRepository", lambda db: None)
    monkeypatch.setattr(
        router.PermissionService,
        "can_edit_article",
        staticmethod(lambda user, article: state.can_edit),
    )
    monkeypatch.setattr(router, "load_source", lambda key: state.loaded)
    monkeypatch.setattr(
        router,
        "extract_source_pages",
        lambda name, data: [{"page_number": 1, "text": data.decode()}],
    )
    monkeypatch.setattr(
        router,
        "extract_source_markdown",
        lambda name, data, pages=None: "REREAD:" + data.decode(),
    )

    def save(file_hash, filename, data, company_domain=None):
        if state.save_error:
            raise state.save_error
        state.stored.append((file_hash, filename, data, company_domain))
        return f"s3://bucket/{file_hash}/{filename}"

    monkeypatch.setattr(router, "save_source", save)
    monkeypatch.setattr(router, "delete_source", lambda key: state.deleted.append(key))

    async def submit(id, article_in, current_user, db, source_attachment=None):
        if state.submit_error:
            raise state.submit_error
        state.submitted.append((article_in, source_attachment))
        return router.DraftSubmissionResponse(
            id=uuid.uuid4(),
            title="Quy dinh",
            status="draft",
            workflow="pending_approval",
            message="submitted",
        )

    monkeypatch.setattr(router, "_submit_article_update", submit)
    state.db = _DB(_source())
    state.user = SimpleNamespace(id=uuid.uuid4())
    return state


def run(coro):
    return asyncio.run(coro)


def status_of(excinfo):
    return excinfo.value.status_code


def test_preview_counts_and_stores_nothing(world):
    result = run(
        preview_source_edit(ARTICLE_ID, _body(("1234", "1244")), world.user, world.db)
    )
    assert result.counts == [2]
    assert world.stored == [] and world.submitted == []


def test_preview_reports_an_unsupported_file_as_422(world):
    world.db = _DB(_source("scan.png"))
    with pytest.raises(HTTPException) as error:
        run(preview_source_edit(ARTICLE_ID, _body(("a", "b")), world.user, world.db))
    assert status_of(error) == 422 and "pixels" in error.value.detail


def test_edit_submits_the_corrected_file_through_the_draft_workflow(world):
    response = run(
        edit_article_source(ARTICLE_ID, _body(("1234", "1244")), world.user, world.db)
    )
    ((article_in, attachment),) = world.submitted
    assert article_in.body_md == "Muc luong 1244 dong."
    assert article_in.version == 3
    assert attachment["original_filename"] == "quy-dinh.txt"
    assert attachment["storage_key"].startswith("s3://bucket/")
    assert attachment["page_texts"][0]["text"] == "Muc luong 1244 dong. Hom nay 1244."
    ((file_hash, _, stored, company),) = world.stored
    assert stored == b"Muc luong 1244 dong. Hom nay 1244." and company == "acme.test"
    assert response.workflow == "pending_approval"
    assert response.counts == [2] and response.text_regenerated is False


def test_the_reading_text_is_reread_when_it_does_not_contain_the_change(world):
    world.article = _article(body="Reformatted: **Muc luong** mot nghin")
    response = run(
        edit_article_source(ARTICLE_ID, _body(("1234", "1244")), world.user, world.db)
    )
    ((article_in, _),) = world.submitted
    assert article_in.body_md.startswith("REREAD:")
    assert response.text_regenerated is True


def test_a_change_that_is_not_in_the_file_stores_and_submits_nothing(world):
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID,
                _body(("1234", "1244"), ("khong co", "x")),
                world.user,
                world.db,
            )
        )
    assert status_of(error) == 422
    assert error.value.detail["code"] == "text_not_found"
    assert error.value.detail["missing"] == ["khong co"]
    assert world.stored == [] and world.submitted == []


def test_someone_who_may_not_edit_is_refused_before_anything_is_read(world):
    world.can_edit = False
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244")), world.user, world.db
            )
        )
    assert status_of(error) == 403 and world.stored == []


def test_a_stale_version_is_refused(world):
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244"), version=2), world.user, world.db
            )
        )
    assert status_of(error) == 409 and world.stored == []


def test_an_article_with_no_stored_original_is_404(world):
    world.db = _DB(None)
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244")), world.user, world.db
            )
        )
    assert status_of(error) == 404


def test_a_missing_file_in_storage_is_404(world, monkeypatch):
    def gone(key):
        raise FileNotFoundError(key)

    monkeypatch.setattr(router, "load_source", gone)
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244")), world.user, world.db
            )
        )
    assert status_of(error) == 404


def test_a_storage_failure_is_503_and_creates_no_draft(world):
    world.save_error = RuntimeError("R2 down")
    with pytest.raises(HTTPException) as error:
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244")), world.user, world.db
            )
        )
    assert status_of(error) == 503 and world.submitted == []


def test_the_stored_file_is_deleted_when_the_draft_cannot_be_created(world):
    world.submit_error = HTTPException(status_code=409, detail="duplicate")
    with pytest.raises(HTTPException):
        run(
            edit_article_source(
                ARTICLE_ID, _body(("1234", "1244")), world.user, world.db
            )
        )
    assert len(world.stored) == 1
    assert world.deleted == [f"s3://bucket/{world.stored[0][0]}/quy-dinh.txt"]


def test_a_name_without_an_extension_falls_back_to_the_storage_key(world):
    world.db = _DB(_source("Quy dinh"))
    run(edit_article_source(ARTICLE_ID, _body(("1234", "1244")), world.user, world.db))
    assert world.stored[0][1] == "quy-dinh.txt"


def test_requests_are_bounded():
    with pytest.raises(ValueError):
        SourceEditRequest(replacements=[])
    with pytest.raises(ValueError):
        SourceEditItem(find="", replace="x")
    with pytest.raises(ValueError):
        SourceEditItem(find="a" * 501, replace="x")


def test_both_routes_are_registered():
    paths = [route.path for route in router.router.routes]
    assert "/{id}/source-edits" in paths
    assert "/{id}/source-edits/preview" in paths
