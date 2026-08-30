"""An uploaded draft must be approvable, which means its `dept` must be organisational.

A draft uploaded into the access group "public" was created happily and could then never
be approved. Every attempt returned

    422 {"detail": "Department does not exist or is inactive"}

about a department that plainly existed and was active — the department listing showed
`{"name": "public", "active": true, "kind": "access"}`.

Two rules disagreed. `department_ids` is an AUDIENCE selection: `Article.departments`
joins it with `kind == "access"`, and permissions.py / rbac.py read only access-kind rows.
`dept` is an ORGANISATIONAL department, and `approve_draft` re-resolves it through
`resolve_active_department`, which requires `kind == "org"`. Both upload paths took the
primary from `selected_departments[0]`, so the audience's name landed in `dept` and the
draft was born unapprovable.

Nothing caught it because each half is correct alone: the upload validated the selection,
and approve validated the name. Only the round trip was broken.

`asyncio.run` rather than pytest-asyncio, matching test_query_authorization.py — the
marker is not available in every environment this suite runs in.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from src.api.routers import articles as articles_router
from src.domain.departments import resolve_active_department


class _Department:
    def __init__(self, name: str, kind: str, active: bool = True):
        self.id = uuid.uuid4()
        self.name = name
        self.kind = kind
        self.active = active


class _User:
    company_domain = "qnsc.vn"
    dept = "Nhan su"


ORG = _Department("Nhan su", "org")
ACCESS_PUBLIC = _Department("public", "access")
ACCESS_HR = _Department("dept_nhan su", "access")
ALL = (ORG, ACCESS_PUBLIC, ACCESS_HR)


@pytest.fixture
def resolvers(monkeypatch):
    """Stand in for the two resolvers, preserving the kind rule that matters."""

    async def resolve_active_departments(db, company_domain, ids):
        by_id = {d.id: d for d in ALL}
        return [by_id[i] for i in ids]

    async def resolve_one(db, company_domain, name, required=True):
        # The real one filters kind == "org", so an access name misses.
        for candidate in ALL:
            if candidate.name.lower() == str(name).lower() and candidate.kind == "org":
                return candidate
        raise HTTPException(
            status_code=422, detail="Department does not exist or is inactive"
        )

    monkeypatch.setattr(
        articles_router, "resolve_active_departments", resolve_active_departments
    )
    monkeypatch.setattr(articles_router, "resolve_active_department", resolve_one)


def _resolve(dept, ids):
    return asyncio.run(
        articles_router._resolve_upload_departments(object(), _User(), dept, ids)
    )


def test_an_access_only_selection_still_yields_an_org_dept(resolvers):
    """The exact reported case: upload into "public" alone."""
    primary, audiences = _resolve(None, [ACCESS_PUBLIC.id])

    assert primary.kind == "org", "dept must be organisational or approve_draft rejects it"
    assert primary.name == "Nhan su"
    # The audience selection is preserved untouched — it is a different concept.
    assert [d.id for d in audiences] == [ACCESS_PUBLIC.id]


def test_an_org_department_in_the_selection_becomes_the_primary(resolvers):
    primary, audiences = _resolve(None, [ACCESS_PUBLIC.id, ORG.id])
    assert primary.name == "Nhan su"
    assert len(audiences) == 2


def test_an_explicit_org_dept_wins(resolvers):
    primary, _ = _resolve("Nhan su", [ACCESS_PUBLIC.id])
    assert primary.name == "Nhan su"


def test_an_explicit_access_dept_is_rejected_at_upload(resolvers):
    """Better to refuse the upload than to create a draft that can never be approved."""
    with pytest.raises(HTTPException) as raised:
        _resolve("public", [ACCESS_PUBLIC.id])
    assert raised.value.status_code == 422


def test_no_selection_keeps_the_previous_shape(resolvers):
    primary, audiences = _resolve(None, None)
    assert primary.name == "Nhan su"
    assert audiences == [primary]


def test_multipart_ids_are_still_parsed_from_json():
    ids = articles_router._parse_department_ids(f'["{ACCESS_PUBLIC.id}"]')
    assert ids == [ACCESS_PUBLIC.id]
    assert articles_router._parse_department_ids(None) is None
    with pytest.raises(HTTPException):
        articles_router._parse_department_ids("not json")


class _DB:
    """Returns the org-filtered query's result, then the name-only query's."""

    def __init__(self, *results):
        self._results = list(results)

    async def scalar(self, _statement):
        return self._results.pop(0)


def test_an_access_group_name_is_reported_as_such():
    """The message said "does not exist or is inactive" about a department that existed
    and was active — which is what sent the investigation to CloudWatch."""
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            resolve_active_department(_DB(None, ACCESS_PUBLIC), "qnsc.vn", "public")
        )

    assert raised.value.status_code == 422
    assert "access group" in raised.value.detail
    assert "does not exist" not in raised.value.detail


def test_a_genuinely_missing_department_still_says_so():
    with pytest.raises(HTTPException) as raised:
        asyncio.run(resolve_active_department(_DB(None, None), "qnsc.vn", "nope"))

    assert raised.value.detail == "Department does not exist or is inactive"
