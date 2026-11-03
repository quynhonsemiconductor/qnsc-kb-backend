"""An uploaded draft must be approvable, which means its `dept` must be a real one.

A draft uploaded into "public" was created happily and could then never be approved.
Every attempt returned

    422 {"detail": "Department does not exist or is inactive"}

Both upload paths took the primary from `selected_departments[0]` and wrote that name
into `Article.dept` without validating it as a department in its own right, while
`approve_draft` re-resolves `dept` through `resolve_active_department`. Nothing caught
it because each half is correct alone: the upload validated the selection, and approve
validated the name. Only the round trip was broken.

`department_ids` is an AUDIENCE selection (`Article.departments`); `Article.dept` is the
single PRIMARY department that drives approval routing. Whatever lands in `dept` must be
a real, active, tenant-owned department, and these are the invariants that guarantee it:

  * an explicit `dept` wins, else the first selected department, else the uploader's own;
  * the primary is always resolved through the shared active/tenant check;
  * the primary is always part of the returned audience, and never duplicated in it.

`asyncio.run` rather than pytest-asyncio, matching test_query_authorization.py — the
marker is not available in every environment this suite runs in.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from src.api.routers import articles as articles_router
from src.domain.departments import resolve_active_department, resolve_active_departments


class _Department:
    def __init__(
        self, name: str, active: bool = True, company_domain: str = "qnsc.vn"
    ) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.company_domain = company_domain
        self.active = active


class _User:
    company_domain = "qnsc.vn"
    dept = "Nhan su"


HR = _Department("Nhan su")
OPERATIONS = _Department("Van hanh")
PUBLIC = _Department("public")
RETIRED = _Department("Phong cu", active=False)
OTHER_TENANT = _Department("Ke toan", company_domain="other.vn")
ALL = (HR, OPERATIONS, PUBLIC, RETIRED, OTHER_TENANT)


@pytest.fixture
def resolvers(monkeypatch):
    """Stand in for the two resolvers, preserving the active/tenant rule that matters."""

    def visible(company_domain: str) -> list[_Department]:
        return [
            department
            for department in ALL
            if department.active and department.company_domain == company_domain
        ]

    async def resolve_departments(db, company_domain, ids, *, required=True):
        by_id = {department.id: department for department in visible(company_domain)}
        unique_ids = list(dict.fromkeys(ids))
        resolved = [by_id[i] for i in unique_ids if i in by_id]
        if len(resolved) != len(unique_ids):
            raise HTTPException(
                status_code=422,
                detail="Every department must be active and belong to the user's company",
            )
        return resolved

    async def resolve_one(db, company_domain, name, *, required=True):
        for candidate in visible(company_domain):
            if candidate.name.lower() == str(name).lower():
                return candidate
        raise HTTPException(
            status_code=422, detail="Department does not exist or is inactive"
        )

    monkeypatch.setattr(
        articles_router, "resolve_active_departments", resolve_departments
    )
    monkeypatch.setattr(articles_router, "resolve_active_department", resolve_one)


def _resolve(dept, ids):
    return asyncio.run(
        articles_router._resolve_upload_departments(object(), _User(), dept, ids)
    )


def test_an_explicit_dept_wins_over_the_selection(resolvers):
    primary, _ = _resolve("Van hanh", [PUBLIC.id])
    assert primary.name == "Van hanh"


def test_the_first_selected_department_becomes_the_primary(resolvers):
    primary, audiences = _resolve(None, [OPERATIONS.id, PUBLIC.id])
    assert primary.name == "Van hanh"
    assert [d.id for d in audiences] == [OPERATIONS.id, PUBLIC.id]


def test_the_uploaders_own_department_is_the_last_resort(resolvers):
    primary, audiences = _resolve(None, None)
    assert primary.name == "Nhan su"
    assert audiences == [primary]


def test_the_primary_is_always_part_of_the_audience(resolvers):
    """The exact reported case: upload into "public" alone, but routed via HR."""
    primary, audiences = _resolve("Nhan su", [PUBLIC.id])

    assert primary.id == HR.id
    # The selection is preserved and the primary is prepended, never dropped.
    assert [d.id for d in audiences] == [HR.id, PUBLIC.id]


def test_a_primary_already_selected_is_not_duplicated(resolvers):
    primary, audiences = _resolve("public", [PUBLIC.id, OPERATIONS.id])
    assert primary.id == PUBLIC.id
    assert [d.id for d in audiences] == [PUBLIC.id, OPERATIONS.id]


def test_an_inactive_dept_is_rejected_at_upload(resolvers):
    """Better to refuse the upload than to create a draft that can never be approved."""
    with pytest.raises(HTTPException) as raised:
        _resolve("Phong cu", [PUBLIC.id])
    assert raised.value.status_code == 422


def test_an_inactive_department_cannot_be_selected_as_an_audience(resolvers):
    with pytest.raises(HTTPException) as raised:
        _resolve(None, [RETIRED.id])
    assert raised.value.status_code == 422


def test_a_cross_tenant_dept_is_rejected_at_upload(resolvers):
    with pytest.raises(HTTPException) as raised:
        _resolve("Ke toan", None)
    assert raised.value.status_code == 422


def test_a_cross_tenant_department_cannot_be_selected_as_an_audience(resolvers):
    with pytest.raises(HTTPException) as raised:
        _resolve(None, [OTHER_TENANT.id])
    assert raised.value.status_code == 422


def test_multipart_ids_are_still_parsed_from_json():
    ids = articles_router._parse_department_ids(f'["{PUBLIC.id}"]')
    assert ids == [PUBLIC.id]
    assert articles_router._parse_department_ids(None) is None
    with pytest.raises(HTTPException):
        articles_router._parse_department_ids("not json")


class _DB:
    """Answers the tenant-scoped, active-only queries the real resolvers issue."""

    def __init__(self, scalar_result=None, rows=()):
        self._scalar_result = scalar_result
        self._rows = list(rows)

    async def scalar(self, _statement):
        return self._scalar_result

    async def execute(self, _statement):
        rows = self._rows

        class _Result:
            @staticmethod
            def scalars():
                class _Scalars:
                    @staticmethod
                    def all():
                        return rows

                return _Scalars()

        return _Result()


def test_a_department_the_tenant_cannot_see_is_reported_as_missing():
    """Inactive, cross-tenant and non-existent are one message on purpose — the caller
    must not be able to probe another company's department list."""
    with pytest.raises(HTTPException) as raised:
        asyncio.run(resolve_active_department(_DB(scalar_result=None), "qnsc.vn", "public"))

    assert raised.value.status_code == 422
    assert raised.value.detail == "Department does not exist or is inactive"


def test_a_required_primary_name_cannot_be_blank():
    with pytest.raises(HTTPException) as raised:
        asyncio.run(resolve_active_department(_DB(), "qnsc.vn", "   "))

    assert raised.value.status_code == 422
    assert raised.value.detail == "A department is required"


def test_a_partially_visible_audience_selection_is_refused():
    """One unseen id fails the whole selection rather than silently narrowing it."""
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            resolve_active_departments(
                _DB(rows=[PUBLIC]), "qnsc.vn", [PUBLIC.id, OTHER_TENANT.id]
            )
        )

    assert raised.value.status_code == 422
    assert "belong to the user's company" in raised.value.detail
