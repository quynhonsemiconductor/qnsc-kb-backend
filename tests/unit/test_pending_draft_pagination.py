"""The review queue must page, and a page must never hide a row.

The old endpoint applied a hard `.limit(500)` with nothing reporting it: past 500 pending
drafts the OLDEST were present in no tab of the UI, and the count shown stayed at 500
however many were added -- oldest because the order is newest-first, which is exactly the
set most likely to be stuck and in need of attention.

Two properties carry the weight here:

* paging is exact -- the union of consecutive pages is the whole set, with nothing
  skipped and nothing served twice, even when `created_at` ties;
* the total respects authorization -- counting rows the caller may not see would leak
  the existence of another department's drafts.
"""
from __future__ import annotations

import asyncio
import unicodedata
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from src.domain.governance import GovernanceService
from src.models.governance import PendingDraft
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import Department, User
from src.repositories.governance import GovernanceRepository

LIMIT = GovernanceRepository.DRAFT_PAGE_LIMIT


def _draft(index: int, created_at: datetime, **overrides) -> PendingDraft:
    values = {
        "id": uuid.uuid4(),
        "title": f"Draft {index}",
        "company_domain": "acme.test",
        "dept": "Engineering",
        "source_ref": f"upload://draft-{index}.pdf",
        "source_hash": f"{index:064d}",
        "status": "pending",
        "created_at": created_at,
    }
    values.update(overrides)
    return PendingDraft(**values)


class _PagingRepository:
    """An in-memory stand-in that orders and slices exactly as the SQL does.

    Ordering is the contract under test, so this mirrors
    `created_at DESC, id DESC` rather than returning insertion order -- a fake that
    preserved insertion order would pass while the real query skipped rows.
    """

    DRAFT_PAGE_LIMIT = LIMIT

    def __init__(self, drafts, db=None):
        self.drafts = list(drafts)
        self.db = db
        self.calls = []

    @staticmethod
    def _fold(value) -> str:
        """Lower-case and strip accents, as `immutable_unaccent(...) ILIKE` does in SQL."""
        decomposed = unicodedata.normalize("NFKD", str(value or ""))
        return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()

    def _matching(
        self, status, company_domain, dept, depts, assigned_approver_id, search=None
    ):
        rows = list(self.drafts)
        if search and search.strip():
            needle = self._fold(search.strip())
            rows = [
                row
                for row in rows
                if any(
                    needle in self._fold(value)
                    for value in (row.title, row.source_ref, row.dept)
                )
            ]
        if status:
            rows = [row for row in rows if row.status == status]
        if company_domain:
            rows = [row for row in rows if row.company_domain == company_domain]
        if depts:
            rows = [row for row in rows if row.dept in set(depts)]
        elif dept:
            rows = [row for row in rows if row.dept == dept]
        if assigned_approver_id:
            rows = [
                row
                for row in rows
                if row.assigned_approver_id in {None, assigned_approver_id}
                or (depts and row.dept in set(depts))
            ]
        return sorted(rows, key=lambda row: (row.created_at, row.id.bytes), reverse=True)

    async def list_drafts(
        self, status=None, company_domain=None, dept=None, depts=None,
        assigned_approver_id=None, *, limit=None, offset=0, search=None,
    ):
        self.calls.append({"limit": limit, "offset": offset})
        rows = self._matching(
            status, company_domain, dept, depts, assigned_approver_id, search
        )
        window = rows[max(0, offset):]
        return window if limit is None else window[:limit]

    async def count_drafts(
        self, status=None, company_domain=None, dept=None, depts=None,
        assigned_approver_id=None, *, search=None,
    ):
        return len(
            self._matching(
                status, company_domain, dept, depts, assigned_approver_id, search
            )
        )


def _service(drafts):
    return GovernanceService(_PagingRepository(drafts), object())


def _publisher() -> User:
    user = User(
        id=uuid.uuid4(), email="publisher@acme.test", name="Publisher",
        company_domain="acme.test", role="Reviewer", active=True,
    )
    permission = Permission(key="article.publish", name="Publish")
    role = Role(name="Publisher", company_domain=None)
    role.permissions.append(RolePermission(permission=permission, scope="global"))
    user.roles.append(role)
    return user


def _page(service, user, *, limit, offset):
    return asyncio.run(service.list_drafts(user, "pending", limit=limit, offset=offset))


# --- paging arithmetic -----------------------------------------------------


@pytest.fixture
def seven_drafts():
    base = datetime(2026, 10, 13, 12, 0, 0)
    return [_draft(index, base - timedelta(minutes=index)) for index in range(7)]


def test_the_first_page_returns_the_newest_rows(seven_drafts):
    service = _service(seven_drafts)
    page = _page(service, _publisher(), limit=3, offset=0)
    assert [item.title for item in page] == ["Draft 0", "Draft 1", "Draft 2"]


def test_a_middle_page_continues_where_the_previous_one_stopped(seven_drafts):
    service = _service(seven_drafts)
    page = _page(service, _publisher(), limit=3, offset=3)
    assert [item.title for item in page] == ["Draft 3", "Draft 4", "Draft 5"]


def test_the_last_page_may_be_partial(seven_drafts):
    service = _service(seven_drafts)
    page = _page(service, _publisher(), limit=3, offset=6)
    assert [item.title for item in page] == ["Draft 6"]


def test_an_offset_past_the_end_is_empty_rather_than_an_error(seven_drafts):
    service = _service(seven_drafts)
    assert _page(service, _publisher(), limit=3, offset=99) == []


def test_omitting_the_paging_arguments_preserves_the_old_behaviour(seven_drafts):
    """A client built against the un-paginated API must keep working unchanged."""
    service = _service(seven_drafts)
    drafts = asyncio.run(service.list_drafts(_publisher(), "pending"))
    assert len(drafts) == 7
    assert service.gov_repo.calls[-1] == {"limit": None, "offset": 0}


# --- the invariant the old cap broke ---------------------------------------


def test_consecutive_pages_cover_every_row_exactly_once_despite_identical_timestamps():
    """`created_at` is NOT unique, and that is what makes OFFSET paging dangerous.

    A bulk upload commits many rows inside one transaction, so their timestamps collide.
    Under `ORDER BY created_at DESC` alone the database may return tied rows in either
    order, and two orderings across two queries straddle the page boundary: one row is
    served twice and another is never served at all. The `id` tie-breaker makes the
    order total, which is what makes the union below exact.
    """
    collided = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [_draft(index, collided) for index in range(10)]
    service = _service(drafts)
    user = _publisher()

    seen = []
    for offset in range(0, 10, 3):
        seen.extend(item.id for item in _page(service, user, limit=3, offset=offset))

    assert len(seen) == 10, "every row is served"
    assert len(set(seen)) == 10, "no row is served twice"
    assert set(seen) == {item.id for item in drafts}


def test_a_total_larger_than_one_page_is_still_reported():
    """The failure this whole change exists to fix: a truncated list reporting nothing.

    534 rows behind a 500 cap looked identical to 500 rows in total.
    """
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [_draft(index, base - timedelta(seconds=index)) for index in range(LIMIT + 34)]
    service = _service(drafts)
    user = _publisher()

    first_page = _page(service, user, limit=LIMIT, offset=0)
    total = asyncio.run(service.count_drafts(user, "pending"))

    assert len(first_page) == LIMIT
    assert total == LIMIT + 34, "the rows beyond the page are counted, not hidden"
    remainder = _page(service, user, limit=LIMIT, offset=LIMIT)
    assert len(remainder) == 34
    assert not {item.id for item in first_page} & {item.id for item in remainder}


# --- the total follows the caller's authorization ---------------------------


def _department_reviewer() -> User:
    user = User(
        id=uuid.uuid4(), email="reviewer@acme.test", name="Reviewer",
        company_domain="acme.test", role="Reviewer", active=True,
    )
    permission = Permission(key="article.review", name="Review")
    role = Role(name="Reviewer", company_domain="acme.test")
    role.permissions.append(RolePermission(permission=permission, scope="company"))
    user.roles.append(role)
    user.dept = "Engineering"
    user.departments = [
        Department(id=uuid.uuid4(), name="Engineering", company_domain="acme.test", active=True)
    ]
    return user


def test_the_total_counts_only_what_the_caller_may_see():
    """A count computed from wider predicates than the list leaks other departments.

    The reviewer below may see Engineering; Finance and another tenant must not appear
    in the page OR in the total.
    """
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [
        _draft(0, base, dept="Engineering"),
        _draft(1, base - timedelta(minutes=1), dept="Engineering"),
        _draft(2, base - timedelta(minutes=2), dept="Finance"),
        _draft(3, base - timedelta(minutes=3), dept="Engineering", company_domain="other.test"),
    ]
    service = _service(drafts)
    reviewer = _department_reviewer()

    page = asyncio.run(service.list_drafts(reviewer, "pending"))
    total = asyncio.run(service.count_drafts(reviewer, "pending"))

    assert total == len(page) == 2
    assert {item.dept for item in page} == {"Engineering"}
    assert all(item.company_domain == "acme.test" for item in page)


def test_a_company_lead_is_counted_within_its_own_tenant():
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [
        _draft(0, base),
        _draft(1, base - timedelta(minutes=1), company_domain="other.test"),
    ]
    service = _service(drafts)
    # CEO, not Admin: the legacy Admin role is a GLOBAL publisher and so sees every tenant
    # (see the next test); the company-scoped governance lead is the CEO.
    lead = User(
        id=uuid.uuid4(), email="ceo@acme.test", name="CEO",
        company_domain="acme.test", role="CEO", active=True,
    )

    assert asyncio.run(service.count_drafts(lead, "pending")) == 1


def test_a_global_admin_is_counted_across_tenants():
    """The other side of the test above: Admin is a global publisher, so its scope has no
    tenant filter and the total spans every tenant, exactly as the list always has."""
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [
        _draft(0, base),
        _draft(1, base - timedelta(minutes=1), company_domain="other.test"),
    ]
    service = _service(drafts)
    admin = User(
        id=uuid.uuid4(), email="admin@acme.test", name="Admin",
        company_domain="acme.test", role="Admin", active=True,
    )

    assert asyncio.run(service.count_drafts(admin, "pending")) == 2


def test_an_unauthorized_caller_cannot_obtain_a_total():
    """The count must fail closed exactly where the list does, not leak a number."""
    service = _service([_draft(0, datetime(2026, 10, 13, 12, 0, 0))])
    outsider = User(
        id=uuid.uuid4(), email="staff@acme.test", name="Staff",
        company_domain="acme.test", role="Staff", active=True,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(service.count_drafts(outsider, "pending"))
    assert error.value.status_code == 403


def test_every_authorization_branch_pages_and_counts_through_the_same_scope():
    """The page and the total must be built from ONE scope resolution, not two.

    Three branches exist -- global publisher, company lead, department reviewer -- and a
    count that disagreed with the list in any of them would either hide work or reveal
    another department's.
    """
    service = _service([])
    for user in (_publisher(), _department_reviewer()):
        scope = service._draft_queue_scope(user)
        asyncio.run(service.list_drafts(user, "pending", limit=10, offset=0))
        asyncio.run(service.count_drafts(user, "pending"))
        assert service._draft_queue_scope(user) == scope, "scope is deterministic"


# --- search ------------------------------------------------------------------
#
# The browser used to filter the rows it had loaded, so with more than one page a search
# for anything outside them found nothing and said nothing. Search now goes to the server
# and applies to the whole queue, narrowing inside the caller's scope and never widening it.


def _searchable():
    base = datetime(2026, 10, 13, 12, 0, 0)
    rows = [
        ("Quy trình xin nghỉ phép", "upload://nghi-phep.docx", "Nhân sự"),
        ("Hướng dẫn VPN", "upload://vpn.pdf", "IT"),
        ("Báo cáo doanh thu 50% quý 3", "upload://q3.xlsx", "Tài chính"),
        ("Server access", "sharepoint://drive-1/it-runbook.pdf", "IT"),
    ]
    return [
        _draft(index, base - timedelta(minutes=index), title=title, source_ref=ref, dept=dept)
        for index, (title, ref, dept) in enumerate(rows)
    ]


def _titles(drafts):
    return [item.title for item in drafts]


def test_search_ignores_case_and_accents():
    service = _service(_searchable())

    found = asyncio.run(service.list_drafts(_publisher(), "pending", search="QUY TRINH xin nghi"))

    assert _titles(found) == ["Quy trình xin nghỉ phép"]


def test_search_matches_the_source_reference_and_the_department():
    service = _service(_searchable())
    publisher = _publisher()

    by_reference = asyncio.run(service.list_drafts(publisher, "pending", search="it-runbook"))
    by_department = asyncio.run(service.list_drafts(publisher, "pending", search="tai chinh"))

    assert _titles(by_reference) == ["Server access"]
    assert _titles(by_department) == ["Báo cáo doanh thu 50% quý 3"]


def test_the_total_describes_the_rows_the_search_matches():
    """Not the whole queue: a header saying 4 over a page of 1 would invite a pager that
    walks three empty pages."""
    service = _service(_searchable())
    publisher = _publisher()

    assert asyncio.run(service.count_drafts(publisher, "pending")) == 4
    assert asyncio.run(service.count_drafts(publisher, "pending", search="vpn")) == 1
    assert asyncio.run(service.count_drafts(publisher, "pending", search="nothing matches")) == 0


def test_a_blank_search_is_no_search():
    service = _service(_searchable())
    publisher = _publisher()

    for blank in (None, "", "   "):
        assert len(asyncio.run(service.list_drafts(publisher, "pending", search=blank))) == 4
        assert asyncio.run(service.count_drafts(publisher, "pending", search=blank)) == 4


def test_pages_of_search_results_cover_every_match_exactly_once():
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [
        _draft(index, base, title=f"Quarterly report {index}") for index in range(7)
    ] + [_draft(99, base, title="Something else entirely")]
    service = _service(drafts)
    publisher = _publisher()

    seen = []
    for offset in (0, 3, 6):
        page = asyncio.run(
            service.list_drafts(publisher, "pending", limit=3, offset=offset, search="report")
        )
        seen.extend(item.id for item in page)

    assert len(seen) == len(set(seen)) == 7
    assert asyncio.run(service.count_drafts(publisher, "pending", search="report")) == 7


def test_a_search_never_widens_what_the_caller_may_see():
    """A department reviewer searching for another department's draft finds nothing, and the
    total does not confirm that it exists."""
    base = datetime(2026, 10, 13, 12, 0, 0)
    drafts = [
        _draft(0, base, title="Payroll calendar", dept="Engineering"),
        _draft(1, base - timedelta(minutes=1), title="Payroll audit", dept="Finance"),
    ]
    service = _service(drafts)
    reviewer = _department_reviewer()

    found = asyncio.run(service.list_drafts(reviewer, "pending", search="payroll"))
    total = asyncio.run(service.count_drafts(reviewer, "pending", search="payroll"))

    assert _titles(found) == ["Payroll calendar"]
    assert total == 1


# The doubles above model the matching; these pin the SQL the real repository builds.


def test_the_sql_search_covers_three_columns_and_is_accent_insensitive():
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    conditions = GovernanceRepository._draft_scope_filters(search="nghi phep")
    sql = str(
        select(PendingDraft.id)
        .where(*conditions)
        .compile(dialect=postgresql.dialect())
    )

    for column in ("title", "source_ref", "dept"):
        assert f"immutable_unaccent(pending_drafts.{column})" in sql
    # One predicate per column, each an escaped, case-insensitive match.
    assert sql.count("ILIKE") == 3
    assert sql.count("ESCAPE") == 3


def test_the_sql_search_treats_like_wildcards_in_user_text_as_literal():
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    from src.repositories.governance import _escape_like

    assert _escape_like("50%") == "50!%"
    assert _escape_like("a_b") == "a!_b"
    # The escape character itself is escaped, or "wow!" would swallow the next character.
    assert _escape_like("wow!") == "wow!!"
    assert _escape_like("100%_wow!") == "100!%!_wow!!"

    compiled = (
        select(PendingDraft.id)
        .where(*GovernanceRepository._draft_scope_filters(search="50%_off"))
        .compile(dialect=postgresql.dialect())
    )
    assert "%50!%!_off%" in compiled.params.values()
    # No backslash anywhere in the statement, so it cannot depend on the server's
    # `standard_conforming_strings`.
    assert "\\" not in str(compiled)
    assert "ESCAPE '!'" in str(compiled)


def test_a_blank_search_adds_no_sql_condition():
    assert GovernanceRepository._draft_scope_filters(search="   ") == []
    assert GovernanceRepository._draft_scope_filters(search=None) == []


class _CapturingDb:
    """Records the statements the repository sends, so the SQL itself can be asserted."""

    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)

        class _Result:
            def scalars(self):
                return self

            def all(self):
                return []

        return _Result()

    async def scalar(self, statement):
        self.statements.append(statement)
        return 0


def _pg(statement) -> str:
    from sqlalchemy.dialects import postgresql

    return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())


def test_the_real_page_query_has_a_total_order_and_the_requested_window():
    """The fake repository above mirrors the order; this pins what the SQL actually says.
    Without `id` in the ORDER BY, rows that share a created_at can straddle a page boundary."""
    db = _CapturingDb()
    repository = GovernanceRepository(db)

    asyncio.run(repository.list_drafts("pending", "acme.test", limit=3, offset=6, search="x"))

    sql = _pg(db.statements[0])
    assert "ORDER BY pending_drafts.created_at DESC, pending_drafts.id DESC" in sql
    assert "LIMIT %(param_1)s OFFSET %(param_2)s" in sql


def test_the_real_page_and_the_real_total_are_built_from_the_same_predicates():
    """If the count's WHERE differed from the page's, the total would describe another set
    of rows -- and for a scoped reviewer, count rows they may not see."""
    db = _CapturingDb()
    repository = GovernanceRepository(db)
    scope = dict(
        status="pending", company_domain="acme.test", depts=["Engineering"],
        assigned_approver_id=uuid.uuid4(),
    )

    asyncio.run(repository.list_drafts(**scope, limit=10, search="quy trinh"))
    asyncio.run(repository.count_drafts(**scope, search="quy trinh"))

    page_sql, count_sql = (_pg(statement) for statement in db.statements)
    assert page_sql.split(" WHERE ")[1].split(" ORDER BY ")[0] == count_sql.split(" WHERE ")[1]
    assert "immutable_unaccent" in count_sql, "the search applies to the total too"


# --- the endpoint -----------------------------------------------------------


def test_the_total_header_is_exposed_to_browser_javascript():
    """Same defect as X-Request-ID (see test_audit_defects): the frontend is another origin,
    so without `expose_headers` the browser hides the header and the pager cannot know how
    many pages there are."""
    from starlette.middleware.cors import CORSMiddleware

    from src.api.main import app

    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            assert "X-Total-Count" in (middleware.kwargs.get("expose_headers") or [])
            return
    raise AssertionError("CORS middleware is not configured")


def test_the_endpoint_sets_the_total_header_and_passes_search_to_the_page_and_the_count(
    monkeypatch,
):
    """`search` must reach BOTH calls. Passed to only one, the page would be filtered while
    the header still reported the whole queue, or the reverse."""
    from fastapi import Response

    from src.api.routers import governance as router

    seen = {}

    class _Service:
        def __init__(self, *args):
            pass

        async def list_drafts(self, user, status=None, *, limit=None, offset=0, search=None):
            seen["list"] = (status, limit, offset, search)
            return []

        async def count_drafts(self, user, status=None, *, search=None):
            seen["count"] = (status, search)
            return 534

    monkeypatch.setattr(router, "GovernanceService", _Service)
    monkeypatch.setattr(router, "GovernanceRepository", lambda db: None)
    monkeypatch.setattr(router, "ArticleRepository", lambda db: None)
    response = Response()

    body = asyncio.run(
        router.list_pending_drafts(
            response=response, status="pending", limit=500, offset=500,
            search="quy trinh", current_user=_publisher(), db=None,
        )
    )

    assert body == []
    assert response.headers["X-Total-Count"] == "534"
    assert seen["list"] == ("pending", 500, 500, "quy trinh")
    assert seen["count"] == ("pending", "quy trinh")
