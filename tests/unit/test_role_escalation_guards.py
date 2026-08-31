"""No route may hand a tenant identity the role that escapes tenant isolation.

The global Admin role — the Role row with company_domain = NULL — is the only identity
AuthorizationService.is_global_administrator recognizes, and holding it turns off row-level
security for every table. Exactly one producer is allowed to grant it: the startup
bootstrap in src/domain/admin_bootstrap.py, plus scripts/create_admin.py doing the same
thing deliberately.

Two paths used to reach it from inside a tenant, and both are covered here:

* Invitations. create_invitation checked only that the role was one of the four managed
  names, never that the inviter was allowed to grant it. A company-scoped `user.manage`
  holder could invite an address in their own domain as "Admin"; accept_invitation copies
  invitation.role onto the new account and calls bootstrap_rbac, which resolved the string
  "Admin" to the global role. The invitee came out a cross-tenant administrator.

* The backfill itself. bootstrap_rbac assigned the global Admin role to ANY role-less user
  whose scalar `role` column read "Admin". Even with the invitation hole closed that keeps
  the escalation one write away: anything that sets that string — a fixture, a support
  script, a future endpoint — promotes the account on the next backfill. Now the backfill
  resolves to the user's OWN company Admin role, so the global role is unreachable without
  asking for it by name.

The tests call the real functions. Asserting on source text would pass against code that
imports the guard and never runs it, which is the failure mode this file exists to catch.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User


class _Result:
    """Enough of a SQLAlchemy result for the scalar and collection reads under test."""

    def __init__(self, rows=None, rowcount=0):
        self.rows = list(rows or [])
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self.rows

    def scalar_one(self):
        return self.rows[0] if self.rows else 0

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _InviteDB:
    """Routes reads by target table; records what create_invitation would persist."""

    def __init__(self, *, existing_user: User | None = None, department_count: int = 0):
        self.existing_user = existing_user
        self.department_count = department_count
        self.added: list[object] = []
        self.commits = 0

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(dialect=postgresql.dialect()))
        except Exception:
            rendered = str(statement)
        lowered = " ".join(rendered.lower().split())
        if "count" in lowered and "from departments" in lowered:
            return _Result([self.department_count])
        if "from users" in lowered:
            return _Result([self.existing_user] if self.existing_user else [])
        return _Result()

    def add(self, item) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        for item in self.added:
            if getattr(item, "id", None) is None:
                item.id = uuid.uuid4()

    async def commit(self) -> None:
        self.commits += 1
        await self.flush()

    async def refresh(self, _item) -> None:
        return None


def _role(name: str, company_domain: str | None, scope: str) -> Role:
    """A role carrying `user.manage` at the given scope, as the guards evaluate it."""
    role = Role(name=name, company_domain=company_domain, active=True)
    role.permissions = [
        RolePermission(
            permission=Permission(key="user.manage", name="Manage users"), scope=scope
        )
    ]
    return role


def _company_manager(company: str = "qnsc.vn") -> User:
    """A tenant identity holding company-scoped user.manage: the attacker in the chain."""
    user = User(id=uuid.uuid4(), role="CEO", company_domain=company)
    user.roles = [_role("CEO", company, "company")]
    return user


def _global_manager(company: str = "qnsc.vn") -> User:
    user = User(id=uuid.uuid4(), role="Admin", company_domain=company)
    user.roles = [_role("Admin", None, "global")]
    return user


def _invitation(**overrides):
    from src.api.routers.auth import InvitationCreate

    payload = {"email": "new.hire@qnsc.vn", "name": "New Hire", "role": "Staff"}
    payload.update(overrides)
    return InvitationCreate(**payload)


def _create_invitation(db, payload, user):
    from src.api.routers.auth import create_invitation

    return asyncio.run(create_invitation(payload=payload, current_user=user, db=db))


# --------------------------------------------------------------- invitation role grants


@pytest.mark.parametrize("role", ["Admin", "CEO"])
def test_a_company_manager_cannot_invite_a_privileged_role(role):
    """The head of the chain. Blocking it here is what keeps the rest unreachable."""
    db = _InviteDB()

    with pytest.raises(HTTPException) as error:
        _create_invitation(db, _invitation(role=role), _company_manager())

    assert error.value.status_code == 403
    assert not db.added, "a refused invitation must not be persisted"


def test_the_refusal_reuses_the_wording_every_other_role_grant_uses():
    """Same detail as _validate_role_assignment_authority, so clients need no new case."""
    with pytest.raises(HTTPException) as error:
        _create_invitation(_InviteDB(), _invitation(role="Admin"), _company_manager())

    assert (
        error.value.detail
        == "Only global user managers can assign global or executive roles"
    )


def test_a_company_manager_can_still_invite_an_ordinary_employee():
    """The guard must be about authority, not a blanket refusal of invitations."""
    from src.models.user import Invitation

    db = _InviteDB()

    result = _create_invitation(db, _invitation(role="Staff"), _company_manager())

    assert result["role"] == "Staff"
    assert result["status"] == "pending"
    assert len([item for item in db.added if isinstance(item, Invitation)]) == 1


def test_a_global_manager_may_still_invite_an_administrator():
    """Someone has to be able to; the check is who, not whether."""
    db = _InviteDB()

    result = _create_invitation(db, _invitation(role="Admin"), _global_manager())

    assert result["role"] == "Admin"


def test_an_unmanaged_role_name_is_still_refused_as_unprocessable():
    """The new authority check must not shadow the existing role-name validation."""
    with pytest.raises(HTTPException) as error:
        _create_invitation(
            _InviteDB(), _invitation(role="Superuser"), _global_manager()
        )

    assert error.value.status_code == 422


# ------------------------------------------------------------------ the RBAC backfill


class _BootstrapDB:
    """Serves bootstrap_rbac's three read shapes from in-memory rows.

    bootstrap_rbac reads the permission catalog, the user list, then one role per
    (name, company_domain) it wants, creating what it does not find. Rows added here are
    visible to the next lookup, which is what lets the test observe WHICH role a user was
    given rather than only that some role was.
    """

    def __init__(self, users: list[User], roles: list[Role] | None = None):
        self.users = users
        self.roles = list(roles or [])
        self.permissions: dict[str, Permission] = {}
        self.added: list[object] = []
        self.commits = 0

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            ))
        except Exception:
            rendered = str(statement)
        # Identifiers compile lowercase, so routing and literal extraction can share one
        # case-PRESERVED rendering. Lowercasing it would turn 'Admin' into 'admin' and
        # every role lookup would miss.
        sql = " ".join(rendered.split())
        lowered = sql.lower()

        if "from permissions" in lowered:
            key = _quoted_value(sql, "permissions.key =")
            return _Result([self.permissions[key]] if key in self.permissions else [])
        if "from users" in lowered:
            return _Result(self.users)
        if "from role_permissions" in lowered:
            return _Result()
        if "from roles" in lowered:
            name = _quoted_value(sql, "roles.name =")
            domain = (
                None
                if "roles.company_domain is null" in lowered
                else _quoted_value(sql, "roles.company_domain =")
            )
            return _Result(
                [
                    role
                    for role in self.roles
                    if role.name == name and role.company_domain == domain
                ]
            )
        return _Result()

    def add(self, item) -> None:
        self.added.append(item)
        if isinstance(item, Permission):
            self.permissions[item.key] = item
        if isinstance(item, Role):
            self.roles.append(item)

    async def flush(self) -> None:
        for item in self.added:
            if getattr(item, "id", None) is None:
                item.id = uuid.uuid4()

    async def commit(self) -> None:
        self.commits += 1
        await self.flush()


def _quoted_value(sql: str, prefix: str) -> str | None:
    """The literal on the right of `prefix` in a compiled WHERE clause."""
    if prefix not in sql:
        return None
    tail = sql.split(prefix, 1)[1].strip()
    if not tail.startswith("'"):
        return None
    return tail[1:].split("'", 1)[0]


def _run_bootstrap(db):
    from src.domain.rbac import bootstrap_rbac

    return asyncio.run(bootstrap_rbac(db))


def _assigned(user: User) -> Role:
    assert len(user.roles) == 1, f"expected exactly one backfilled role, got {user.roles}"
    return user.roles[0]


def test_an_admin_inside_a_company_is_backfilled_with_that_company_admin_role():
    """The tail of the escalation chain: this used to resolve to the global role."""
    user = User(id=uuid.uuid4(), role="Admin", company_domain="acme.test")
    user.roles = []
    db = _BootstrapDB([user])

    _run_bootstrap(db)

    role = _assigned(user)
    assert role.name == "Admin"
    assert role.company_domain == "acme.test", (
        "a tenant Admin holding the company_domain=NULL role bypasses row-level security "
        "for every other company"
    )


def test_no_backfilled_user_with_a_company_receives_a_global_role():
    """The invariant, across every managed role rather than just Admin."""
    users = []
    for name in ("Admin", "CEO", "Reviewer", "Staff"):
        user = User(id=uuid.uuid4(), role=name, company_domain="acme.test")
        user.roles = []
        users.append(user)
    db = _BootstrapDB(users)

    _run_bootstrap(db)

    for user in users:
        assert _assigned(user).company_domain == user.company_domain


def test_the_seeded_administrator_without_a_company_still_gets_the_global_role():
    """The one legitimate holder. Scoping it to NULL is what admin_bootstrap looks up."""
    user = User(id=uuid.uuid4(), role="Admin", company_domain=None)
    user.roles = []
    db = _BootstrapDB([user])

    _run_bootstrap(db)

    role = _assigned(user)
    assert role.name == "Admin"
    assert role.company_domain is None


def test_the_global_admin_role_is_created_even_when_no_user_needs_it():
    """admin_bootstrap refuses to seed an administrator when the role is absent."""
    db = _BootstrapDB([])

    _run_bootstrap(db)

    assert any(
        role.name == "Admin" and role.company_domain is None for role in db.roles
    )


def test_every_company_gets_its_own_admin_role_so_the_backfill_has_one_to_use():
    """Previously skipped for Admin, which is why the backfill had to reach for global."""
    user = User(id=uuid.uuid4(), role="Staff", company_domain="acme.test")
    user.roles = []
    db = _BootstrapDB([user])

    _run_bootstrap(db)

    assert any(
        role.name == "Admin" and role.company_domain == "acme.test" for role in db.roles
    )


def test_a_user_that_already_has_roles_is_left_alone():
    """The backfill is for pre-RBAC rows; re-running it must not reset assignments."""
    existing = _role("Reviewer", "acme.test", "company")
    user = User(id=uuid.uuid4(), role="Admin", company_domain="acme.test")
    user.roles = [existing]
    db = _BootstrapDB([user], roles=[existing])

    _run_bootstrap(db)

    assert user.roles == [existing]


# ------------------------------------------------------- promotion through _set_primary_role


class _RoleLookupDB:
    """Records which (name, company_domain) pair _set_primary_role asked for."""

    def __init__(self):
        self.requested: list[tuple[str | None, str | None]] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        sql = " ".join(
            str(statement.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )).split()
        )
        name = _quoted_value(sql, "roles.name =")
        domain = (
            None
            if "roles.company_domain IS NULL" in sql
            else _quoted_value(sql, "roles.company_domain =")
        )
        self.requested.append((name, domain))
        return _Result([Role(name=name, company_domain=domain, active=True)])


def _set_primary_role(db, user, name):
    from src.api.routers.auth import _set_primary_role as target

    return asyncio.run(target(db, user, name))


def test_promoting_a_tenant_user_to_admin_resolves_their_company_admin_role():
    """update_managed_user's role path goes through here; global would cross tenants."""
    user = User(id=uuid.uuid4(), role="Staff", company_domain="acme.test")
    user.roles = []
    db = _RoleLookupDB()

    _set_primary_role(db, user, "Admin")

    assert db.requested == [("Admin", "acme.test")]
    assert user.roles[0].company_domain == "acme.test"
    assert user.role == "Admin"


def test_promoting_an_account_with_no_company_still_resolves_the_global_role():
    user = User(id=uuid.uuid4(), role="Staff", company_domain=None)
    user.roles = []
    db = _RoleLookupDB()

    _set_primary_role(db, user, "Admin")

    assert db.requested == [("Admin", None)]
    assert user.roles[0].company_domain is None


# ------------------------------------------------------------ authentication side channels


def test_an_unknown_address_still_pays_for_a_password_comparison():
    """Answering before bcrypt runs tells an attacker which addresses have accounts.

    Asserting the comparison happens rather than timing it: a wall-clock assertion is
    exactly the kind of test that fails on loaded CI for reasons unrelated to the code.
    """
    from src.domain import auth as auth_domain

    class _EmptyRepo:
        async def get_by_email(self, _email):
            return None

    comparisons: list[str] = []
    original = auth_domain.verify_password

    def _record(password, hashed):
        comparisons.append(hashed)
        return original(password, hashed)

    auth_domain.verify_password = _record
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                auth_domain.AuthService(_EmptyRepo()).authenticate_user(
                    "nobody@acme.test", "whatever"
                )
            )
    finally:
        auth_domain.verify_password = original

    assert error.value.status_code == 401
    assert comparisons == [auth_domain._ABSENT_ACCOUNT_PASSWORD_HASH]


def test_the_dummy_hash_is_a_real_bcrypt_digest_that_nothing_can_match():
    """A short-circuiting placeholder would cost nothing and equalize nothing."""
    from src.core.security import verify_password
    from src.domain.auth import _ABSENT_ACCOUNT_PASSWORD_HASH

    assert _ABSENT_ACCOUNT_PASSWORD_HASH.startswith("$2b$")
    assert not verify_password("", _ABSENT_ACCOUNT_PASSWORD_HASH)


# ------------------------------------------------------------------- rate limiter failure


def _allow_with_redis_down(limiter, monkeypatch):
    """Drive the limiter's except branch by making the Redis constructor raise."""
    from src.core import rate_limit

    def _explode(*_args, **_kwargs):
        raise OSError("redis unreachable")

    monkeypatch.setattr(rate_limit.Redis, "from_url", _explode)
    return asyncio.run(limiter.allow("login:203.0.113.9"))


def test_auth_limiting_fails_closed_when_redis_is_unreachable(monkeypatch):
    """The in-memory fallback multiplies the limit by the replica count and, during an
    outage, removes it entirely — an open credential-stuffing window on /login."""
    from src.core import rate_limit

    monkeypatch.setattr(rate_limit.settings, "ENVIRONMENT", "production")

    allowed, retry_after = _allow_with_redis_down(rate_limit.auth_rate_limiter, monkeypatch)

    assert not allowed
    assert retry_after >= 1


def test_a_developer_without_redis_can_still_sign_in(monkeypatch):
    """Failing closed in development would make a local stack unusable."""
    from src.core import rate_limit

    monkeypatch.setattr(rate_limit.settings, "ENVIRONMENT", "development")

    allowed, _ = _allow_with_redis_down(rate_limit.auth_rate_limiter, monkeypatch)

    assert allowed


def test_non_auth_namespaces_keep_degrading_rather_than_refusing(monkeypatch):
    """An AI or upload limiter has no credential to protect; a 429 storm is worse."""
    from src.core import rate_limit

    monkeypatch.setattr(rate_limit.settings, "ENVIRONMENT", "production")

    allowed, _ = _allow_with_redis_down(rate_limit.ai_rate_limiter, monkeypatch)

    assert allowed
