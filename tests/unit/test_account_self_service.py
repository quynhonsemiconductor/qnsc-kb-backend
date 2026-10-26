"""The signed-in user's own account: name, signed-in devices, recent activity.

Handlers are awaited directly through `asyncio.run` against a table-routed session double,
like `test_password_auth_flows.py`. A fake session cannot prove RLS or the real queries'
filters, so those are checked by reading the compiled SQL, not by claiming a result.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from sqlalchemy.dialects import postgresql

from src.api.routers import account, auth
from src.domain.device import describe_device
from src.models.governance import AuditLog
from src.models.sessions import RefreshSession
from src.models.user import User

CHROME_WINDOWS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
EDGE_WINDOWS = CHROME_WINDOWS + " Edg/130.0.0.0"
SAFARI_IPHONE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
)
CHROME_ANDROID = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36"
)


class _Result:
    def __init__(self, rows=None):
        self.rows = [row for row in (rows or []) if row is not None]

    def scalars(self):
        return self

    def all(self):
        return list(self.rows)

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _DB:
    def __init__(
        self, user: User, sessions: list[RefreshSession] | None = None, audit=None
    ):
        self.user = user
        self.sessions = list(sessions or [])
        self.audit_rows = list(audit or [])
        self.sql: list[str] = []
        self.added: list[object] = []
        self.commits = 0

    async def execute(self, statement, *args, **kwargs):
        compiled = statement.compile(dialect=postgresql.dialect())
        lowered = " ".join(str(compiled).lower().split())
        self.sql.append(lowered)
        if lowered.startswith("update refresh_sessions"):
            for item in self.sessions:
                if item.user_id == self.user.id and item.revoked_at is None:
                    item.revoked_at = datetime.utcnow()
            return _Result()
        if "from refresh_sessions" in lowered:
            wanted = [v for v in compiled.params.values() if isinstance(v, uuid.UUID)]
            if "refresh_sessions.id =" in lowered:
                match = [
                    s
                    for s in self.sessions
                    if s.id in wanted and s.user_id in wanted and s.revoked_at is None
                ]
                return _Result(match)
            return _Result([s for s in self.sessions if s.revoked_at is None])
        if "from audit_logs" in lowered:
            return _Result(self.audit_rows)
        if "from users" in lowered:
            return _Result([self.user])
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

    def audit_actions(self) -> list[str]:
        return [item.action for item in self.added if isinstance(item, AuditLog)]


def _user() -> User:
    user = User(
        id=uuid.uuid4(),
        email="member@qnsc.vn",
        name="Existing Member",
        password_hash="x",
        company_domain="qnsc.vn",
        role="Staff",
        active=True,
        auth_version=4,
    )
    user.roles = []
    user.departments = []
    user.department_ownerships = []
    return user


def _session(
    user: User, token: str, *, agent: str = CHROME_WINDOWS, ip: str = "203.0.113.9"
) -> RefreshSession:
    now = datetime.utcnow()
    return RefreshSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=auth._refresh_hash(token),
        expires_at=now + timedelta(days=20),
        family_id=uuid.uuid4(),
        user_agent=agent,
        ip_address=ip,
        signed_in_at=now - timedelta(days=2),
        last_used_at=now - timedelta(minutes=5),
        created_at=now - timedelta(minutes=5),
    )


def _request(agent: str = CHROME_WINDOWS) -> SimpleNamespace:
    return SimpleNamespace(
        headers={"user-agent": agent}, client=SimpleNamespace(host="198.51.100.7")
    )


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _development(monkeypatch):
    # The plain cookie name is read only in development, which is what these tests set up.
    monkeypatch.setattr(account.settings, "ENVIRONMENT", "development", raising=False)
    monkeypatch.setattr(auth.settings, "ENVIRONMENT", "development", raising=False)


# -- the device name ----------------------------------------------------------------------


def test_a_browser_and_system_are_named_in_the_right_order():
    assert describe_device(CHROME_WINDOWS) == {"browser": "Chrome", "os": "Windows"}
    assert describe_device(EDGE_WINDOWS) == {"browser": "Edge", "os": "Windows"}
    assert describe_device(SAFARI_IPHONE) == {"browser": "Safari", "os": "iOS"}
    assert describe_device(CHROME_ANDROID) == {"browser": "Chrome", "os": "Android"}


def test_an_unknown_or_missing_agent_is_empty_rather_than_guessed():
    assert describe_device(None) == {"browser": "", "os": ""}
    assert describe_device("curl/8.5.0") == {"browser": "", "os": ""}


# -- the name -----------------------------------------------------------------------------


def test_a_name_is_single_spaced_and_stripped_of_control_characters():
    assert account._clean_name("  Nguyễn \t Văn\nA  ") == "Nguyễn Văn A"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t", "x" * 256])
def test_an_empty_or_oversized_name_is_refused(raw):
    with pytest.raises(HTTPException) as error:
        account._clean_name(raw)
    assert error.value.status_code == 422


def test_changing_the_name_updates_it_and_records_before_and_after():
    user = _user()
    db = _DB(user)
    result = _run(
        account.update_profile(
            request=_request(),
            payload=account.ProfileUpdate(name="Trần  Thị B"),
            current_user=user,
            db=db,
        )
    )
    assert user.name == "Trần Thị B"
    assert result["name"] == "Trần Thị B"
    entry = next(item for item in db.added if isinstance(item, AuditLog))
    assert entry.action == "profile_update"
    assert entry.detail_json == {
        "before": {"name": "Existing Member"},
        "after": {"name": "Trần Thị B"},
    }


def test_saving_the_same_name_writes_nothing():
    user = _user()
    db = _DB(user)
    _run(
        account.update_profile(
            request=_request(),
            payload=account.ProfileUpdate(name=user.name),
            current_user=user,
            db=db,
        )
    )
    assert db.added == []
    assert db.commits == 0


def test_a_cross_site_request_cannot_rename_the_account():
    user = _user()
    request = SimpleNamespace(
        headers={"origin": "https://evil.example"}, client=SimpleNamespace(host="x")
    )
    with pytest.raises(HTTPException) as error:
        _run(
            account.update_profile(
                request=request,
                payload=account.ProfileUpdate(name="Hacked"),
                current_user=user,
                db=_DB(user),
            )
        )
    assert error.value.status_code == 403


# -- the sessions list --------------------------------------------------------------------


def test_the_list_marks_this_device_and_names_the_others():
    user = _user()
    here = _session(user, "here")
    phone = _session(user, "phone", agent=SAFARI_IPHONE, ip="192.0.2.8")
    db = _DB(user, [here, phone])
    listed = _run(
        account.list_sessions(
            refresh_token_cookie="here",
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    by_id = {item["id"]: item for item in listed}
    assert by_id[str(here.id)]["current"] is True
    assert by_id[str(phone.id)]["current"] is False
    assert by_id[str(phone.id)]["os"] == "iOS"
    assert by_id[str(phone.id)]["ip_address"] == "192.0.2.8"
    assert by_id[str(here.id)]["signed_in_at"].endswith("Z")


def test_the_list_only_reads_live_sessions_of_the_caller():
    user = _user()
    db = _DB(user, [_session(user, "here")])
    _run(
        account.list_sessions(
            refresh_token_cookie="here",
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    sql = next(item for item in db.sql if "from refresh_sessions" in item)
    assert "refresh_sessions.user_id =" in sql
    assert "refresh_sessions.revoked_at is null" in sql
    assert "refresh_sessions.expires_at >" in sql


def test_no_cookie_means_no_session_is_marked_current():
    user = _user()
    db = _DB(user, [_session(user, "here")])
    listed = _run(
        account.list_sessions(
            refresh_token_cookie=None,
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    assert [item["current"] for item in listed] == [False]


# -- ending one session -------------------------------------------------------------------


def test_ending_another_device_revokes_only_that_row_and_leaves_the_cookies():
    user = _user()
    here = _session(user, "here")
    phone = _session(user, "phone", agent=SAFARI_IPHONE)
    db = _DB(user, [here, phone])
    response = Response()
    result = _run(
        account.revoke_session(
            request=_request(),
            response=response,
            session_id=phone.id,
            refresh_token_cookie="here",
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    assert result == {"status": "revoked", "current": False}
    assert phone.revoked_at is not None
    assert here.revoked_at is None
    assert "set-cookie" not in response.headers
    assert db.audit_actions() == ["session_revoke"]


def test_ending_this_device_clears_the_cookies():
    user = _user()
    here = _session(user, "here")
    db = _DB(user, [here])
    response = Response()
    result = _run(
        account.revoke_session(
            request=_request(),
            response=response,
            session_id=here.id,
            refresh_token_cookie="here",
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    assert result["current"] is True
    assert here.revoked_at is not None
    assert response.headers.getlist("set-cookie")


def test_someone_elses_session_is_not_found():
    user = _user()
    stranger = _user()
    theirs = _session(stranger, "theirs")
    db = _DB(user, [theirs])
    with pytest.raises(HTTPException) as error:
        _run(
            account.revoke_session(
                request=_request(),
                response=Response(),
                session_id=theirs.id,
                refresh_token_cookie="here",
                host_refresh_token_cookie=None,
                current_user=user,
                db=db,
            )
        )
    assert error.value.status_code == 404
    assert theirs.revoked_at is None


# -- signing out everywhere else ----------------------------------------------------------


def test_signing_out_everywhere_else_keeps_this_device_signed_in_with_fresh_tokens():
    user = _user()
    here = _session(user, "here")
    others = [_session(user, "a"), _session(user, "b", agent=SAFARI_IPHONE)]
    db = _DB(user, [here, *others])
    response = Response()
    result = _run(
        account.revoke_other_sessions(
            request=_request(),
            response=response,
            refresh_token_cookie="here",
            host_refresh_token_cookie=None,
            current_user=user,
            db=db,
        )
    )
    assert result == {"status": "revoked", "revoked": 2}
    assert user.auth_version == 5  # cuts the other devices' access tokens off at once
    assert all(item.revoked_at is not None for item in (here, *others))
    replacement = [item for item in db.added if isinstance(item, RefreshSession)]
    assert len(replacement) == 1
    assert replacement[0].revoked_at is None
    # The replacement is the same sign-in: the device list must not show it as a new one.
    assert replacement[0].family_id == here.family_id
    assert replacement[0].signed_in_at == here.signed_in_at
    assert response.headers.getlist("set-cookie")
    assert db.audit_actions() == ["sessions_revoke_others"]


def test_nothing_is_revoked_when_the_current_session_cannot_be_identified():
    user = _user()
    sessions = [_session(user, "a"), _session(user, "b")]
    db = _DB(user, sessions)
    for cookie in (None, "not-one-of-them"):
        with pytest.raises(HTTPException) as error:
            _run(
                account.revoke_other_sessions(
                    request=_request(),
                    response=Response(),
                    refresh_token_cookie=cookie,
                    host_refresh_token_cookie=None,
                    current_user=user,
                    db=db,
                )
            )
        assert error.value.status_code == 400
    assert user.auth_version == 4
    assert all(item.revoked_at is None for item in sessions)


# -- what a refresh keeps -----------------------------------------------------------------


def test_a_new_sign_in_records_where_it_came_from():
    user = _user()
    db = _DB(user)
    token = auth.AuthService(None).create_refresh_token(user)
    _run(auth._store_refresh_session(db, user, token, _request(EDGE_WINDOWS)))
    stored = next(item for item in db.added if isinstance(item, RefreshSession))
    assert stored.user_agent == EDGE_WINDOWS
    assert stored.ip_address == "198.51.100.7"
    assert stored.family_id is not None
    assert stored.signed_in_at is not None


def test_a_rotation_keeps_the_original_sign_in_and_family():
    user = _user()
    db = _DB(user)
    previous = _session(user, "old", agent=SAFARI_IPHONE, ip="192.0.2.8")
    token = auth.AuthService(None).create_refresh_token(user)
    # A refresh from a request that carries no agent must not blank what was recorded.
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host=""))
    _run(auth._store_refresh_session(db, user, token, request, previous=previous))
    stored = next(item for item in db.added if isinstance(item, RefreshSession))
    assert stored.family_id == previous.family_id
    assert stored.signed_in_at == previous.signed_in_at
    assert stored.user_agent == SAFARI_IPHONE
    assert stored.ip_address == "192.0.2.8"
    assert stored.last_used_at > previous.last_used_at


# -- recent activity ----------------------------------------------------------------------


def test_activity_is_the_callers_own_account_events_only_and_capped():
    user = _user()
    row = AuditLog(
        id=uuid.uuid4(),
        user_id=user.id,
        action="login",
        target_type="user",
        outcome="success",
        detail_json={
            "method": "password",
            "ip": "203.0.113.9",
            "user_agent": CHROME_WINDOWS,
        },
        created_at=datetime(2026, 10, 26, 8, 30),
    )
    db = _DB(user, audit=[row])
    out = _run(account.recent_activity(limit=1000, current_user=user, db=db))
    sql = next(item for item in db.sql if "from audit_logs" in item)
    assert "audit_logs.user_id =" in sql
    assert "audit_logs.action in" in sql
    assert "limit" in sql
    assert out == [
        {
            "id": str(row.id),
            "action": "login",
            "outcome": "success",
            "at": "2026-10-26T08:30:00Z",
            "ip_address": "203.0.113.9",
            "method": "password",
            "browser": "Chrome",
            "os": "Windows",
        }
    ]


def test_the_listed_actions_are_account_events_not_document_administration():
    assert "login" in account.ACTIVITY_ACTIONS
    assert "password_change" in account.ACTIVITY_ACTIONS
    assert not any(
        action.startswith(("article", "draft", "role"))
        for action in account.ACTIVITY_ACTIONS
    )
