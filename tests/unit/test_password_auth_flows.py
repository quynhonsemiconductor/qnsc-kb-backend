"""The five password/invitation endpoints, pinned on behaviour rather than wording.

Every handler is awaited directly through `asyncio.run` against a hand-rolled session
double, the way `test_knowledge_purge.py` and `test_query_authorization.py` do it: there
is no pytest-asyncio in this environment and no TestClient in these tests.

Password hashing is real. bcrypt is installed and a handful of hashes cost nothing, and
the whole point of the 72-BYTE ceiling is that `get_password_hash` raises above it — a
mocked hasher would hide exactly the defect the policy exists to prevent.

What a fake session cannot prove, and is therefore NOT claimed here: the RLS policies that
let an unauthenticated request insert a reset grant, the UNIQUE constraint on token_hash,
and the partial index behind the repeat-request lookup. Those need the real pg16 schema.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from sqlalchemy.dialects import postgresql

from src.api.routers import auth
from src.core.security import get_password_hash, verify_password
from src.models.ops import NotificationQueue
from src.models.sessions import PasswordResetToken, RefreshSession
from src.models.user import Department, Invitation, User

#: 25 characters, 75 UTF-8 bytes. Reaching past 72 bytes inside 25 characters takes an
#: unbroken run of 3-byte Vietnamese vowels — any ASCII consonant would drag the average
#: below 2.88 bytes per character — which is why this reads as a vowel drill rather than a
#: phrase. It is still a perfectly typable password, and a character-counting policy waves
#: it straight through into a ValueError from bcrypt.
VIETNAMESE_OVER_72_BYTES = "ệỗậốềứừữựặọỏồổỗớờởỡợủữỳỹỵ"


class _Result:
    """The slice of a SQLAlchemy result these handlers touch."""

    def __init__(self, rows=None, rowcount: int = 0):
        self.rows = [row for row in (rows or []) if row is not None]
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return list(self.rows)

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _AuthDB:
    """Routes reads by target table and records every write in issue order.

    Ordering is recorded across statements AND `add()` calls in one `events` list, because
    one of the properties under test is relative: the replacement refresh session must be
    inserted AFTER the blanket revocation, or the revocation kills it too.
    """

    def __init__(
        self,
        *,
        invitation: Invitation | None = None,
        grant: PasswordResetToken | None = None,
        user: User | None = None,
        user_by_email: User | None = None,
        departments: list[Department] | None = None,
    ):
        self.invitation = invitation
        self.grant = grant
        self.user = user
        self.user_by_email = user_by_email
        self.departments = list(departments or [])
        self.sql: list[str] = []
        self.added: list[object] = []
        self.events: list[tuple[str, object]] = []
        self.commits = 0
        self.flushes = 0

    # -- reads -------------------------------------------------------------------
    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            rendered = str(statement.compile(dialect=postgresql.dialect()))
        except Exception:
            rendered = str(statement)
        self.sql.append(rendered)
        lowered = " ".join(rendered.lower().split())
        self.events.append(("sql", lowered))
        if not lowered.startswith("select"):
            return _Result(rowcount=1)
        if "from invitations" in lowered:
            return _Result([self.invitation])
        if "from password_reset_tokens" in lowered:
            return _Result([self.grant])
        if "from departments" in lowered:
            return _Result(self.departments)
        if "from users" in lowered:
            if "users.email =" in lowered:
                return _Result([self.user_by_email])
            if "users.id =" in lowered:
                # The row the handler just inserted is readable straight after the flush.
                return _Result([self.user or self._last_added_user()])
            return _Result()  # bootstrap_rbac's unfiltered backfill scan
        return _Result()

    def _last_added_user(self) -> User | None:
        users = [item for item in self.added if isinstance(item, User)]
        return users[-1] if users else None

    # -- writes ------------------------------------------------------------------
    def add(self, item) -> None:
        self.added.append(item)
        self.events.append(("add", item))

    async def flush(self) -> None:
        self.flushes += 1
        # Stand in for the INSERT's server-side id assignment. Without this, "the
        # invitation records WHICH account accepted it" would compare None to None.
        for item in self.added:
            if getattr(item, "id", None) is None:
                item.id = uuid.uuid4()

    async def commit(self) -> None:
        self.commits += 1
        await self.flush()

    async def refresh(self, _item) -> None:
        return None

    # -- assertion helpers -------------------------------------------------------
    def added_of(self, kind: type) -> list:
        return [item for item in self.added if isinstance(item, kind)]

    def statements_matching(self, fragment: str) -> list[str]:
        return [sql for sql in self.sql if fragment in " ".join(sql.lower().split())]

    def first_event_index(self, kind: str, needle) -> int:
        for index, (event, payload) in enumerate(self.events):
            if event != kind:
                continue
            if kind == "sql" and needle in payload:
                return index
            if kind == "add" and isinstance(payload, needle):
                return index
        raise AssertionError(f"no {kind} event matching {needle!r}")


@pytest.fixture(autouse=True)
def _allow_every_request(monkeypatch):
    """Take Redis and the shared 10/minute window out of the picture.

    The limiter is keyed per address and per IP; several tests reuse both, and a real
    `allow` reaches for Redis on every call. Neither belongs in these assertions.
    """

    async def _allow(_key: str):
        return True, 0

    monkeypatch.setattr(auth.auth_rate_limiter, "allow", _allow)


def _request(*, origin: str | None = None) -> SimpleNamespace:
    headers = {"origin": origin} if origin else {}
    return SimpleNamespace(
        headers=headers, client=SimpleNamespace(host="203.0.113.9")
    )


def _raw_token() -> str:
    """Long enough for the 16-character floor the request models enforce."""
    return uuid.uuid4().hex + uuid.uuid4().hex


def _invitation(
    *,
    token: str,
    role: str = "Staff",
    email: str = "invitee@qnsc.vn",
    used_at: datetime | None = None,
    revoked_at: datetime | None = None,
    expires_at: datetime | None = None,
    audience_ids: list[str] | None = None,
) -> Invitation:
    return Invitation(
        id=uuid.uuid4(),
        email=email,
        name="Invited Person",
        role=role,
        company_domain="qnsc.vn",
        audience_ids=audience_ids,
        token_hash=auth._token_hash(token),
        expires_at=expires_at or (datetime.utcnow() + timedelta(days=3)),
        used_at=used_at,
        revoked_at=revoked_at,
    )


def _grant(
    *,
    token: str,
    user_id: uuid.UUID,
    used_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> PasswordResetToken:
    return PasswordResetToken(
        id=uuid.uuid4(),
        user_id=user_id,
        token_hash=auth._token_hash(token),
        expires_at=expires_at or (datetime.utcnow() + timedelta(hours=1)),
        used_at=used_at,
        requested_for_email="member@qnsc.vn",
    )


def _account(*, password: str = "correct horse battery", auth_version: int = 4) -> User:
    user = User(
        id=uuid.uuid4(),
        email="member@qnsc.vn",
        name="Existing Member",
        password_hash=get_password_hash(password),
        company_domain="qnsc.vn",
        role="Staff",
        active=True,
        auth_version=auth_version,
    )
    user.roles = []
    user.departments = []
    user.department_ownerships = []
    return user


def _route_status(path: str) -> int:
    """The declared status code of a route, read from the router rather than the source."""
    for route in auth.router.routes:
        if getattr(route, "path", None) == path:
            return route.status_code
    raise AssertionError(f"no route registered at {path}")


# ------------------------------------------------------------------- the password policy


def test_a_password_one_character_short_of_the_floor_is_refused():
    with pytest.raises(HTTPException) as error:
        auth._validate_password("Abcdefghij1")  # 11 characters
    assert error.value.status_code == 422


def test_a_password_exactly_at_the_floor_is_accepted():
    assert auth._validate_password("Abcdefghij12") == "Abcdefghij12"


def test_a_vietnamese_password_over_72_bytes_is_refused_despite_its_character_count():
    """The byte/character split IS the policy.

    25 characters is far under any character-based ceiling, but 75 UTF-8 bytes is over
    bcrypt's hard limit, so accepting this would hand `get_password_hash` a ValueError and
    turn a user's password choice into a 500.
    """
    assert len(VIETNAMESE_OVER_72_BYTES) == 25
    assert len(VIETNAMESE_OVER_72_BYTES.encode("utf-8")) == 75

    with pytest.raises(HTTPException) as error:
        auth._validate_password(VIETNAMESE_OVER_72_BYTES)
    assert error.value.status_code == 422


def test_the_byte_ceiling_matches_what_the_hasher_will_actually_accept():
    """Pins the policy to bcrypt rather than to a number written down twice.

    Every password the policy admits must hash; every one it rejects on length must be the
    kind bcrypt refuses. A policy ceiling above the hasher's is a 500 waiting to happen.
    """
    with pytest.raises(ValueError):
        get_password_hash(VIETNAMESE_OVER_72_BYTES)

    under = "Mậtkhẩuđủdàivớitiếngviệt"  # 24 characters, 40 bytes
    assert len(under.encode("utf-8")) <= 72
    assert auth._validate_password(under) == under
    assert verify_password(under, get_password_hash(under))


# ------------------------------------------------------------- forgot: no enumeration


def _forgot(db, email: str):
    return asyncio.run(
        auth.forgot_password(
            request=_request(),
            payload=auth.ForgotPasswordRequest(email=email),
            db=db,
        )
    )


def test_a_known_and_an_unknown_address_get_the_identical_response():
    known_db = _AuthDB(user_by_email=_account())
    unknown_db = _AuthDB(user_by_email=None)

    assert _forgot(known_db, "member@qnsc.vn") == _forgot(
        unknown_db, "nobody@qnsc.vn"
    )


def test_the_forgot_endpoint_answers_202_for_an_address_that_does_not_exist():
    """A 404 here is an oracle: it confirms which of a leaked address list are real."""
    db = _AuthDB(user_by_email=None)
    assert _forgot(db, "nobody@qnsc.vn") == {"status": "sent"}
    assert _route_status("/password/forgot") == 202


def test_an_unknown_address_leaves_no_reset_grant_and_no_queued_email():
    """The response alone is not enough: a written row is observable through timing,
    through the mailbox of whoever owns a mistyped address, and in the audit trail."""
    db = _AuthDB(user_by_email=None)

    _forgot(db, "nobody@qnsc.vn")

    assert db.added_of(PasswordResetToken) == []
    assert db.added_of(NotificationQueue) == []


def test_a_known_address_gets_one_grant_and_one_email_carrying_that_grant():
    """The queued link must be the raw token whose HASH was stored, never the hash."""
    user = _account()
    db = _AuthDB(user_by_email=user)

    _forgot(db, "member@qnsc.vn")

    grants = db.added_of(PasswordResetToken)
    emails = db.added_of(NotificationQueue)
    assert len(grants) == 1
    assert len(emails) == 1
    assert grants[0].user_id == user.id
    assert grants[0].used_at is None

    link = emails[0].payload["text"]
    raw = link.split("token=")[1].split()[0]
    assert auth._token_hash(raw) == grants[0].token_hash
    assert grants[0].token_hash not in link


def test_the_reset_grant_expires_within_the_hour_it_promises():
    db = _AuthDB(user_by_email=_account())

    _forgot(db, "member@qnsc.vn")

    grant = db.added_of(PasswordResetToken)[0]
    assert grant.expires_at - datetime.utcnow() <= timedelta(hours=1)
    assert grant.expires_at > datetime.utcnow()


def test_the_queued_email_is_addressed_to_the_requested_address_only():
    """The notification row is recipient-scoped by RLS; a mismatch leaks it sideways."""
    user = _account()
    db = _AuthDB(user_by_email=user)

    _forgot(db, "Member@Qnsc.VN")

    email = db.added_of(NotificationQueue)[0]
    assert email.recipient_user_id == user.id
    assert email.payload["to"] == "member@qnsc.vn"


# ------------------------------------------------------------------ reset: token states


def _reset(db, token: str, password: str = "Nhậpmậtkhẩumới1"):
    return asyncio.run(
        auth.reset_password(
            request=_request(),
            payload=auth.ResetPasswordRequest(token=token, password=password),
            db=db,
        )
    )


def test_a_reset_token_that_was_never_issued_is_not_found():
    db = _AuthDB(grant=None)

    with pytest.raises(HTTPException) as error:
        _reset(db, _raw_token())
    assert error.value.status_code == 404


def test_a_reset_token_that_was_already_spent_is_gone():
    """410, not 404: a replayed link is a different event from an invented one, and the
    distinction is what makes a replay visible instead of indistinguishable from noise."""
    token = _raw_token()
    user = _account()
    db = _AuthDB(
        grant=_grant(token=token, user_id=user.id, used_at=datetime.utcnow()),
        user=user,
    )

    with pytest.raises(HTTPException) as error:
        _reset(db, token)
    assert error.value.status_code == 410


def test_an_expired_reset_token_is_gone():
    token = _raw_token()
    user = _account()
    db = _AuthDB(
        grant=_grant(
            token=token,
            user_id=user.id,
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        ),
        user=user,
    )

    with pytest.raises(HTTPException) as error:
        _reset(db, token)
    assert error.value.status_code == 410


def test_a_rejected_reset_never_touches_the_stored_password():
    """Each refusal path must return before the write, not after it."""
    token = _raw_token()
    user = _account(password="the original one")
    original = user.password_hash
    db = _AuthDB(
        grant=_grant(token=token, user_id=user.id, used_at=datetime.utcnow()),
        user=user,
    )

    with pytest.raises(HTTPException):
        _reset(db, token)

    assert user.password_hash == original
    assert verify_password("the original one", user.password_hash)


def test_a_reset_for_a_deactivated_account_does_not_reactivate_it():
    """A grant outliving a deactivation must not be a way back in."""
    token = _raw_token()
    user = _account()
    user.active = False
    db = _AuthDB(grant=_grant(token=token, user_id=user.id), user=user)

    with pytest.raises(HTTPException) as error:
        _reset(db, token)
    assert error.value.status_code == 410
    assert user.active is False


# ---------------------------------------------------------------- reset: the happy path


def _successful_reset(new_password: str = "Nhậpmậtkhẩumới1"):
    token = _raw_token()
    user = _account(password="the original one", auth_version=4)
    grant = _grant(token=token, user_id=user.id)
    db = _AuthDB(grant=grant, user=user)
    result = _reset(db, token, new_password)
    return db, user, grant, result


def test_a_successful_reset_reports_success_without_signing_the_user_in():
    db, _user, _grant_row, result = _successful_reset()

    assert result == {"status": "reset"}
    # No cookies to set and no session to store: the UI sends them to the login form,
    # which is what proves the new password actually works.
    assert db.added_of(RefreshSession) == []


def test_a_successful_reset_installs_the_new_password_and_retires_the_old_one():
    _db, user, _grant_row, _result = _successful_reset("Nhậpmậtkhẩumới1")

    assert verify_password("Nhậpmậtkhẩumới1", user.password_hash)
    assert not verify_password("the original one", user.password_hash)


def test_a_successful_reset_increments_auth_version():
    """Without this every access token minted before the reset stays valid to its expiry —
    so whoever the reset was meant to lock out keeps working for up to an hour."""
    _db, user, _grant_row, _result = _successful_reset()

    assert user.auth_version == 5


def test_a_successful_reset_marks_the_grant_used_so_the_link_cannot_be_replayed():
    _db, _user, grant, _result = _successful_reset()

    assert grant.used_at is not None


def test_a_successful_reset_revokes_the_refresh_sessions():
    """auth_version alone stops access tokens; a live refresh row mints new ones."""
    db, user, _grant_row, _result = _successful_reset()

    revocations = [
        sql
        for sql in db.statements_matching("update refresh_sessions")
        if "revoked_at" in sql
    ]
    assert revocations, "no refresh_sessions revocation was issued"
    assert str(user.id) in revocations[0] or "user_id" in revocations[0]


def test_a_successful_reset_voids_the_users_other_outstanding_grants():
    """A second link from the same mailbox would otherwise still work afterwards."""
    db, _user, _grant_row, _result = _successful_reset()

    voided = [
        sql
        for sql in db.statements_matching("update password_reset_tokens")
        if "used_at" in sql
    ]
    assert voided, "other outstanding grants were left usable"


def test_a_reset_refuses_a_password_that_bcrypt_could_not_hash():
    """The policy runs before the token lookup, so an over-long password cannot spend the
    grant on its way to a 500."""
    token = _raw_token()
    user = _account()
    grant = _grant(token=token, user_id=user.id)
    db = _AuthDB(grant=grant, user=user)

    with pytest.raises(HTTPException) as error:
        _reset(db, token, VIETNAMESE_OVER_72_BYTES)

    assert error.value.status_code == 422
    assert grant.used_at is None, "a refused password must not burn the link"


# ---------------------------------------------------------------------- change password


def _change(db, user, current: str, new: str):
    response = Response()
    result = asyncio.run(
        auth.change_password(
            request=_request(),
            response=response,
            payload=auth.ChangePasswordRequest(
                current_password=current, new_password=new
            ),
            current_user=user,
            db=db,
        )
    )
    return result, response


def test_changing_a_password_with_the_wrong_current_one_is_unauthorized():
    """A stolen access token grants requests, not the right to lock the owner out."""
    user = _account(password="the original one")
    db = _AuthDB(user=user)

    with pytest.raises(HTTPException) as error:
        _change(db, user, "not the original one", "Nhậpmậtkhẩumới1")
    assert error.value.status_code == 401


def test_a_refused_change_leaves_the_stored_hash_and_auth_version_untouched():
    user = _account(password="the original one", auth_version=4)
    original = user.password_hash
    db = _AuthDB(user=user)

    with pytest.raises(HTTPException):
        _change(db, user, "not the original one", "Nhậpmậtkhẩumới1")

    assert user.password_hash == original
    assert verify_password("the original one", user.password_hash)
    assert user.auth_version == 4
    assert db.added_of(RefreshSession) == []


def test_reusing_the_current_password_as_the_new_one_is_refused():
    user = _account(password="Mậtkhẩuhiệntại1")
    db = _AuthDB(user=user)

    with pytest.raises(HTTPException) as error:
        _change(db, user, "Mậtkhẩuhiệntại1", "Mậtkhẩuhiệntại1")
    assert error.value.status_code == 422


def _successful_change():
    user = _account(password="the original one", auth_version=4)
    db = _AuthDB(user=user)
    result, response = _change(db, user, "the original one", "Nhậpmậtkhẩumới1")
    return db, user, result, response


def test_a_successful_change_reports_success_and_installs_the_new_password():
    _db, user, result, _response = _successful_change()

    assert result == {"status": "changed"}
    assert verify_password("Nhậpmậtkhẩumới1", user.password_hash)
    assert not verify_password("the original one", user.password_hash)


def test_a_successful_change_increments_auth_version():
    _db, user, _result, _response = _successful_change()

    assert user.auth_version == 5


def test_a_successful_change_issues_a_new_refresh_session():
    """auth_version invalidates THIS caller's tokens too, so without a fresh pair the user
    is signed out by their own successful password change."""
    db, user, _result, _response = _successful_change()

    sessions = db.added_of(RefreshSession)
    assert len(sessions) == 1
    assert sessions[0].user_id == user.id
    assert sessions[0].revoked_at is None


def test_the_replacement_session_is_stored_after_the_blanket_revocation():
    """Order matters: revoking after the insert would revoke the replacement as well and
    sign the caller out anyway, while every assertion above still passed."""
    db, _user, _result, _response = _successful_change()

    revoked_at = db.first_event_index("sql", "update refresh_sessions")
    minted_at = db.first_event_index("add", RefreshSession)
    assert revoked_at < minted_at


def _cookies(response) -> dict[str, str]:
    """The name=value pair of every Set-Cookie header the handler emitted."""
    jar = {}
    for key, value in response.raw_headers:
        if key != b"set-cookie":
            continue
        pair = value.decode().split(";", 1)[0]
        name, _, token = pair.partition("=")
        jar[name] = token
    return jar


def test_the_reissued_tokens_carry_the_incremented_auth_version():
    """`get_current_user` compares the token's `av` against the row. Minting the pair
    BEFORE the increment would hand the caller tokens that are already stale, so the
    re-mint would look present and still sign them out on the next request."""
    import jwt

    from src.core.config import settings

    _db, user, _result, response = _successful_change()

    jar = _cookies(response)
    for name in ("access_token", "refresh_token"):
        claims = jwt.decode(jar[name], settings.SECRET_KEY, algorithms=["HS256"])
        assert claims["av"] == user.auth_version == 5, name
        assert claims["sub"] == user.email


def test_a_successful_change_sets_both_auth_cookies():
    _db, _user, _result, response = _successful_change()

    assert set(_cookies(response)) >= {"access_token", "refresh_token"}


def test_a_change_refuses_a_password_that_bcrypt_could_not_hash():
    user = _account(password="the original one", auth_version=4)
    original = user.password_hash
    db = _AuthDB(user=user)

    with pytest.raises(HTTPException) as error:
        _change(db, user, "the original one", VIETNAMESE_OVER_72_BYTES)

    assert error.value.status_code == 422
    assert user.password_hash == original
    assert user.auth_version == 4


# ------------------------------------------------------------------- invitation preview


def _preview(db, token: str):
    return asyncio.run(auth.preview_invitation(request=_request(), token=token, db=db))


def test_previewing_an_unknown_invitation_is_not_found():
    db = _AuthDB(invitation=None)

    with pytest.raises(HTTPException) as error:
        _preview(db, _raw_token())
    assert error.value.status_code == 404


def test_previewing_shows_the_invitee_who_the_invitation_is_for():
    token = _raw_token()
    invitation = _invitation(token=token, role="Reviewer")
    db = _AuthDB(invitation=invitation)

    preview = _preview(db, token)

    assert preview == {
        "email": invitation.email,
        "name": invitation.name,
        "role": "Reviewer",
        "company_domain": "qnsc.vn",
        "expires_at": invitation.expires_at,
    }


def test_a_preview_never_leaks_the_token_hash_it_looked_up():
    token = _raw_token()
    invitation = _invitation(token=token)
    db = _AuthDB(invitation=invitation)

    preview = _preview(db, token)

    assert invitation.token_hash not in str(preview)


# -------------------------------------------------------------------- invitation accept


def _accept(db, token: str, password: str = "Mậtkhẩumớicủatôi1"):
    response = Response()
    result = asyncio.run(
        auth.accept_invitation(
            request=_request(),
            response=response,
            payload=auth.InvitationAccept(token=token, password=password),
            db=db,
        )
    )
    return result, response


def test_accepting_an_unknown_invitation_is_not_found():
    db = _AuthDB(invitation=None)

    with pytest.raises(HTTPException) as error:
        _accept(db, _raw_token())
    assert error.value.status_code == 404


def test_accepting_an_already_accepted_invitation_is_a_conflict():
    """409 rather than 410: the link worked, an account exists, and the right advice is
    "sign in" — not "ask for another invitation"."""
    token = _raw_token()
    db = _AuthDB(invitation=_invitation(token=token, used_at=datetime.utcnow()))

    with pytest.raises(HTTPException) as error:
        _accept(db, token)
    assert error.value.status_code == 409


def test_accepting_an_expired_invitation_is_gone():
    token = _raw_token()
    db = _AuthDB(
        invitation=_invitation(
            token=token, expires_at=datetime.utcnow() - timedelta(minutes=1)
        )
    )

    with pytest.raises(HTTPException) as error:
        _accept(db, token)
    assert error.value.status_code == 410


def test_accepting_a_revoked_invitation_is_gone():
    token = _raw_token()
    db = _AuthDB(invitation=_invitation(token=token, revoked_at=datetime.utcnow()))

    with pytest.raises(HTTPException) as error:
        _accept(db, token)
    assert error.value.status_code == 410


def test_a_refused_invitation_creates_no_account():
    token = _raw_token()
    db = _AuthDB(invitation=_invitation(token=token, used_at=datetime.utcnow()))

    with pytest.raises(HTTPException):
        _accept(db, token)

    assert db.added_of(User) == []
    assert db.added_of(RefreshSession) == []


def _successful_accept(*, role: str = "Reviewer", audience_ids=None, departments=None):
    token = _raw_token()
    invitation = _invitation(token=token, role=role, audience_ids=audience_ids)
    db = _AuthDB(
        invitation=invitation, user_by_email=None, departments=departments or []
    )
    result, response = _accept(db, token)
    created = db.added_of(User)[0]
    return db, invitation, created, result, response


def test_accepting_creates_the_account_the_invitation_described():
    _db, invitation, created, _result, _response = _successful_accept()

    assert created.email == invitation.email
    assert created.name == invitation.name
    assert created.company_domain == invitation.company_domain
    assert created.active is True


def test_the_created_account_gets_the_role_the_inviter_chose():
    """Dropping this silently downgrades every invited reviewer to the default role."""
    _db, _invitation, created, _result, _response = _successful_accept(role="Reviewer")

    assert created.role == "Reviewer"


def test_the_created_account_can_sign_in_with_the_password_it_chose():
    _db, _invitation, created, _result, _response = _successful_accept()

    assert verify_password("Mậtkhẩumớicủatôi1", created.password_hash)
    assert created.password_hash != "Mậtkhẩumớicủatôi1"


def test_accepting_marks_the_invitation_used_and_records_which_account_took_it():
    """Without `used_at` the link is reusable; without `accepted_user_id` nothing ties the
    invitation to the account it produced."""
    _db, invitation, created, _result, _response = _successful_accept()

    assert invitation.used_at is not None
    assert invitation.accepted_user_id == created.id
    assert created.id is not None


def test_accepting_applies_the_invitations_audience_departments():
    audience = Department(
        id=uuid.uuid4(),
        company_domain="qnsc.vn",
        name="Engineering",
        active=True,
    )
    _db, _invitation, created, _result, _response = _successful_accept(
        audience_ids=[str(audience.id)], departments=[audience]
    )

    assert [department.id for department in created.departments] == [audience.id]
    assert created.dept == "Engineering"


def test_accepting_signs_the_invitee_in_rather_than_bouncing_them_to_a_login_form():
    db, _invitation, created, result, response = _successful_accept()

    assert result["token_type"] == "bearer"
    assert result["access_token"]
    assert result["user"]["email"] == created.email
    assert set(_cookies(response)) >= {"access_token", "refresh_token"}
    assert len(db.added_of(RefreshSession)) == 1


def test_accepting_refuses_a_password_that_bcrypt_could_not_hash():
    token = _raw_token()
    invitation = _invitation(token=token)
    db = _AuthDB(invitation=invitation, user_by_email=None)

    with pytest.raises(HTTPException) as error:
        _accept(db, token, VIETNAMESE_OVER_72_BYTES)

    assert error.value.status_code == 422
    assert invitation.used_at is None
    assert db.added_of(User) == []


def test_an_invitation_cannot_take_over_an_active_account():
    """Otherwise an old invitation to an address is a password-set primitive for it."""
    token = _raw_token()
    existing = _account(password="the original one")
    db = _AuthDB(
        invitation=_invitation(token=token, email=existing.email),
        user_by_email=existing,
    )

    with pytest.raises(HTTPException) as error:
        _accept(db, token)

    assert error.value.status_code == 409
    assert verify_password("the original one", existing.password_hash)


def test_reinviting_a_deactivated_account_reuses_the_row_instead_of_orphaning_it():
    """A new id would strand the authored articles and audit trail of the old one."""
    token = _raw_token()
    existing = _account(password="the original one", auth_version=4)
    existing.active = False
    original_id = existing.id
    db = _AuthDB(
        invitation=_invitation(token=token, email=existing.email, role="Reviewer"),
        user_by_email=existing,
        user=existing,
    )

    _accept(db, token)

    assert db.added_of(User) == [], "a second row was inserted for the same person"
    assert existing.id == original_id
    assert existing.active is True
    assert existing.role == "Reviewer"
    assert verify_password("Mậtkhẩumớicủatôi1", existing.password_hash)
    # The reactivation resets the password, so tokens issued before it must stop working.
    assert existing.auth_version == 5
