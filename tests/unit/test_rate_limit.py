import asyncio
from types import SimpleNamespace

from src.core.rate_limit import RedisRateLimiter


def test_rate_limiter_falls_back_when_redis_is_unavailable(monkeypatch):
    limiter = RedisRateLimiter("test", limit=1, window_seconds=60)

    async def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    class BrokenRedis:
        @classmethod
        def from_url(cls, *_args, **_kwargs):
            class Client:
                incr = unavailable

                async def aclose(self):
                    pass

            return Client()

    monkeypatch.setattr("src.core.rate_limit.Redis", BrokenRedis)
    assert asyncio.run(limiter.allow("user")) == (True, 0)
    allowed, retry = asyncio.run(limiter.allow("user"))
    assert not allowed and retry > 0


# --- P0-12: the limiters bucket on the visitor address, not the proxy ---

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from src.api.routers import auth
from src.core.config import settings
from src.core.rate_limit import client_ip_from_request


class _Request:
    """The slice of `Request` the helpers touch, with case-insensitive headers."""

    def __init__(self, headers=None, client_host=None):
        self.headers = Headers(headers or {})
        self.client = SimpleNamespace(host=client_host) if client_host else None


def test_client_ip_prefers_cf_connecting_ip():
    request = _Request(
        headers={"CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": "10.9.8.7"},
        client_host="10.0.0.1",
    )
    assert client_ip_from_request(request) == "203.0.113.7"


def test_client_ip_ignores_x_forwarded_for():
    # X-Forwarded-For is client-appendable, so it must never become the bucket key.
    request = _Request(
        headers={"X-Forwarded-For": "203.0.113.9"}, client_host="10.0.0.1"
    )
    assert client_ip_from_request(request) == "10.0.0.1"


def test_client_ip_falls_back_to_socket_then_unknown():
    assert client_ip_from_request(_Request(client_host="10.0.0.1")) == "10.0.0.1"
    assert client_ip_from_request(_Request()) == "unknown"


class _CountingRedis:
    """A fixed-window Redis double backed by one shared dict."""

    store: dict[str, int] = {}

    def __init__(self):
        self._ttl = 60

    async def incr(self, key):
        _CountingRedis.store[key] = _CountingRedis.store.get(key, 0) + 1
        return _CountingRedis.store[key]

    async def expire(self, _key, _seconds):
        return True

    async def ttl(self, _key):
        return self._ttl

    async def aclose(self):
        pass

    @classmethod
    def from_url(cls, *_args, **_kwargs):
        return cls()


def test_auth_limiter_buckets_per_client(monkeypatch):
    """Two visitors behind the same proxy stay in separate buckets.

    Both sockets are identical (the cloudflared pod), so the pre-P0-12 behaviour —
    keying on `request.client.host` — would have shared one bucket; the CF header is
    what separates them.
    """
    _CountingRedis.store.clear()
    monkeypatch.setattr("src.core.rate_limit.Redis", _CountingRedis)
    limiter = RedisRateLimiter("auth-test", limit=2, window_seconds=60)
    a1 = SimpleNamespace(
        headers=Headers({"CF-Connecting-IP": "203.0.113.1"}), client=None
    )
    a2 = SimpleNamespace(
        headers=Headers({"CF-Connecting-IP": "203.0.113.1"}), client=None
    )
    b1 = SimpleNamespace(
        headers=Headers({"CF-Connecting-IP": "203.0.113.2"}), client=None
    )
    assert asyncio.run(limiter.allow(client_ip_from_request(a1))) == (True, 0)
    assert asyncio.run(limiter.allow(client_ip_from_request(a2))) == (True, 0)
    assert asyncio.run(limiter.allow(client_ip_from_request(a1)))[0] is False
    # A different client is not dragged down by the first one's exhausted bucket.
    assert asyncio.run(limiter.allow(client_ip_from_request(b1))) == (True, 0)


def test_ai_limiter_shares_one_per_user_bucket_across_pods(monkeypatch):
    """The AI limit is per person and stored in Valkey, so API pods count together.

    A per-process counter would give every pod its own `AI_RATE_LIMIT_PER_MINUTE`, which
    resets on deploy and multiplies the limit by the replica count. Two limiter objects
    sharing one store stand in for two pods here.
    """
    _CountingRedis.store.clear()
    monkeypatch.setattr("src.core.rate_limit.Redis", _CountingRedis)
    pod_a = RedisRateLimiter("ai", limit=2, window_seconds=60)
    pod_b = RedisRateLimiter("ai", limit=2, window_seconds=60)

    assert asyncio.run(pod_a.allow("user-a")) == (True, 0)
    assert asyncio.run(pod_b.allow("user-a")) == (True, 0)
    assert asyncio.run(pod_a.allow("user-a"))[0] is False
    assert asyncio.run(pod_b.allow("user-a"))[0] is False
    # A different person — including one behind the same office address — is unaffected.
    assert asyncio.run(pod_b.allow("user-b")) == (True, 0)


def test_the_ai_endpoint_buckets_on_the_authenticated_user(monkeypatch):
    """A shared office IP must not put every member in one AI bucket.

    The handler passes the authenticated user's id and never sees an address, so two
    people behind the same Cloudflare address are limited independently.
    """
    import uuid

    from src.api.routers import ai

    user = SimpleNamespace(id=uuid.uuid4())
    seen: list[str] = []

    async def record(key):
        seen.append(key)
        return True, 0

    class _FakeRepository:
        def __init__(self, _db):
            self.conversation_id = uuid.uuid4()

        async def create_conversation(self, *_args):
            return SimpleNamespace(id=self.conversation_id, title="Question")

        async def start_user_turn(self, *_args, **_kwargs):
            return None

        async def add_message(self, *_args, **_kwargs):
            return None

    class _FakeAIService:
        async def ask(self, *_args, **_kwargs):
            return {"answer": "Answer", "citations": [], "log_id": None}

    monkeypatch.setattr(ai, "AIRepository", _FakeRepository)
    monkeypatch.setattr(ai, "get_ai_service", lambda _db, _sources=None: _FakeAIService())
    monkeypatch.setattr(ai, "ai_rate_limiter", SimpleNamespace(allow=record))

    asyncio.run(
        ai.ask_question(
            ai.AskRequest(question="What is the leave policy?"),
            current_user=user,
            db=object(),
        )
    )

    assert seen == [str(user.id)]


def test_auth_limiter_fails_closed_when_redis_is_unavailable(monkeypatch):
    """The auth limiter refuses instead of degrading on a Redis outage."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")

    async def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    async def aclose(*_args, **_kwargs):
        pass

    class BrokenRedis:
        @classmethod
        def from_url(cls, *_args, **_kwargs):
            return SimpleNamespace(
                incr=unavailable,
                expire=unavailable,
                ttl=unavailable,
                aclose=aclose,
            )

    monkeypatch.setattr("src.core.rate_limit.Redis", BrokenRedis)
    limiter = RedisRateLimiter("auth", limit=10, window_seconds=60, fail_closed=True)
    assert asyncio.run(limiter.allow("ip:203.0.113.1")) == (False, 60)


def test_invite_preview_buckets_on_the_visitor_address(monkeypatch):
    """The endpoint passes the CF-derived address, not the socket, to the limiter.

    The limiter is stubbed to refuse, so the handler must raise 429 before it touches
    the database — the recorded key is the whole assertion.
    """
    seen: list[str] = []

    async def record(key):
        seen.append(key)
        return False, 42

    monkeypatch.setattr(auth, "auth_rate_limiter", SimpleNamespace(allow=record))
    request = _Request(
        headers={"CF-Connecting-IP": "203.0.113.7"}, client_host="10.250.0.5"
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(auth.preview_invitation(request=request, token="t" * 32, db=None))
    assert excinfo.value.status_code == 429
    assert seen == ["invite-preview:203.0.113.7"]
