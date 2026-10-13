import asyncio
from types import SimpleNamespace

import httpx
import jwt
import pytest
from fastapi import HTTPException

from src.api.routers import llm as llm_router
from src.domain import llm_client, llm_endpoint_security, llm_verification
from src.domain.llm_client import Provider
from src.domain.llm_endpoint_security import ResolvedLLMEndpoint


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _DB:
    def __init__(self, row=None):
        self.row = row
        self.added = []
        self.commits = 0

    async def execute(self, _statement):
        return _Result(self.row)

    def add(self, row):
        self.added.append(row)
        self.row = row

    async def commit(self):
        self.commits += 1

    async def refresh(self, _row):
        return None


def test_custom_endpoint_is_available_without_official_override_flag(monkeypatch):
    monkeypatch.setattr(llm_router.settings, "LLM_ALLOW_CUSTOM_BASE_URL", False)

    assert (
        llm_router._clean_base_url(
            "custom", "https://models.example.com/v1/chat/completions/"
        )
        == "https://models.example.com/v1/chat/completions"
    )


def test_custom_endpoint_rejects_queries_and_local_names():
    with pytest.raises(HTTPException, match="query or fragment"):
        llm_router._clean_base_url(
            "custom", "https://models.example.com/v1/chat/completions?key=secret"
        )
    with pytest.raises(HTTPException, match="local network"):
        llm_router._clean_base_url(
            "custom", "https://provider.internal/v1/chat/completions"
        )


def test_dns_resolution_fails_closed_when_any_answer_is_private(monkeypatch):
    class Loop:
        async def getaddrinfo(self, *_args, **_kwargs):
            return [
                (2, 1, 6, "", ("203.0.113.20", 443)),
                (2, 1, 6, "", ("10.0.0.5", 443)),
            ]

    monkeypatch.setattr(
        llm_endpoint_security.asyncio, "get_running_loop", lambda: Loop()
    )

    with pytest.raises(
        llm_endpoint_security.UnsafeLLMEndpointError, match="private or reserved"
    ):
        asyncio.run(
            llm_endpoint_security.validate_public_endpoint_resolution(
                "https://models.example.com/v1/chat/completions"
            )
        )


def test_verification_token_is_bound_to_user_and_exact_candidate():
    token, _expires = llm_verification.issue_verification_token(
        "admin-1",
        "custom",
        "model-a",
        "https://models.example.com/v1/chat/completions",
        "secret-key",
    )

    assert llm_verification.verify_token(
        token,
        "admin-1",
        "custom",
        "model-a",
        "https://models.example.com/v1/chat/completions",
        "secret-key",
    )
    with pytest.raises(ValueError, match="changed"):
        llm_verification.verify_token(
            token,
            "admin-2",
            "custom",
            "model-a",
            "https://models.example.com/v1/chat/completions",
            "secret-key",
        )
    with pytest.raises(ValueError, match="changed"):
        llm_verification.verify_token(
            token,
            "admin-1",
            "custom",
            "model-b",
            "https://models.example.com/v1/chat/completions",
            "secret-key",
        )


def test_verification_token_rejects_expired_or_tampered_value(monkeypatch):
    monkeypatch.setattr(
        llm_verification.jwt,
        "decode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(jwt.ExpiredSignatureError()),
    )

    with pytest.raises(ValueError, match="expired or is invalid"):
        llm_verification.verify_token(
            "invalid",
            "admin-1",
            "custom",
            "model-a",
            "https://models.example.com/v1/chat/completions",
            "secret-key",
        )


def test_probe_accepts_openai_response_and_never_follows_redirects(monkeypatch):
    observed = {}

    async def validate(_url):
        observed["validated"] = True
        return ResolvedLLMEndpoint("models.example.com", "93.184.216.34")

    class Response:
        is_redirect = False
        is_error = False
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_bytes(self):
            yield b'{"choices":[{"message":{"content":"OK"}}]}'

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, method, url, **kwargs):
            observed["method"] = method
            observed["url"] = url
            observed["authorization"] = kwargs["headers"]["Authorization"]
            observed["host"] = kwargs["headers"]["Host"]
            observed["sni"] = kwargs["extensions"]["sni_hostname"]
            return Response()

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", validate)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    provider = Provider(
        "custom",
        "model-a",
        "https://models.example.com/v1/chat/completions",
        "secret-key",
        custom_endpoint=True,
    )

    assert asyncio.run(llm_client.probe_provider(provider)) >= 1
    assert observed["validated"] is True
    assert observed["follow_redirects"] is False
    assert observed["trust_env"] is False
    assert observed["method"] == "POST"
    assert observed["url"] == "https://93.184.216.34/v1/chat/completions"
    assert observed["host"] == "models.example.com"
    assert observed["sni"] == "models.example.com"
    assert observed["authorization"] == "Bearer secret-key"


def _probe_against_body(monkeypatch, body: bytes) -> int:
    """Run `probe_provider` against one exact response body."""

    async def validate(_url):
        return ResolvedLLMEndpoint("models.example.com", "93.184.216.34")

    class Response:
        is_redirect = False
        is_error = False
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_bytes(self):
            yield body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", validate)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    provider = Provider(
        "custom",
        "model-a",
        "https://models.example.com/v1/chat/completions",
        "secret-key",
        custom_endpoint=True,
    )
    return asyncio.run(llm_client.probe_provider(provider))


def test_probe_accepts_an_event_stream_from_a_gateway_that_always_streams(monkeypatch):
    """Some OpenAI-compatible gateways answer `text/event-stream` for EVERY request.

    The body is then `data:` chunks rather than one JSON object, so the probe's
    `json.loads` saw nothing parseable and reported "did not return a valid Chat
    Completions response". Because saving a provider requires a successful test, an
    endpoint the workspace can genuinely use could not be configured at all.
    """
    body = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"O"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"K"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert _probe_against_body(monkeypatch, body) >= 1


def test_probe_accepts_a_single_event_carrying_a_whole_message(monkeypatch):
    """A gateway that streams one non-streaming response as a single `message` event."""
    body = b'data: {"choices":[{"message":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
    assert _probe_against_body(monkeypatch, body) >= 1


def test_probe_recovers_an_answer_left_in_a_streamed_reasoning_channel(monkeypatch):
    """Same recovery the streaming answer path performs, via the shared sentinel."""
    from src.domain.llm_client import GROUNDED_SENTINEL

    body = (
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking '
        + GROUNDED_SENTINEL.encode()
        + b' OK"}}]}\n\ndata: [DONE]\n\n'
    )
    assert _probe_against_body(monkeypatch, body) >= 1


@pytest.mark.parametrize(
    "body",
    [
        b"not json and not an event stream",
        b"data: {not json}\n\n",
        b'data: {"choices":[]}\n\ndata: [DONE]\n\n',
        b'data: {"choices":[{"delta":{"content":""}}]}\n\ndata: [DONE]\n\n',
        b"",
    ],
)
def test_probe_still_rejects_a_response_with_no_assistant_text(monkeypatch, body):
    """Accepting SSE must not turn the probe into something that accepts anything."""
    with pytest.raises(RuntimeError):
        _probe_against_body(monkeypatch, body)


def test_probe_rejects_redirect_without_reading_or_following_it(monkeypatch):
    class Response:
        is_redirect = True
        is_error = False
        status_code = 307

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_bytes(self):
            raise AssertionError("A redirect body must not be read")
            yield b""

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    async def validate(_url):
        return ResolvedLLMEndpoint("models.example.com", "93.184.216.34")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", validate)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)

    with pytest.raises(RuntimeError, match="redirected"):
        asyncio.run(
            llm_client.probe_provider(
                Provider(
                    "custom",
                    "model-a",
                    "https://models.example.com/v1/chat/completions",
                    "secret-key",
                    custom_endpoint=True,
                )
            )
        )


def test_guarded_streaming_error_never_reads_provider_body(monkeypatch):
    class Response:
        is_redirect = False
        is_error = True
        status_code = 401
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aread(self):
            raise AssertionError("A guarded provider body must not be read")

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    async def validate(_url):
        return ResolvedLLMEndpoint("models.example.com", "93.184.216.34")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", validate)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        llm_client,
        "resolve_provider",
        lambda *_args, **_kwargs: Provider(
            "glm",
            "glm-compatible",
            "https://models.example.com/v1/chat/completions",
            "secret-key",
            custom_endpoint=True,
        ),
    )

    with pytest.raises(llm_client.ProviderAuthError, match="rejected"):
        asyncio.run(llm_client.complete([{"role": "user", "content": "test"}]))


def test_admin_test_reuses_saved_key_without_mutating_config(monkeypatch):
    row = SimpleNamespace(provider="custom", encrypted_api_key="encrypted-key")
    db = _DB(row)
    user = SimpleNamespace(id="admin-1")
    observed = {}

    monkeypatch.setattr(llm_router, "decrypt_api_key", lambda value: "saved-key")

    async def probe(provider):
        observed["provider"] = provider
        return 24

    monkeypatch.setattr(llm_router, "probe_provider", probe)
    payload = llm_router.LLMConfigCandidate(
        provider="custom",
        model="model-a",
        base_url="https://models.example.com/v1/chat/completions",
    )

    result = asyncio.run(llm_router.test_llm_config(payload, user, db))

    assert result["ok"] is True
    assert result["latency_ms"] == 24
    assert result["verification_token"]
    assert observed["provider"].api_key == "saved-key"
    assert db.commits == 0
    assert db.added == []


def test_saved_key_is_not_reused_for_a_different_provider(monkeypatch):
    row = SimpleNamespace(provider="openai", encrypted_api_key="encrypted-key")
    db = _DB(row)
    user = SimpleNamespace(id="admin-1")
    monkeypatch.setattr(llm_router, "decrypt_api_key", lambda _value: "openai-key")
    payload = llm_router.LLMConfigCandidate(
        provider="custom",
        model="model-a",
        base_url="https://models.example.com/v1/chat/completions",
    )

    with pytest.raises(HTTPException, match="Enter an API key for this provider"):
        asyncio.run(llm_router.test_llm_config(payload, user, db))


def test_changed_config_requires_matching_test_before_save(monkeypatch):
    db = _DB(None)
    user = SimpleNamespace(id="admin-1")
    payload = llm_router.LLMConfigUpdate(
        enabled=True,
        provider="custom",
        model="model-a",
        base_url="https://models.example.com/v1/chat/completions",
        api_key="secret-key",
    )

    with pytest.raises(HTTPException, match="Test this provider successfully"):
        asyncio.run(llm_router.update_llm_config(payload, user, db))
    assert db.commits == 0


def test_unchanged_saved_config_can_toggle_without_retesting(monkeypatch):
    row = SimpleNamespace(
        enabled=True,
        provider="custom",
        model="model-a",
        base_url="https://models.example.com/v1/chat/completions",
        encrypted_api_key="saved-key",
    )
    db = _DB(row)
    user = SimpleNamespace(id="admin-1")
    monkeypatch.setattr(llm_router, "decrypt_api_key", lambda value: value)
    monkeypatch.setattr(
        llm_router, "public_config", lambda value: {"enabled": value.enabled}
    )
    monkeypatch.setattr(llm_router, "set_runtime_config", lambda _value: None)
    payload = llm_router.LLMConfigUpdate(
        enabled=False,
        provider="custom",
        model="model-a",
        base_url="https://models.example.com/v1/chat/completions",
    )

    result = asyncio.run(llm_router.update_llm_config(payload, user, db))

    assert result == {"enabled": False}
    assert db.commits == 1


def test_matching_test_token_allows_save(monkeypatch):
    db = _DB(None)
    user = SimpleNamespace(id="admin-1")
    base_url = "https://models.example.com/v1/chat/completions"
    token, _expires = llm_verification.issue_verification_token(
        "admin-1", "custom", "model-a", base_url, "secret-key"
    )
    monkeypatch.setattr(
        llm_router, "encrypt_api_key", lambda value: f"encrypted:{value}"
    )
    monkeypatch.setattr(
        llm_router, "public_config", lambda value: {"provider": value.provider}
    )
    monkeypatch.setattr(llm_router, "set_runtime_config", lambda _value: None)
    payload = llm_router.LLMConfigUpdate(
        enabled=True,
        provider="custom",
        model="model-a",
        base_url=base_url,
        api_key="secret-key",
        verification_token=token,
    )

    result = asyncio.run(llm_router.update_llm_config(payload, user, db))

    assert result == {"provider": "custom"}
    assert db.commits == 1
    assert db.row.encrypted_api_key == "encrypted:secret-key"
