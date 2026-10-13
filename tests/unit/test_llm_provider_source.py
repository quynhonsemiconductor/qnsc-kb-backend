import asyncio

from src.core.config import settings
from src.domain import llm_client
from src.domain.llm_config import (
    RuntimeLLMConfig,
    get_runtime_config,
    public_config,
    set_runtime_config,
)
from src.domain.llm_endpoint_security import ResolvedLLMEndpoint


def teardown_function() -> None:
    set_runtime_config(None)


def test_llm_provider_does_not_fall_back_to_environment(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "environment-key")
    monkeypatch.setattr(settings, "LLM_MODEL", "llama-3.3-70b-versatile")

    set_runtime_config(None)

    assert get_runtime_config() is None
    assert public_config(None)["source"] == "none"
    assert public_config(None)["enabled"] is False


def test_llm_provider_uses_saved_runtime_configuration():
    set_runtime_config(
        RuntimeLLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1/chat/completions",
            api_key="saved-admin-key",
        )
    )

    config = public_config(None)

    assert config["source"] == "admin"
    assert config["enabled"] is True
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-4o-mini"


def test_glm_uses_streaming_events_to_collect_content(monkeypatch):
    class Response:
        is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def aiter_lines(self):
            for line in (
                'data: {"choices":[{"delta":{"reasoning_content":"internal"}}]}',
                'data: {"choices":[{"delta":{"content":"Formatted "}}]}',
                'data: {"choices":[{"delta":{"content":"document"}}],"usage":{"total_tokens":7}}',
                "data: [DONE]",
            ):
                yield line

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, method, url, **kwargs):
            assert method == "POST"
            assert url == "https://93.184.216.34/chat"
            assert kwargs["headers"]["Host"] == "example.test"
            assert kwargs["extensions"]["sni_hostname"] == "example.test"
            assert kwargs["json"]["stream"] is True
            assert kwargs["json"]["thinking"] == {"type": "disabled"}
            return Response()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)

    async def public_endpoint(_url):
        return ResolvedLLMEndpoint("example.test", "93.184.216.34")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", public_endpoint)
    set_runtime_config(
        RuntimeLLMConfig("glm", "glm-5.2", "https://example.test/chat", "key")
    )

    answer, tokens, model, provider = asyncio.run(
        llm_client.complete(
            [{"role": "user", "content": "Format this"}],
            thinking=False,
            max_tokens=100,
        )
    )

    assert answer == "Formatted document"
    assert tokens == 7
    assert model == "glm-5.2"
    assert provider == "glm"


def test_a_custom_gateway_that_always_streams_is_still_read(monkeypatch):
    """An OpenAI-compatible gateway may answer `text/event-stream` for EVERY request.

    The non-streaming path read the body with `response.json()`, which failed on that
    shape with "Expecting value: line 1 column 1" and reached the user as "AI generation
    failed. Please try again." — while the provider had in fact answered correctly, and
    had already passed its connection test.
    """

    class Response:
        status_code = 200
        is_error = False
        text = (
            'data: {"choices":[{"delta":{"role":"assistant","content":"Grounded "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            assert url == "https://93.184.216.34/v1/chat/completions"
            assert kwargs["headers"]["Host"] == "example.test"
            return Response()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)

    async def public_endpoint(_url):
        return ResolvedLLMEndpoint("example.test", "93.184.216.34")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", public_endpoint)
    set_runtime_config(
        RuntimeLLMConfig(
            "custom", "kr/model", "https://example.test/v1/chat/completions", "key"
        )
    )

    answer, _tokens, model, provider = asyncio.run(
        llm_client.complete([{"role": "user", "content": "Ask"}], max_tokens=64)
    )

    assert answer == "Grounded answer"
    assert (model, provider) == ("kr/model", "custom")


def test_a_custom_gateway_returning_nothing_readable_still_fails(monkeypatch):
    """Accepting SSE must not make this path accept an empty or unparseable answer."""

    class Response:
        status_code = 200
        is_error = False
        text = 'data: {"choices":[{"delta":{"content":""}}]}\n\ndata: [DONE]\n\n'

        def json(self):
            raise ValueError("not json")

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            return Response()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)

    async def public_endpoint(_url):
        return ResolvedLLMEndpoint("example.test", "93.184.216.34")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", public_endpoint)
    set_runtime_config(
        RuntimeLLMConfig(
            "custom", "kr/model", "https://example.test/v1/chat/completions", "key"
        )
    )

    try:
        asyncio.run(
            llm_client.complete([{"role": "user", "content": "Ask"}], max_tokens=64)
        )
    except RuntimeError as exc:
        assert "no readable assistant text" in str(exc)
    else:
        raise AssertionError("an empty streamed answer must not be accepted")
