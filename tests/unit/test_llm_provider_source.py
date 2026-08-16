import asyncio

from src.core.config import settings
from src.domain import llm_client
from src.domain.llm_config import (
    RuntimeLLMConfig,
    get_runtime_config,
    public_config,
    set_runtime_config,
)


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
            assert kwargs["json"]["stream"] is True
            assert kwargs["json"]["thinking"] == {"type": "disabled"}
            return Response()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
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
