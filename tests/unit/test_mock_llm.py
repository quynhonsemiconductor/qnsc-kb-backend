"""The deterministic local answer provider: no network, no key, only with GEMINI_MODEL=mock in development."""

import pytest

from src.core.config import settings
from src.domain import llm_client
from src.domain.llm_config import (
    RuntimeLLMConfig,
    load_runtime_config,
    set_runtime_config,
)

REFUSAL = (
    "I could not produce a grounded answer from the authorized Knowledge Base sources."
)

# The shape the answer prompt uses for each retrieved passage.
WITH_CONTEXT = [
    {"role": "system", "content": "Answer from the context."},
    {
        "role": "user",
        "content": (
            '<authorized-document id="C1">\n<title>Leave policy</title>\n'
            "<untrusted-passage>\nThe leave policy is 20 days.\n</untrusted-passage>\n"
            "</authorized-document>\n\nQuestion: how many leave days?"
        ),
    },
]
NO_CONTEXT = [{"role": "user", "content": "Question: what is the moon made of?"}]


@pytest.fixture(autouse=True)
def _no_runtime_config_leaks():
    set_runtime_config(None)
    yield
    set_runtime_config(None)


class _NoRowResult:
    def scalar_one_or_none(self):
        return None


class _NoRowSession:
    async def execute(self, _statement):
        return _NoRowResult()


def test_the_mock_cites_the_first_authorized_passage_and_is_deterministic():
    first = llm_client._mock_completion(WITH_CONTEXT)
    assert first == llm_client._mock_completion(WITH_CONTEXT)
    text, tokens, model, provider = first
    assert "The leave policy is 20 days." in text
    assert text.endswith("[C1]")
    assert text.startswith("[mock answer")
    assert tokens == len(text.split())
    assert (model, provider) == ("mock", "mock")


def test_without_an_authorized_passage_the_mock_refuses_like_a_grounded_model():
    text, _tokens, _model, _provider = llm_client._mock_completion(NO_CONTEXT)
    assert text == REFUSAL


@pytest.mark.asyncio
async def test_complete_answers_with_the_mock_and_streams_it_when_asked():
    set_runtime_config(RuntimeLLMConfig("mock", "mock", "mock://local", "mock"))
    streamed: list[str] = []

    async def on_token(chunk: str) -> None:
        streamed.append(chunk)

    text, _tokens, model, provider = await llm_client.complete(
        WITH_CONTEXT, on_token=on_token
    )
    assert text.endswith("[C1]")
    assert streamed == [text]
    assert (model, provider) == ("mock", "mock")


@pytest.mark.asyncio
async def test_the_mock_connection_test_reaches_no_endpoint():
    provider = llm_client.Provider("mock", "mock", "mock://local", "mock")
    assert await llm_client.probe_provider(provider) == 0


@pytest.mark.asyncio
async def test_the_mock_cannot_read_images():
    set_runtime_config(RuntimeLLMConfig("mock", "mock", "mock://local", "mock"))
    with pytest.raises(RuntimeError, match="does not read images"):
        await llm_client.complete_multimodal("system", "user", [("image/png", b"x")])


@pytest.mark.asyncio
async def test_no_saved_provider_plus_gemini_model_mock_selects_the_mock_in_development(
    monkeypatch,
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "mock")
    await load_runtime_config(_NoRowSession())
    config = llm_client.get_runtime_config()
    assert config is not None and config.provider == "mock"


@pytest.mark.asyncio
async def test_no_saved_provider_and_the_default_model_leaves_ai_unconfigured(
    monkeypatch,
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-flash-lite-latest")
    await load_runtime_config(_NoRowSession())
    assert llm_client.get_runtime_config() is None


@pytest.mark.asyncio
async def test_gemini_model_mock_is_inert_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "mock")
    assert not settings.uses_mock_llm
    await load_runtime_config(_NoRowSession())
    assert llm_client.get_runtime_config() is None
