"""The deterministic local answer provider: no network, no key. Only with LLM_PROVIDER=mock in development."""

import pytest

from src.core.config import Settings, settings
from src.domain import llm_client
from src.domain.llm_config import RuntimeLLMConfig, load_runtime_config, public_config, set_runtime_config

REFUSAL = "I could not produce a grounded answer from the authorized Knowledge Base sources."
SENTINEL = "<<<GROUNDED>>>"

# The shape of the grounded-answer prompt: the sentinel instruction in the system prompt, one authorized passage in the user turn.
ANSWER_WITH_CONTEXT = [
    {"role": "system", "content": f"Emit the literal sentinel line `{SENTINEL}` on its own line, then the grounded answer."},
    {
        "role": "user",
        "content": (
            '<authorized-document id="C1">\n<title>Leave policy</title>\n'
            "<untrusted-passage>\nThe leave policy is 20 days.\n</untrusted-passage>\n"
            "</authorized-document>\n\nQuestion: how many leave days?"
        ),
    },
]
ANSWER_NO_CONTEXT = [
    {"role": "system", "content": f"Emit the literal sentinel line `{SENTINEL}` on its own line, then the grounded answer."},
    {"role": "user", "content": "Question: what is the moon made of?"},
]
# Any other call, such as a contextual chunk header or a query rewrite.
OTHER_CALL = [
    {"role": "system", "content": "Write a one-line context header for this section."},
    {"role": "user", "content": "Section: Leave policy. Question style: none."},
]


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


def test_the_answer_prompt_gets_a_deterministic_citation_from_the_first_authorized_passage():
    first = llm_client._mock_completion(ANSWER_WITH_CONTEXT)
    assert first == llm_client._mock_completion(ANSWER_WITH_CONTEXT)
    text, tokens, model, provider = first
    assert text.startswith(SENTINEL + "\n")
    assert "The leave policy is 20 days." in text
    assert text.endswith("[C1]")
    assert tokens == len(text.split())
    assert (model, provider) == ("mock", "mock")


def test_the_answer_prompt_without_a_passage_gets_the_refusal_after_the_sentinel():
    text, _tokens, _model, _provider = llm_client._mock_completion(ANSWER_NO_CONTEXT)
    assert text == f"{SENTINEL}\n{REFUSAL}"


def test_other_calls_get_a_neutral_stub_never_the_refusal():
    text, _tokens, _model, _provider = llm_client._mock_completion(OTHER_CALL)
    assert text == "[mock: no answer for this call]"
    assert REFUSAL not in text


@pytest.mark.asyncio
async def test_complete_answers_with_the_mock_and_streams_it_when_asked():
    set_runtime_config(RuntimeLLMConfig("mock", "mock", "mock://local", "mock"))
    streamed: list[str] = []

    async def on_token(chunk: str) -> None:
        streamed.append(chunk)

    text, _tokens, model, provider = await llm_client.complete(ANSWER_WITH_CONTEXT, on_token=on_token)
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
async def test_llm_provider_mock_selects_the_mock_in_development_with_no_saved_provider(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    await load_runtime_config(_NoRowSession())
    config = llm_client.get_runtime_config()
    assert config is not None and config.provider == "mock"


@pytest.mark.asyncio
async def test_no_switch_leaves_ai_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "")
    await load_runtime_config(_NoRowSession())
    assert llm_client.get_runtime_config() is None


@pytest.mark.asyncio
async def test_llm_provider_mock_is_inert_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    assert not settings.uses_mock_llm
    await load_runtime_config(_NoRowSession())
    assert llm_client.get_runtime_config() is None


@pytest.mark.asyncio
async def test_the_admin_screen_reports_the_mock_as_an_environment_source(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    await load_runtime_config(_NoRowSession())
    assert public_config(None)["source"] == "env (mock)"


def test_production_refuses_the_mock_llm(monkeypatch):
    monkeypatch.setattr(Settings, "is_development_environment", property(lambda self: False))
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER=mock is for local development only"):
        settings.validate_production()
