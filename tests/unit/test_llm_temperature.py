"""Claude's newer models reject `temperature`, so the app must not send it to them.

Anthropic answers HTTP 400 "`temperature` is deprecated for this model" to any request that
carries one, and the connection test sends one (so does every other call). Other providers
still need it: it is how the deterministic paths ask for the same answer twice.
"""

from __future__ import annotations

import pytest

from src.domain.llm_client import Provider, _accepts_temperature, _payload

URL = "https://api.anthropic.com/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "hi"}]


def provider(model: str, **kwargs) -> Provider:
    return Provider("custom", model, URL, "key", custom_endpoint=True, **kwargs)


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5-5", "claude-opus-5-5", "Claude-Haiku-5-5", " claude-3-5-sonnet "],
)
def test_claude_models_get_no_temperature(model):
    body = _payload(provider(model), MESSAGES, 0.0, max_tokens=8)
    assert "temperature" not in body
    assert body["model"] == model
    assert body["max_tokens"] == 8 and body["messages"] == MESSAGES


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "glm-4.6", "llama-3.3-70b", "my-claude-finetune", "gemini-2.5-flash"],
)
def test_other_models_still_get_it(model):
    assert _payload(provider(model), MESSAGES, 0.3)["temperature"] == 0.3


def test_the_check_is_on_the_model_name_not_the_host():
    gateway = Provider(
        "custom",
        "claude-sonnet-5-5",
        "https://gateway.example.com/v1/chat/completions",
        "k",
        custom_endpoint=True,
    )
    assert _accepts_temperature(gateway) is False
    assert (
        _accepts_temperature(
            Provider(
                "openai", "gpt-4o", "https://api.openai.com/v1/chat/completions", "k"
            )
        )
        is True
    )


def test_native_gemini_is_untouched():
    gemini = Provider(
        "gemini",
        "gemini-2.5-flash",
        "https://generativelanguage.googleapis.com/v1beta",
        "k",
        native_gemini=True,
    )
    assert _payload(gemini, MESSAGES, 0.2)["generationConfig"]["temperature"] == 0.2
