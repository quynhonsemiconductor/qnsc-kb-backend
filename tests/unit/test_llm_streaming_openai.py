"""Answers stream from every OpenAI-compatible provider, not only GLM.

The reader of the AI assistant waited the whole generation in silence and then got the entire
answer at once: only GLM and Gemini streamed, so OpenAI, Groq and a custom endpoint such as
Anthropic's made one non-streaming call. A non-streaming call is also held to LLM_TIMEOUT_SECONDS
in total, so a long answer that needed more than the cap was lost. These tests drive the real
`complete` against a fake HTTP client and pin what streaming must do.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.core.config import settings
from src.domain import llm_client
from src.domain.llm_client import Provider
from src.domain.llm_endpoint_security import ResolvedLLMEndpoint

MESSAGES = [{"role": "user", "content": "hi"}]


def sse(*events: dict | str) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines.append(
            "data: " + (event if isinstance(event, str) else json.dumps(event))
        )
        lines.append("")
    return lines


def chunk(text: str | None = None, **extra) -> dict:
    delta = {"content": text} if text is not None else {}
    return {"choices": [{"delta": delta}], **extra}


class Recorder:
    def __init__(self):
        self.requests: list[dict] = []
        self.streamed = 0
        self.posted = 0


def install(monkeypatch, *, lines=None, status=200, body_text="", provider=None):
    recorder = Recorder()

    class Response:
        def __init__(self, ok_status):
            self.status_code = ok_status
            self.is_error = ok_status >= 400
            self.is_redirect = False
            self.headers = {}
            self.text = body_text

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def aiter_lines(self):
            for line in lines or []:
                yield line

        async def aread(self):
            raise AssertionError("a guarded provider body must not be read")

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "whole answer"}}],
                "usage": {"total_tokens": 9},
            }

    class Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        def stream(self, method, url, **kwargs):
            recorder.streamed += 1
            recorder.requests.append(kwargs["json"])
            return Response(status)

        async def post(self, url, **kwargs):
            recorder.posted += 1
            recorder.requests.append(kwargs["json"])
            return Response(status)

    async def validate(_url):
        return ResolvedLLMEndpoint("api.anthropic.com", "160.79.104.10")

    monkeypatch.setattr(llm_client, "validate_public_endpoint_resolution", validate)
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", Client)
    chosen = provider or Provider(
        "custom",
        "claude-sonnet-5-5",
        "https://api.anthropic.com/v1/chat/completions",
        "k",
        custom_endpoint=True,
    )
    monkeypatch.setattr(llm_client, "resolve_provider", lambda *_a, **_k: chosen)
    return recorder


def run(coro):
    return asyncio.run(coro)


def test_a_custom_endpoint_streams_tokens_as_they_arrive(monkeypatch):
    recorder = install(
        monkeypatch,
        lines=sse(
            chunk("Dự thảo "),
            chunk("Luật "),
            chunk("gồm hai nhóm."),
            {"choices": [], "usage": {"total_tokens": 21}},
            "[DONE]",
        ),
    )
    seen: list[str] = []

    async def on_token(text):
        seen.append(text)

    text, tokens, model, provider = run(
        llm_client.complete(MESSAGES, on_token=on_token)
    )
    assert seen == [
        "Dự thảo ",
        "Luật ",
        "gồm hai nhóm.",
    ]  # in pieces, in order, not one lump
    assert (
        text == "Dự thảo Luật gồm hai nhóm."
        and tokens == 21
        and model == "claude-sonnet-5-5"
    )
    assert recorder.streamed == 1 and recorder.posted == 0
    assert recorder.requests[0]["stream"] is True


def test_a_call_with_no_token_listener_is_still_one_request(monkeypatch):
    recorder = install(monkeypatch)
    text, *_ = run(llm_client.complete(MESSAGES))
    assert text == "whole answer"
    assert recorder.posted == 1 and recorder.streamed == 0
    assert "stream" not in recorder.requests[0]


def test_the_kill_switch_restores_the_single_request(monkeypatch):
    monkeypatch.setattr(settings, "LLM_STREAM_ANSWERS", False)
    recorder = install(monkeypatch)
    seen: list[str] = []

    async def on_token(text):
        seen.append(text)

    run(llm_client.complete(MESSAGES, on_token=on_token))
    assert recorder.posted == 1 and recorder.streamed == 0
    assert seen == ["whole answer"]


def test_empty_keepalive_and_non_text_events_are_skipped(monkeypatch):
    install(
        monkeypatch,
        lines=sse(
            chunk(), ": keepalive", chunk("ok"), "not json", chunk(None), "[DONE]"
        ),
    )
    seen: list[str] = []

    async def on_token(text):
        seen.append(text)

    text, *_ = run(llm_client.complete(MESSAGES, on_token=on_token))
    assert text == "ok" and seen == ["ok"]


def test_an_official_openai_provider_streams_too(monkeypatch):
    recorder = install(
        monkeypatch,
        lines=sse(chunk("A"), chunk("B"), "[DONE]"),
        provider=Provider(
            "openai", "gpt-4o", "https://api.openai.com/v1/chat/completions", "k"
        ),
    )

    async def on_token(_text):
        return None

    text, *_ = run(llm_client.complete(MESSAGES, on_token=on_token))
    assert text == "AB" and recorder.streamed == 1


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_on_a_stream_is_an_auth_error_and_the_body_is_never_read(
    monkeypatch, status
):
    install(monkeypatch, status=status)

    async def on_token(_text):
        return None

    with pytest.raises(llm_client.ProviderAuthError):
        run(llm_client.complete(MESSAGES, on_token=on_token))


def test_other_stream_errors_on_a_custom_endpoint_name_only_the_status(monkeypatch):
    install(monkeypatch, status=400)

    async def on_token(_text):
        return None

    with pytest.raises(RuntimeError, match="HTTP 400"):
        run(llm_client.complete(MESSAGES, on_token=on_token))


def test_claude_models_still_get_no_temperature_when_streaming(monkeypatch):
    recorder = install(monkeypatch, lines=sse(chunk("x"), "[DONE]"))

    async def on_token(_text):
        return None

    run(llm_client.complete(MESSAGES, on_token=on_token))
    assert "temperature" not in recorder.requests[0]
