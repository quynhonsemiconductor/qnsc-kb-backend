"""Small provider adapter for the text-generation paths in the application.

The active provider is selected by the administrator's persisted workspace
configuration. OpenAI, GLM, and Groq all use their OpenAI-compatible chat API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from src.core.config import settings
from src.core.retry import with_exponential_retry
from src.domain.llm_config import get_runtime_config


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    url: str
    api_key: str
    native_gemini: bool = False


def resolve_provider(model_override: str | None = None) -> Provider | None:
    config = get_runtime_config()
    if config is None:
        return None
    # Gemini is not OpenAI-compatible — different request body, different auth header,
    # different SSE framing, different response shape. All of that is implemented below
    # and was unreachable: native_gemini defaulted to False and nothing ever set it, so
    # the flag existed while the path it guards could not run.
    return Provider(
        config.provider,
        model_override or config.model,
        config.base_url,
        config.api_key,
        native_gemini=config.provider == "gemini",
    )


def _gemini_contents(
    messages: list[dict[str, str]]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_message = next(
        (item for item in messages if item.get("role") == "system"), None
    )
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": message.get("content", "")}],
            }
        )
    return (
        (
            {"parts": [{"text": system_message.get("content", "")}]}
            if system_message
            else None
        ),
        contents,
    )


def _payload(
    provider: Provider,
    messages: list[dict[str, str]],
    temperature: float,
    *,
    thinking: bool | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if provider.native_gemini:
        system_instruction, contents = _gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": (
                    max_tokens
                    if max_tokens is not None
                    else settings.GEMINI_MAX_OUTPUT_TOKENS
                ),
                "thinkingConfig": {"thinkingLevel": settings.GEMINI_THINKING_LEVEL},
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        return payload
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
    }
    if thinking is not None and provider.name == "glm":
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _headers(provider: Provider) -> dict[str, str]:
    if provider.native_gemini:
        return {"x-goog-api-key": provider.api_key, "Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }


def _gemini_url(provider: Provider, streaming: bool) -> str:
    action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    return f"{provider.url}/models/{provider.model}:{action}"


def _extract_gemini_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            # Gemma 4 may return internal thought parts alongside the final
            # answer. Never expose or persist those as document content.
            if isinstance(part, dict) and not part.get("thought") and part.get("text"):
                parts.append(str(part["text"]))
    return "".join(parts)


def _extract_openai_text(data: dict[str, Any]) -> str:
    return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))


def _extract_usage(data: dict[str, Any]) -> int:
    usage = data.get("usageMetadata") or data.get("usage") or {}
    return int(usage.get("totalTokenCount") or usage.get("total_tokens") or 0)


async def complete(
    messages: list[dict[str, str]],
    *,
    model_override: str | None = None,
    timeout: float = 30.0,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, int, str, str]:
    """Generate text and return ``(text, token_count, model, provider)``."""
    provider = resolve_provider(model_override)
    if provider is None:
        raise RuntimeError("No LLM provider is configured.")

    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider.native_gemini and on_token:
            url = _gemini_url(provider, streaming=True)
            answer = ""
            async with client.stream(
                "POST",
                url,
                headers=_headers(provider),
                json=_payload(
                    provider, messages, 0.0, thinking=thinking, max_tokens=max_tokens
                ),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = _extract_gemini_text(data)
                    if text:
                        answer += text
                        await on_token(text)
                    tokens = _extract_usage(data)
            return answer, locals().get("tokens", 0), provider.model, provider.name

        if provider.name == "glm":
            # GLM's synchronous endpoint returns only after it has completed
            # both its internal reasoning and final content. For a lossless
            # document rewrite that can leave an otherwise healthy connection
            # idle until httpx raises ReadTimeout. Use its OpenAI-compatible
            # SSE format so each received event keeps the read timeout alive.
            payload = _payload(
                provider, messages, 0.0, thinking=thinking, max_tokens=max_tokens
            )
            payload["stream"] = True

            async def request_stream() -> tuple[str, int]:
                answer = ""
                tokens = 0
                async with client.stream(
                    "POST",
                    provider.url,
                    headers={**_headers(provider), "Accept": "text/event-stream"},
                    json=payload,
                ) as response:
                    if response.is_error:
                        detail = (
                            (await response.aread())
                            .decode(errors="replace")[:500]
                            .replace("\n", " ")
                        )
                        raise RuntimeError(
                            f"{provider.name} request failed with HTTP "
                            f"{response.status_code}: {detail}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        delta = choices[0].get("delta") if choices else None
                        text = str((delta or {}).get("content") or "")
                        if text:
                            answer += text
                            if on_token:
                                await on_token(text)
                        tokens = _extract_usage(data) or tokens
                return answer, tokens

            # A new request is safe here because a failed stream is discarded;
            # no partial document is ever persisted. One retry handles brief
            # provider queueing or network interruptions without blocking
            # forever behind a slow model.
            answer, tokens = await with_exponential_retry(
                request_stream, attempts=2, base_delay=1.0
            )
            return answer, tokens, provider.model, provider.name

        url = (
            _gemini_url(provider, streaming=False)
            if provider.native_gemini
            else provider.url
        )
        payload = _payload(
            provider, messages, 0.0, thinking=thinking, max_tokens=max_tokens
        )

        async def request_completion() -> httpx.Response:
            response = await client.post(url, headers=_headers(provider), json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:500].replace("\n", " ")
                raise RuntimeError(
                    f"{provider.name} request failed with HTTP {response.status_code}: {detail}"
                ) from exc
            return response

        # Formatting is optional. A single bounded request keeps the review
        # screen responsive; content_restructure will preserve the source with
        # a local lossless fallback when the provider is slow or unavailable.
        response = await with_exponential_retry(request_completion, attempts=1)
        data = response.json()
        answer = (
            _extract_gemini_text(data)
            if provider.native_gemini
            else _extract_openai_text(data)
        )
        if on_token:
            await on_token(answer)
        return answer, _extract_usage(data), provider.model, provider.name
