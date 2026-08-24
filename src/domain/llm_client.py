"""Small provider adapter for the text-generation paths in the application.

The active provider is selected by the administrator's persisted workspace
configuration. OpenAI, GLM, and Groq all use their OpenAI-compatible chat API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
import structlog

from src.core.config import settings
from src.core.retry import with_exponential_retry
from src.domain.llm_config import get_runtime_config
from src.rag.answer_sections import GROUNDED_SENTINEL

logger = structlog.get_logger()


class ProviderRateLimitError(RuntimeError):
    """The provider refused the request for quota reasons, not for its content."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderAuthError(RuntimeError):
    """The provider rejected our credentials.

    Separated from every other provider failure because the fix is somewhere else
    entirely: nobody can repair this by retrying, rephrasing, or re-indexing. It is the
    workspace API key, and the operator needs to be told so.
    """


_RETRY_AFTER_IN_MESSAGE = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _retry_after_seconds(response: httpx.Response) -> int | None:
    """Seconds the provider says to wait, from the header or from its message."""

    header = response.headers.get("retry-after")
    if header:
        try:
            return max(1, int(float(header)))
        except ValueError:
            pass
    # Groq puts the wait only in the error body: "Please try again in 27.17s".
    match = _RETRY_AFTER_IN_MESSAGE.search(response.text[:1000])
    return max(1, int(float(match.group(1)))) if match else None


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
    """Read the assistant text, including the case where it lands in `reasoning`.

    A reasoning model does not always close its turn with a final channel. Observed on
    Groq's `openai/gpt-oss-120b`: `finish_reason: stop`, 659 reasoning tokens, a fully
    written answer — sentinels, citation and all — inside `message.reasoning`, and
    `message.content` an empty string. Reading only `content` turned that into a blank
    answer, and downstream the blank answer became "I could not produce a grounded
    answer from the authorized Knowledge Base sources": a grounding error reported for
    a response that was never read.

    The fallback keeps only what follows our own `<<<GROUNDED>>>` sentinel, because
    everything before it is deliberation ("We need to produce answer in Vietnamese…")
    and must never reach a user. With no sentinel there is no answer in there to
    recover, so the empty string stands and the caller can retry.
    """

    message = data.get("choices", [{}])[0].get("message", {}) or {}
    content = str(message.get("content") or "")
    if content.strip():
        return content
    # `reasoning` is Groq's field name; `reasoning_content` is GLM's and DeepSeek's.
    thinking = str(message.get("reasoning") or message.get("reasoning_content") or "")
    return _answer_from_reasoning(thinking) or content


def _answer_from_reasoning(thinking: str) -> str:
    """Pull a finished answer out of a reasoning channel, or return nothing.

    Only the text from our own `<<<GROUNDED>>>` sentinel onward is an answer. Whatever
    precedes it is deliberation and must never be shown, so no sentinel means nothing
    to recover — better an empty answer the caller can handle than a user reading the
    model talk to itself.
    """

    marker = (thinking or "").find(GROUNDED_SENTINEL)
    if marker == -1:
        return ""
    logger.warning(
        "Recovered an answer the provider left in its reasoning channel",
        reasoning_length=len(thinking),
        recovered_length=len(thinking) - marker,
    )
    return thinking[marker:]


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
                # Held separately and never streamed out: glm-4.5-flash returns its
                # deliberation in `reasoning_content` and, like other reasoning models,
                # can finish the answer there without ever opening a content delta.
                # Reading only `content` turned that into an empty answer, which the RAG
                # citation guard then reported as "no grounded sources".
                thinking = ""
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
                        # Classified here as well as on the non-streaming path: GLM is
                        # the only provider that streams, so an unusable key or an
                        # exhausted quota reached the user as a generic failure, and
                        # with_exponential_retry below then retried it for nothing.
                        if response.status_code in (401, 403):
                            raise ProviderAuthError(
                                f"{provider.name} rejected the configured API key "
                                f"(HTTP {response.status_code}): {detail}"
                            )
                        if response.status_code == 429:
                            raise ProviderRateLimitError(
                                f"{provider.name} rate limit reached",
                                retry_after=_retry_after_seconds(response),
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
                        else:
                            thinking += str(
                                (delta or {}).get("reasoning_content")
                                or (delta or {}).get("reasoning")
                                or ""
                            )
                        tokens = _extract_usage(data) or tokens
                if not answer.strip():
                    recovered = _answer_from_reasoning(thinking)
                    if recovered and on_token:
                        await on_token(recovered)
                    answer = recovered or answer
                return answer, tokens

            # A new request is safe here because a failed stream is discarded;
            # no partial document is ever persisted. One retry handles brief
            # provider queueing or network interruptions without blocking
            # forever behind a slow model.
            answer, tokens = await with_exponential_retry(
                request_stream,
                attempts=2,
                base_delay=1.0,
                give_up_on=(ProviderAuthError, ProviderRateLimitError),
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
            if response.status_code == 429:
                # Distinct from any other provider failure: nothing is wrong with the
                # request, the tenant is simply over its quota for the moment. Reported
                # as itself so the caller can say when to try again instead of showing
                # the "generation failed" screen for what is a wait.
                raise ProviderRateLimitError(
                    f"{provider.name} rate limit reached",
                    retry_after=_retry_after_seconds(response),
                )
            if response.status_code in (401, 403):
                raise ProviderAuthError(
                    f"{provider.name} rejected the configured API key "
                    f"(HTTP {response.status_code})"
                )
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
