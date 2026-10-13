"""Small provider adapter for the text-generation paths in the application.

The active provider is selected by the administrator's persisted workspace
configuration. OpenAI, GLM, and Groq all use their OpenAI-compatible chat API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time
import weakref
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx
import structlog

from src.core.config import settings
from src.core.retry import with_exponential_retry
from src.domain.llm_config import DEFAULT_BASE_URLS, get_runtime_config
from src.domain.llm_endpoint_security import validate_public_endpoint_resolution
from src.rag.answer_sections import GROUNDED_SENTINEL

logger = structlog.get_logger()


class ProviderRateLimitError(RuntimeError):
    """The provider refused the request for quota reasons, not for its content."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderEmptyReplyError(RuntimeError):
    """The provider answered 200 with no text. A failed call that is worth retrying, not an answer."""


class ProviderAuthError(RuntimeError):
    """The provider rejected our credentials.

    Separated from every other provider failure because the fix is somewhere else
    entirely: nobody can repair this by retrying, rephrasing, or re-indexing. It is the
    workspace API key, and the operator needs to be told so.
    """


_RETRY_AFTER_IN_MESSAGE = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _retry_after_seconds(
    response: httpx.Response, *, include_message: bool = True
) -> int | None:
    """Seconds the provider says to wait, from the header or a trusted message."""

    header = response.headers.get("retry-after")
    if header:
        try:
            return max(1, int(float(header)))
        except ValueError:
            pass
    if not include_message:
        return None
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
    custom_endpoint: bool = False


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
        custom_endpoint=(
            config.provider == "custom"
            or config.base_url != DEFAULT_BASE_URLS.get(config.provider)
        ),
    )


async def _guarded_request_target(
    provider: Provider, url: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Pin a custom endpoint to the public IP we validated, preserving TLS identity."""

    if not provider.custom_endpoint:
        return url, {}, {}
    resolved = await validate_public_endpoint_resolution(provider.url)
    parsed = urlsplit(url)
    if (parsed.hostname or "").lower().rstrip(".") != resolved.hostname:
        raise RuntimeError("The provider request host changed after endpoint validation")
    address = f"[{resolved.address}]" if ":" in resolved.address else resolved.address
    # The port has to survive the rewrite. Dropping it was invisible while every custom
    # endpoint was public HTTPS on the default 443, but a locally hosted model listens
    # elsewhere (11434 for Ollama, 8000 for vLLM) and the pinned URL would then reach
    # port 80 at the right address.
    authority = f"{address}:{parsed.port}" if parsed.port else address
    pinned_url = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))
    host_header = parsed.netloc.split("@", 1)[-1] or resolved.hostname
    return (
        pinned_url,
        {"Host": host_header},
        {"sni_hostname": resolved.hostname},
    )


async def probe_provider(provider: Provider, timeout: float = 15.0) -> int:
    """Test one exact candidate without changing the active runtime configuration."""
    url = _gemini_url(provider, streaming=False) if provider.native_gemini else provider.url
    request_url, host_headers, extensions = await _guarded_request_target(
        provider, url
    )
    payload = _payload(
        provider,
        [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "Connection test"},
        ],
        0.0,
        max_tokens=8,
    )
    started = perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=not provider.custom_endpoint,
        ) as client:
            async with client.stream(
                "POST",
                request_url,
                headers={**_headers(provider), **host_headers},
                json=payload,
                extensions=extensions,
            ) as response:
                latency_ms = max(1, round((perf_counter() - started) * 1000))
                if getattr(response, "is_redirect", False):
                    raise RuntimeError(
                        "The provider redirected the request; use its final HTTPS Chat Completions endpoint"
                    )
                if response.status_code in (401, 403):
                    raise ProviderAuthError("The provider rejected the configured API key")
                if response.status_code == 429:
                    raise ProviderRateLimitError(
                        "The provider rate limited the connection test"
                    )
                if response.is_error:
                    raise RuntimeError(
                        f"The provider returned HTTP {response.status_code}"
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 262_144:
                        raise RuntimeError("The provider response was unexpectedly large")
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "The provider did not respond before the connection test timed out"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("The provider could not be reached") from exc
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise TypeError
        answer = (
            _extract_gemini_text(data)
            if provider.native_gemini
            else _extract_openai_text(data)
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        # Not JSON: some OpenAI-compatible gateways answer `text/event-stream` for every
        # request, including one that never asked to stream. The streaming path already
        # reads that shape, so a connection test must accept it too — otherwise an
        # endpoint the workspace can genuinely use cannot be saved, because saving
        # requires a successful test.
        text, reasoning = _extract_sse_text(body.decode("utf-8", "replace"))
        answer = text or _answer_from_reasoning(reasoning)
        if not answer.strip():
            raise RuntimeError(
                "The provider did not return a valid Chat Completions response"
            ) from exc
    if not answer.strip():
        raise RuntimeError("The provider response did not contain assistant text")
    return latency_ms



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
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "thinkingConfig": {"thinkingLevel": settings.GEMINI_THINKING_LEVEL},
        }
        # Omitted entirely when the caller passes None, same as every other provider
        # below: `maxOutputTokens` is optional in Gemini's own API, and a caller that
        # explicitly asked for no cap (e.g. RAG_MAX_ANSWER_TOKENS=None) should get the
        # model's own maximum, not a second, Gemini-only ceiling nothing else in this
        # file has. The runaway-generation risk a cap like this usually guards against
        # is thinking tokens silently eating the budget -- already addressed here by
        # GEMINI_THINKING_LEVEL="minimal" above, the same fix GLM needed for the same
        # failure mode (see the `thinking` flag below).
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        return payload
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
    }
    if thinking is not None and provider.name == "glm":
        # `level`, never `{"type": "disabled"}`. Newer GLM models cannot turn thinking
        # off and reject that payload outright with HTTP 400 code 1210 ("This model
        # always engages in thinking and cannot be disabled; please use low, high, or
        # max"). Because `thinking=False` is what every non-chat caller passes --
        # restructuring, department routing, auto-tagging, the approval agent -- the
        # rejection hit each one: document preparation failed EVERY section and always
        # fell back to the lossless reading view, so an upload could never reach review
        # as AI-formatted. `level` is accepted by both the current and the previous
        # generation, so "as little deliberation as the model allows" replaces "off".
        payload["thinking"] = {"level": "high" if thinking else "low"}
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


def _extract_sse_text(body: str) -> tuple[str, str]:
    """Collect `(content, reasoning)` from a Chat Completions SSE stream.

    Some OpenAI-compatible gateways answer `text/event-stream` for EVERY request, even
    one that never asked to stream. The body is then a sequence of `data:` chunks rather
    than a single JSON object, so a plain `json.loads` sees nothing parseable at all.
    """
    content = ""
    reasoning = ""
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        # `delta` while streaming, `message` for a gateway that streams one whole
        # non-streaming response as a single event.
        part = choices[0].get("delta") or choices[0].get("message") or {}
        content += str(part.get("content") or "")
        reasoning += str(part.get("reasoning_content") or part.get("reasoning") or "")
    return content, reasoning


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


# ── rate-limit management ────────────────────────────────────────────────────────────────
# Gates are per event loop: a Celery task runs each coroutine on a fresh loop (`sync_run`), and an asyncio
# primitive created on one loop cannot be awaited on another.
_gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()
# time.monotonic() before which the provider has asked not to be called again (process-wide, any loop).
_rate_limited_until = 0.0


def _gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = asyncio.Semaphore(max(1, settings.LLM_MAX_CONCURRENCY))
    return gate


def _note_rate_limit(retry_after: int | None) -> float:
    """Remember the provider's wait so the next caller does not spend a request finding out."""
    global _rate_limited_until
    wait = float(retry_after or 5)
    _rate_limited_until = max(_rate_limited_until, time.monotonic() + wait)
    return wait


async def complete(
    messages: list[dict[str, str]],
    *,
    model_override: str | None = None,
    timeout: float = 30.0,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    background: bool = False,
) -> tuple[str, int, str, str]:
    """Generate text and return ``(text, token_count, model, provider)``.

    TWO CALLING MODES, and the difference matters:

    * ``background=True`` is for work nobody is waiting on a screen for (document restructuring, tag
      suggestion in the worker). It is capped at LLM_MAX_CONCURRENCY calls at once per process, and a 429
      with a short Retry-After is waited out and retried, because waiting costs nothing there and a burst
      would otherwise draw a 429 for every call over the limit.
    * the default is INTERACTIVE (chat, streamed answers, multi-query, judging, routing on a request). It is
      NOT gated: a concurrency cap here would make the third simultaneous chat user queue behind two streams
      of up to LLM_TIMEOUT_SECONDS each, for every provider, rate limited or not. It never sleeps on a 429
      either: it raises ProviderRateLimitError at once so the caller can tell the user when to retry.

    Both modes: while the provider has asked not to be called (a recent 429), a call fails fast with the
    remaining wait instead of spending a request to find out (a background call waits it out when it is
    short), and an empty reply is a failed call, retried and then raised as ProviderEmptyReplyError.
    """
    rate_retries = 0
    empty_retries = 0
    while True:
        remaining = _rate_limited_until - time.monotonic()
        if remaining > 0:
            if not background or remaining > settings.LLM_RATE_LIMIT_INLINE_WAIT_SECONDS:
                raise ProviderRateLimitError(
                    "The provider asked us to wait; not calling it again yet", retry_after=int(remaining) + 1
                )
            await asyncio.sleep(remaining + random.uniform(0, 1))
        pause: float | None = None
        async with (_gate() if background else _NoGate()):
            try:
                result = await _complete_once(
                    messages,
                    model_override=model_override,
                    timeout=timeout,
                    thinking=thinking,
                    max_tokens=max_tokens,
                    on_token=on_token,
                )
            except ProviderRateLimitError as exc:
                wait = _note_rate_limit(exc.retry_after)
                if (
                    not background
                    or wait > settings.LLM_RATE_LIMIT_INLINE_WAIT_SECONDS
                    or rate_retries >= settings.LLM_RATE_LIMIT_MAX_INLINE_RETRIES
                ):
                    logger.warning("LLM rate limited; giving up on this call", retry_after=wait, retries=rate_retries)
                    raise
                rate_retries += 1
                pause = wait + random.uniform(0, 1)
                logger.info("LLM rate limited; waiting and retrying", retry_after=wait, attempt=rate_retries)
            else:
                if result[0].strip():
                    return result
                if empty_retries >= settings.LLM_EMPTY_REPLY_RETRIES:
                    raise ProviderEmptyReplyError(f"{result[3]} returned an empty reply {empty_retries + 1} times")
                empty_retries += 1
                pause = 1.0 * empty_retries
                logger.warning("LLM returned an empty reply; retrying", attempt=empty_retries)
        # The gate is released while sleeping, so a waiting call does not hold a slot.
        await asyncio.sleep(pause or 0)


class _NoGate:
    """The interactive path: same `async with` shape, no limit."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> None:
        return None


async def _complete_once(
    messages: list[dict[str, str]],
    *,
    model_override: str | None = None,
    timeout: float = 30.0,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, int, str, str]:
    """One provider request, no retry policy. Use ``complete``."""
    provider = resolve_provider(model_override)
    if provider is None:
        raise RuntimeError("No LLM provider is configured.")

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=not provider.custom_endpoint,
    ) as client:
        if provider.native_gemini and on_token:
            url = _gemini_url(provider, streaming=True)
            request_url, host_headers, extensions = await _guarded_request_target(
                provider, url
            )
            answer = ""
            # Bound before the loop, because the loop may never assign it: a stream
            # that yields no parseable `data:` line leaves it unset. This used to be
            # read back with `locals().get("tokens", 0)`, which works but hides the
            # fact that the name can be unbound and reads as dynamic-scope trickery.
            tokens = 0
            async with client.stream(
                "POST",
                request_url,
                headers={**_headers(provider), **host_headers},
                json=_payload(
                    provider, messages, 0.0, thinking=thinking, max_tokens=max_tokens
                ),
                extensions=extensions,
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
            return answer, tokens, provider.model, provider.name

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
            request_url, host_headers, extensions = await _guarded_request_target(
                provider, provider.url
            )

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
                    request_url,
                    headers={
                        **_headers(provider),
                        **host_headers,
                        "Accept": "text/event-stream",
                    },
                    json=payload,
                    extensions=extensions,
                ) as response:
                    if getattr(response, "is_redirect", False):
                        raise RuntimeError(
                            "The provider redirected the request; configure its final HTTPS endpoint"
                        )
                    if response.is_error:
                        # A guarded endpoint is administrator-controlled and its response
                        # is untrusted. Do not read or echo its body: besides leaking a
                        # provider diagnostic, ``aread`` could buffer an unbounded body.
                        if provider.custom_endpoint:
                            if response.status_code in (401, 403):
                                raise ProviderAuthError(
                                    "The provider rejected the configured API key"
                                )
                            if response.status_code == 429:
                                raise ProviderRateLimitError(
                                    "The provider rate limit was reached",
                                    retry_after=_retry_after_seconds(
                                        response, include_message=False
                                    ),
                                )
                            raise RuntimeError(
                                f"The provider returned HTTP {response.status_code}"
                            )
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
        request_url, host_headers, extensions = await _guarded_request_target(
            provider, url
        )

        async def request_completion() -> httpx.Response:
            response = await client.post(
                request_url,
                headers={**_headers(provider), **host_headers},
                json=payload,
                extensions=extensions,
            )
            if getattr(response, "is_redirect", False):
                raise RuntimeError(
                    "The provider redirected the request; configure its final HTTPS endpoint"
                )
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
                if provider.custom_endpoint:
                    raise RuntimeError(
                        f"The provider returned HTTP {response.status_code}"
                    ) from exc
                detail = response.text[:500].replace("\n", " ")
                raise RuntimeError(
                    f"{provider.name} request failed with HTTP {response.status_code}: {detail}"
                ) from exc
            return response

        # Formatting is optional. A single bounded request keeps the review
        # screen responsive; content_restructure will preserve the source with
        # a local lossless fallback when the provider is slow or unavailable.
        response = await with_exponential_retry(request_completion, attempts=1)
        try:
            data = response.json()
        except ValueError:
            # Not JSON. An OpenAI-compatible gateway may answer `text/event-stream` for
            # EVERY request, including this one, which never asked to stream. Reading it
            # as JSON failed with "Expecting value: line 1 column 1" and surfaced to the
            # user as "AI generation failed", even though the provider had answered
            # correctly. `probe_provider` accepts the same shape, so a provider that
            # passed its connection test must not then fail every real call.
            text, reasoning = _extract_sse_text(response.text)
            answer = text or _answer_from_reasoning(reasoning)
            if not answer.strip():
                raise RuntimeError(
                    f"{provider.name} returned no readable assistant text"
                )
            if on_token:
                await on_token(answer)
            return answer, 0, provider.model, provider.name
        answer = (
            _extract_gemini_text(data)
            if provider.native_gemini
            else _extract_openai_text(data)
        )
        if on_token:
            await on_token(answer)
        return answer, _extract_usage(data), provider.model, provider.name


async def complete_multimodal(
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, bytes]],
    *,
    timeout: float = 30.0,
    max_tokens: int | None = None,
) -> tuple[str, int, str, str]:
    """Complete against bounded, server-selected images; never accepts client URLs."""
    # Validate caller-controlled shape before configuration. An unsupported media type is
    # invalid regardless of whether this deployment happens to have a provider configured,
    # and resolving credentials first made this guard impossible to test or reason about.
    if not images or len(images) > settings.EVIDENCE_VISION_MAX_IMAGES:
        raise ValueError("invalid number of evidence images")
    allowed_types = {"image/png", "image/jpeg", "image/webp"}
    if any(mime not in allowed_types for mime, _data in images):
        raise ValueError("unsupported evidence image type")
    if sum(len(data) for _mime, data in images) > settings.EVIDENCE_VISION_MAX_TOTAL_BYTES:
        raise ValueError("evidence images exceed the request size limit")
    provider = resolve_provider()
    if provider is None:
        raise RuntimeError("No LLM provider is configured.")

    if provider.native_gemini:
        parts: list[dict[str, Any]] = []
        for mime, data in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
            )
        parts.append({"text": user_prompt})
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "thinkingConfig": {"thinkingLevel": settings.GEMINI_THINKING_LEVEL},
            },
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        url = _gemini_url(provider, streaming=False)
    else:
        content: list[dict[str, Any]] = []
        for mime, data in images:
            encoded = base64.b64encode(data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": user_prompt})
        payload = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        url = provider.url

    request_url, host_headers, extensions = await _guarded_request_target(
        provider, url
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=not provider.custom_endpoint,
    ) as client:
        response = await client.post(
            request_url,
            headers={**_headers(provider), **host_headers},
            json=payload,
            extensions=extensions,
        )
        if getattr(response, "is_redirect", False):
            raise RuntimeError(
                "The provider redirected the request; configure its final HTTPS endpoint"
            )
        if response.status_code in (400, 404, 415, 422):
            raise RuntimeError("The configured provider does not support evidence images")
        if response.status_code in (401, 403):
            raise ProviderAuthError("The provider rejected the configured API key")
        if response.status_code == 429:
            raise ProviderRateLimitError(
                "The provider rate limited the vision request",
                retry_after=_retry_after_seconds(response),
            )
        response.raise_for_status()
        result = response.json()
    answer = _extract_gemini_text(result) if provider.native_gemini else _extract_openai_text(result)
    return answer, _extract_usage(result), provider.model, provider.name
