"""Gemini must be reachable as a text-generation provider, not just as an embedder.

The Gemini transport in llm_client.py — request body, x-goog-api-key header, SSE framing,
candidate/thought extraction — was fully implemented and completely unreachable:
`native_gemini` defaulted to False and nothing ever set it, and Gemini was not a
selectable provider. A workspace configured for Gemini therefore used it for embeddings,
resolved NO chat provider, and answered every question with a canned extract of the top
chunk. That fallback reads like a real answer, complete with citations, which is why it
went unnoticed.

The provider itself comes from the admin-managed, encrypted workspace configuration —
see test_llm_provider_source.py. Process-environment keys are deliberately not a source.
"""
from __future__ import annotations

import pytest

from src.domain import llm_config
from src.domain.llm_client import resolve_provider
from src.domain.llm_config import (
    DEFAULT_BASE_URLS,
    SUPPORTED_PROVIDERS,
    RuntimeLLMConfig,
    set_runtime_config,
)


@pytest.fixture(autouse=True)
def _reset_runtime_config():
    yield
    llm_config._runtime_config = None
    llm_config._runtime_config_loaded = False


def test_gemini_is_a_supported_provider():
    assert "gemini" in SUPPORTED_PROVIDERS


def test_gemini_default_url_is_the_api_root_not_a_chat_path():
    # llm_client._gemini_url appends "/models/<model>:generateContent", so a full path
    # here would 404 on every call.
    assert DEFAULT_BASE_URLS["gemini"].endswith("/v1beta")
    assert "generateContent" not in DEFAULT_BASE_URLS["gemini"]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("gemini", True), ("openai", False), ("groq", False), ("glm", False)],
)
def test_native_gemini_is_set_from_the_provider(provider: str, expected: bool):
    set_runtime_config(RuntimeLLMConfig(provider, "some-model", "https://example.test", "key"))

    resolved = resolve_provider()

    assert resolved is not None
    assert resolved.native_gemini is expected, (
        "native_gemini selects an entirely different request body, auth header and "
        "response shape; deriving it wrongly silently sends OpenAI JSON to Gemini"
    )
