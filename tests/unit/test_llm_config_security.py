import pytest
from fastapi import HTTPException

from src.api.routers.llm import _clean_base_url
from src.domain.llm_config import DEFAULT_BASE_URLS


def test_llm_endpoint_uses_the_provider_default_when_omitted():
    assert _clean_base_url("openai", None) == DEFAULT_BASE_URLS["openai"]
    assert DEFAULT_BASE_URLS["glm"] == "https://api.z.ai/api/coding/paas/v4/chat/completions"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openai.com/v1/chat/completions",
        "https://localhost/v1/chat/completions",
        "https://127.0.0.1/v1/chat/completions",
        "https://user:password@api.openai.com/v1/chat/completions",
        "https://api.openai.com:8443/v1/chat/completions",
    ],
)
def test_llm_endpoint_rejects_unsafe_custom_urls(url):
    with pytest.raises(HTTPException, match="Custom LLM endpoints are disabled"):
        _clean_base_url("openai", url)


def test_llm_endpoint_rejects_unsafe_url_even_when_custom_endpoints_are_enabled(monkeypatch):
    monkeypatch.setattr("src.api.routers.llm.settings.LLM_ALLOW_CUSTOM_BASE_URL", True)
    with pytest.raises(HTTPException, match="private or reserved"):
        _clean_base_url("openai", "https://127.0.0.1/v1/chat/completions")
    with pytest.raises(HTTPException, match="credential-free HTTPS"):
        _clean_base_url("openai", "http://public.example/v1/chat/completions")
