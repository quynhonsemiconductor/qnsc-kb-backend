"""Persisted and runtime configuration for the workspace LLM provider."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.secrets import decrypt_secret, encrypt_secret
from src.models.ops import LLMProviderConfig

SUPPORTED_PROVIDERS = {"openai", "glm", "groq", "gemini"}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "glm": "https://api.z.ai/api/coding/paas/v4/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    # NOT a chat-completions endpoint like the others: the Gemini transport in
    # llm_client.py appends "/models/<model>:generateContent" itself, so this is the API
    # ROOT. Pointing it at a full path produces a 404 on every call.
    "gemini": settings.GEMINI_API_BASE_URL.rstrip("/"),
}


@dataclass(frozen=True)
class RuntimeLLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str


_runtime_config: RuntimeLLMConfig | None = None
_runtime_config_loaded = False


def normalize_api_key(api_key: str | None) -> str | None:
    """Drop every whitespace character from a provider key.

    No provider issues a key containing whitespace, but a key copied out of a console
    or a PDF arrives with one: a trailing newline, or a space where the copy wrapped.
    A Zhipu key is `{32 hex}.{16 alnum}`, and one stored as `{32 hex}. {16 alnum}` is
    rejected with `{"code":"1000","message":"Authentication Failed"}` — a message that
    points at the key's validity, not at the space in the middle of it. Stripping is
    applied on write AND on read so a key already saved that way starts working without
    anyone having to notice the space.
    """

    if api_key is None:
        return None
    return "".join(api_key.split())


def encrypt_api_key(api_key: str) -> str:
    encrypted = encrypt_secret(normalize_api_key(api_key))
    if encrypted is None:
        raise ValueError("An API key is required")
    return encrypted


def decrypt_api_key(value: str | None) -> str | None:
    return normalize_api_key(decrypt_secret(value))


def set_runtime_config(config: RuntimeLLMConfig | None) -> None:
    global _runtime_config, _runtime_config_loaded
    _runtime_config = config
    _runtime_config_loaded = True


def get_runtime_config() -> RuntimeLLMConfig | None:
    # The workspace provider is an administrator-managed database setting.
    # Never fall back to API keys or model names from the process environment.
    return _runtime_config if _runtime_config_loaded else None


async def load_runtime_config(db: AsyncSession) -> None:
    result = await db.execute(select(LLMProviderConfig).where(LLMProviderConfig.config_key == "workspace"))
    row = result.scalar_one_or_none()
    if row is None:
        set_runtime_config(None)
        return
    key = decrypt_api_key(row.encrypted_api_key)
    set_runtime_config(RuntimeLLMConfig(row.provider, row.model, row.base_url.rstrip("/"), key) if row.enabled and key else None)


def public_config(row: LLMProviderConfig | None) -> dict:
    runtime = get_runtime_config()
    if row is None:
        return {
            "configured": runtime is not None,
            "source": "admin" if runtime else "none",
            "enabled": bool(runtime),
            "provider": runtime.provider if runtime else "openai",
            "model": runtime.model if runtime else "gpt-4o-mini",
            "base_url": runtime.base_url if runtime else DEFAULT_BASE_URLS["openai"],
            "allow_custom_base_url": settings.LLM_ALLOW_CUSTOM_BASE_URL,
            "api_key_configured": runtime is not None,
            "api_key_hint": "Configured" if runtime else None,
        }
    key = decrypt_api_key(row.encrypted_api_key)
    return {
        "configured": bool(key),
        "source": "admin",
        "enabled": row.enabled,
        "provider": row.provider,
        "model": row.model,
        "base_url": row.base_url,
        "allow_custom_base_url": settings.LLM_ALLOW_CUSTOM_BASE_URL,
        "api_key_configured": bool(key),
        "api_key_hint": "Configured" if key else None,
    }
