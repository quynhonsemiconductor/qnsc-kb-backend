from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db, require_permission
from src.core.config import settings
from src.domain.llm_client import (
    Provider,
    ProviderAuthError,
    ProviderRateLimitError,
    probe_provider,
)
from src.domain.llm_config import (
    DEFAULT_BASE_URLS,
    SUPPORTED_PROVIDERS,
    RuntimeLLMConfig,
    decrypt_api_key,
    encrypt_api_key,
    normalize_api_key,
    public_config,
    set_runtime_config,
)
from src.domain.llm_endpoint_security import (
    UnsafeLLMEndpointError,
    normalize_public_https_url,
)
from src.domain.llm_verification import issue_verification_token, verify_token
from src.models import User
from src.models.ops import LLMProviderConfig

router = APIRouter()


ProviderName = Literal["openai", "glm", "groq", "gemini", "custom"]


class LLMConfigCandidate(BaseModel):
    provider: ProviderName
    model: str = Field(min_length=1, max_length=150)
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=1000)


class LLMConfigUpdate(LLMConfigCandidate):
    enabled: bool = True
    verification_token: str | None = Field(default=None, max_length=4000)


def _clean_base_url(provider: str, base_url: str | None) -> str:
    default_url = DEFAULT_BASE_URLS.get(provider)
    if provider == "custom" and not (base_url or "").strip():
        raise HTTPException(
            status_code=422,
            detail="An OpenAI-compatible Chat Completions endpoint is required",
        )
    value = (base_url or default_url or "").strip().rstrip("/")
    if (
        provider != "custom"
        and value != default_url
        and not settings.LLM_ALLOW_CUSTOM_BASE_URL
    ):
        raise HTTPException(
            status_code=422,
            detail="Custom LLM endpoints are disabled. Choose OpenAI-compatible API to configure another trusted public endpoint.",
        )
    try:
        return normalize_public_https_url(value)
    except UnsafeLLMEndpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _candidate_provider(
    payload: LLMConfigCandidate, base_url: str, api_key: str
) -> Provider:
    return Provider(
        payload.provider,
        payload.model.strip(),
        base_url,
        api_key,
        native_gemini=payload.provider == "gemini",
        custom_endpoint=(
            payload.provider == "custom"
            or base_url != DEFAULT_BASE_URLS.get(payload.provider)
        ),
    )


def _same_config(
    row: LLMProviderConfig | None,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None,
) -> bool:
    if row is None:
        return False
    return bool(
        row.provider == provider
        and row.model == model
        and row.base_url.rstrip("/") == base_url
        and normalize_api_key(decrypt_api_key(row.encrypted_api_key))
        == normalize_api_key(api_key)
    )


@router.get("/config")
async def get_llm_config(current_user: User = Depends(require_permission("role.manage", scope="global")), db: AsyncSession = Depends(get_db)) -> Any:
    result = await db.execute(select(LLMProviderConfig).where(LLMProviderConfig.config_key == "workspace"))
    return public_config(result.scalar_one_or_none())


@router.post("/test")
async def test_llm_config(
    payload: LLMConfigCandidate,
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if payload.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported LLM provider")
    result = await db.execute(
        select(LLMProviderConfig).where(LLMProviderConfig.config_key == "workspace")
    )
    row = result.scalar_one_or_none()
    existing_key = (
        decrypt_api_key(row.encrypted_api_key)
        if row and row.provider == payload.provider
        else None
    )
    api_key = normalize_api_key(payload.api_key) or existing_key
    if not api_key:
        raise HTTPException(
            status_code=422,
            detail="Enter an API key for this provider before testing it",
        )
    base_url = _clean_base_url(payload.provider, payload.base_url)
    provider = _candidate_provider(payload, base_url, api_key)
    try:
        latency_ms = await probe_provider(provider)
    except UnsafeLLMEndpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderAuthError as exc:
        raise HTTPException(status_code=422, detail="The provider rejected the API key") from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=422,
            detail="The provider rate limited the connection test; wait and try again",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    token, expires_at = issue_verification_token(
        str(current_user.id),
        payload.provider,
        payload.model.strip(),
        base_url,
        api_key,
    )
    return {
        "ok": True,
        "provider": payload.provider,
        "model": payload.model.strip(),
        "base_url": base_url,
        "latency_ms": latency_ms,
        "verification_token": token,
        "expires_at": expires_at.isoformat(),
    }


@router.put("/config")
async def update_llm_config(
    payload: LLMConfigUpdate,
    current_user: User = Depends(require_permission("role.manage", scope="global")),
    db: AsyncSession = Depends(get_db),
) -> Any:
    if payload.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=422, detail="Unsupported LLM provider")
    result = await db.execute(
        select(LLMProviderConfig).where(LLMProviderConfig.config_key == "workspace")
    )
    row = result.scalar_one_or_none()
    existing_key = (
        decrypt_api_key(row.encrypted_api_key)
        if row and row.provider == payload.provider
        else None
    )
    api_key = normalize_api_key(payload.api_key) or existing_key
    if payload.enabled and not api_key:
        raise HTTPException(
            status_code=422,
            detail="An API key is required when the LLM provider is enabled",
        )
    base_url = _clean_base_url(payload.provider, payload.base_url)
    model = payload.model.strip()
    unchanged = _same_config(row, payload.provider, model, base_url, api_key)
    if api_key and not unchanged:
        if not payload.verification_token:
            raise HTTPException(
                status_code=422,
                detail="Test this provider successfully before saving it",
            )
        try:
            verify_token(
                payload.verification_token,
                str(current_user.id),
                payload.provider,
                model,
                base_url,
                api_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if row is None:
        row = LLMProviderConfig(config_key="workspace", model=model, base_url=base_url)
        db.add(row)
    row.enabled = payload.enabled
    row.provider = payload.provider
    row.model = model
    row.base_url = base_url
    if payload.api_key and normalize_api_key(payload.api_key):
        row.encrypted_api_key = encrypt_api_key(api_key)
    elif row.encrypted_api_key is None and api_key:
        row.encrypted_api_key = encrypt_api_key(api_key)
    await db.commit()
    await db.refresh(row)
    set_runtime_config(
        RuntimeLLMConfig(row.provider, row.model, row.base_url, api_key)
        if row.enabled and api_key
        else None
    )
    return public_config(row)
