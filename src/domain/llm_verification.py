"""Short-lived proof that an administrator tested an exact LLM configuration."""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import jwt

from src.core.config import settings
from src.domain.llm_config import normalize_api_key

_TOKEN_TYPE = "llm_provider_verification"
_TOKEN_LIFETIME = timedelta(minutes=10)


def config_fingerprint(provider: str, model: str, base_url: str, api_key: str) -> str:
    material = "\0".join(
        (
            provider.strip(),
            model.strip(),
            base_url.strip(),
            normalize_api_key(api_key) or "",
        )
    ).encode("utf-8")
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"), material, hashlib.sha256
    ).hexdigest()


def issue_verification_token(
    user_id: str, provider: str, model: str, base_url: str, api_key: str
) -> tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + _TOKEN_LIFETIME
    token = jwt.encode(
        {
            "type": _TOKEN_TYPE,
            "sub": user_id,
            "fingerprint": config_fingerprint(provider, model, base_url, api_key),
            "exp": expires_at,
        },
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    return token, expires_at


def verify_token(
    token: str,
    user_id: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    try:
        claims = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:
        raise ValueError(
            "The connection test has expired or is invalid; test again"
        ) from exc
    expected = config_fingerprint(provider, model, base_url, api_key)
    if (
        claims.get("type") != _TOKEN_TYPE
        or claims.get("sub") != user_id
        or not isinstance(claims.get("fingerprint"), str)
        or not hmac.compare_digest(claims["fingerprint"], expected)
    ):
        raise ValueError(
            "The provider settings changed after the connection test; test again"
        )
    return expected
