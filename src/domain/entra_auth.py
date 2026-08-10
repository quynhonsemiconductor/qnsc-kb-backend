"""Microsoft Entra authorization-code and ID-token verification helpers."""
from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from jose import jwk, jwt

from src.core.config import settings


def configured() -> bool:
    return bool(settings.MICROSOFT_CLIENT_ID and settings.MICROSOFT_CLIENT_SECRET and settings.MICROSOFT_LOGIN_REDIRECT_URI)


def new_nonce() -> str:
    return secrets.token_urlsafe(32)


def authorization_url(state: str, nonce: str) -> str:
    params = {
        "client_id": settings.MICROSOFT_CLIENT_ID or "",
        "response_type": "code",
        "redirect_uri": settings.MICROSOFT_LOGIN_REDIRECT_URI or "",
        "response_mode": "query",
        "scope": "openid profile email",
        "state": state,
        "nonce": nonce,
    }
    return f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/authorize?{urlencode(params)}"


def _validate_id_token_metadata(header: dict[str, Any], claims: dict[str, Any]) -> tuple[str, str]:
    """Validate untrusted JOSE metadata before fetching or accepting a key."""
    algorithm = str(header.get("alg") or "")
    if algorithm != "RS256":
        raise ValueError("Microsoft ID token algorithm is not allowed")
    tenant_id = str(claims.get("tid") or "").strip()
    if not tenant_id:
        raise ValueError("Microsoft ID token has no tenant claim")
    issuer = str(claims.get("iss") or "").strip()
    expected_issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
    if issuer != expected_issuer:
        raise ValueError("Microsoft ID token issuer is invalid")
    configured_tenant = str(settings.MICROSOFT_TENANT_ID or "").strip()
    if configured_tenant.lower() != "common" and tenant_id.lower() != configured_tenant.lower():
        raise ValueError("Microsoft ID token tenant is invalid")
    return algorithm, tenant_id


async def exchange_code(code: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, trust_env=False) as client:
        response = await client.post(
            f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
            data={
                "client_id": settings.MICROSOFT_CLIENT_ID,
                "client_secret": settings.MICROSOFT_CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.MICROSOFT_LOGIN_REDIRECT_URI,
                "scope": "openid profile email",
            },
        )
        if response.status_code >= 400:
            raise ValueError("Microsoft authorization code exchange failed")
        return response.json()


async def verify_id_token(id_token: str, expected_nonce: str) -> dict[str, Any]:
    """Verify signature, audience, nonce, and tenant claims before mapping."""
    unverified_header = jwt.get_unverified_header(id_token)
    unverified_claims = jwt.get_unverified_claims(id_token)
    kid = str(unverified_header.get("kid") or "")
    if not kid:
        raise ValueError("Microsoft ID token has no key identifier")
    algorithm, _tenant_id = _validate_id_token_metadata(unverified_header, unverified_claims)
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=False, trust_env=False) as client:
        response = await client.get(
            f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/discovery/v2.0/keys"
        )
    if response.status_code >= 400:
        raise ValueError("Microsoft signing keys are unavailable")
    keys = response.json().get("keys", [])
    key_data = next((item for item in keys if item.get("kid") == kid), None)
    if not key_data:
        raise ValueError("Microsoft signing key is unknown")
    # Entra's JWKS can omit the optional `alg` member. The token header was
    # already validated as RS256 above, so pass that validated algorithm
    # explicitly instead of asking python-jose to infer it from the key.
    key = jwk.construct(key_data, algorithm=algorithm)
    claims = jwt.decode(
        id_token,
        key,
        algorithms=["RS256"],
        audience=settings.MICROSOFT_CLIENT_ID,
        options={"verify_iss": False},
    )
    if claims.get("nonce") != expected_nonce:
        raise ValueError("Microsoft ID token nonce is invalid")
    subject = str(claims.get("oid") or claims.get("sub") or "").strip()
    email = str(claims.get("preferred_username") or claims.get("email") or "").strip().lower()
    if not subject or "@" not in email:
        raise ValueError("Microsoft ID token does not identify an internal user")
    claims["subject"] = subject
    claims["email"] = email
    return claims
