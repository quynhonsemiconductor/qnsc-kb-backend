"""Microsoft Entra authorization-code and ID-token verification helpers."""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from src.core.config import settings


def _tenant() -> str:
    """The pinned tenant. Raises when there is none: sign-in never falls back to a multi-tenant authority."""
    tenant = settings.microsoft_tenant
    if tenant is None:
        raise ValueError(
            "MICROSOFT_TENANT_ID must be the Entra tenant GUID (not blank, common, organizations or consumers)"
        )
    return tenant


def configured() -> bool:
    # Fail closed: without a pinned tenant, sign-in is "not configured" (503), not "any Microsoft account".
    return bool(
        settings.MICROSOFT_CLIENT_ID
        and settings.MICROSOFT_CLIENT_SECRET
        and settings.MICROSOFT_LOGIN_REDIRECT_URI
        and settings.microsoft_tenant
    )


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
    return f"https://login.microsoftonline.com/{_tenant()}/oauth2/v2.0/authorize?{urlencode(params)}"


def _validated_algorithm(header: dict[str, Any]) -> str:
    """Validate the untrusted JOSE header before fetching or accepting a key.

    Only ``alg`` and ``kid`` are read before the signature is checked. Nothing else in
    an unverified token has to be trusted: the JWKS is fetched from the CONFIGURED
    tenant, so tenant and issuer are validated afterwards, by
    :func:`_validate_tenant_claims`, against claims the signature already vouches for.
    """
    algorithm = str(header.get("alg") or "")
    if algorithm != "RS256":
        raise ValueError("Microsoft ID token algorithm is not allowed")
    return algorithm


def _validate_tenant_claims(claims: dict[str, Any]) -> str:
    """Pin the tenant and issuer of claims whose signature is already verified.

    Both are compared with OUR configured tenant, never with each other: a token whose ``iss`` agrees with its
    own ``tid`` proves only that it is a well-formed token from SOME tenant. There is no "common" escape hatch;
    with no pinned tenant nothing is accepted.
    """
    pinned = _tenant().lower()
    tenant_id = str(claims.get("tid") or "").strip()
    if not tenant_id:
        raise ValueError("Microsoft ID token has no tenant claim")
    if tenant_id.lower() != pinned:
        raise ValueError("Microsoft ID token tenant is invalid")
    issuer = str(claims.get("iss") or "").strip().lower()
    if issuer != f"https://login.microsoftonline.com/{pinned}/v2.0":
        raise ValueError("Microsoft ID token issuer is invalid")
    return tenant_id


async def exchange_code(code: str) -> dict[str, Any]:
    tenant = (
        _tenant()
    )  # before any network call: an unpinned config never reaches a multi-tenant authority
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=False, trust_env=False
    ) as client:
        response = await client.post(
            f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
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
    tenant = _tenant()
    unverified_header = jwt.get_unverified_header(id_token)
    kid = str(unverified_header.get("kid") or "")
    if not kid:
        raise ValueError("Microsoft ID token has no key identifier")
    algorithm = _validated_algorithm(unverified_header)
    async with httpx.AsyncClient(
        timeout=15.0, follow_redirects=False, trust_env=False
    ) as client:
        response = await client.get(
            f"https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"
        )
    if response.status_code >= 400:
        raise ValueError("Microsoft signing keys are unavailable")
    keys = response.json().get("keys", [])
    key_data = next((item for item in keys if item.get("kid") == kid), None)
    if not key_data:
        raise ValueError("Microsoft signing key is unknown")
    # Entra's JWKS can omit the optional `alg` member and PyJWK refuses to guess. The
    # token header was already validated as RS256 above, so pass that validated
    # algorithm explicitly rather than letting the key material choose it.
    key = jwt.PyJWK.from_dict(key_data, algorithm=algorithm)
    # `audience` is not optional here. PyJWT RAISES InvalidAudienceError for a token
    # that carries `aud` when no audience is passed, where python-jose silently skipped
    # the check — and every Entra id_token carries `aud`.
    claims = jwt.decode(
        id_token,
        key.key,
        algorithms=["RS256"],
        audience=settings.MICROSOFT_CLIENT_ID,
    )
    # Tenant and issuer only mean something once the signature is verified.
    _validate_tenant_claims(claims)
    if claims.get("nonce") != expected_nonce:
        raise ValueError("Microsoft ID token nonce is invalid")
    subject = str(claims.get("oid") or claims.get("sub") or "").strip()
    email = (
        str(claims.get("preferred_username") or claims.get("email") or "")
        .strip()
        .lower()
    )
    if not subject or "@" not in email:
        raise ValueError("Microsoft ID token does not identify an internal user")
    claims["subject"] = subject
    claims["email"] = email
    return claims
