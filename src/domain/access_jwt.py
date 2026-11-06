"""Cloudflare Access JWT verification for the MCP endpoint.

The origin side of `product-docs` `platform/MCP-PLAN.md` §5, and the Python equivalent of
app-platform's `identity/access-jwt` subpath. With Access Managed OAuth the client presents
an opaque Access OAuth token, Access validates it at the edge, and the origin receives the
person's identity as a JWT in `Cf-Access-Jwt-Assertion`.

Verification is full and pinned:

- signature against the team JWKS (``https://<team>.cloudflareaccess.com/cdn-cgi/access/certs``),
  with PyJWT's key cache and rotation-safe refresh on an unknown ``kid``;
- ``iss`` = the configured team origin, ``aud`` = the MCP Access application's AUD tag;
- RS256 only; ``exp`` and ``nbf`` with a small configurable leeway;
- ``sub`` is required as the Access-stable identity of the person at this team.

Identity mapping mirrors the app-platform verifier: the Entra object id arrives in a
configurable claim path (default top-level ``oid``). The P0 spike proved that claim does
not reach the origin today, so the email fallback is a deliberate, configurable downgrade:
the email is IdP-verified but can be reassigned. Error messages are for logs and tests,
never for clients.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWK, PyJWKClient
from jwt.exceptions import (
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    MissingRequiredClaimError,
    PyJWKClientConnectionError,
    PyJWKClientError,
    PyJWKSetError,
    PyJWTError,
)

ACCESS_JWT_HEADER = "cf-access-jwt-assertion"

DEFAULT_CLOCK_TOLERANCE_SECONDS = 60
DEFAULT_JWKS_TIMEOUT_SECONDS = 10.0
DEFAULT_JWKS_COOLDOWN_SECONDS = 60.0
DEFAULT_UNKNOWN_KID_TTL_SECONDS = 60.0
DEFAULT_OID_CLAIM_PATH = "oid"
MAX_NEGATIVE_KID_ENTRIES = 1024


class AccessJwtError(Exception):
    """A rejected Access token, or an unavailable team JWKS.

    ``code`` is stable and is what callers branch on; the message is for logs and tests.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AccessIdentity:
    """Who the caller is, mapped from a verified Access JWT."""

    sub: str
    email: str | None
    oid: str | None


@dataclass(frozen=True)
class AccessJwtConfig:
    """Pinned verification settings for one Access application."""

    team_domain: str
    audience: str | Sequence[str]
    oid_claim_path: str = DEFAULT_OID_CLAIM_PATH
    allow_email_fallback: bool = False
    clock_tolerance_seconds: int = DEFAULT_CLOCK_TOLERANCE_SECONDS
    jwks_timeout_seconds: float = DEFAULT_JWKS_TIMEOUT_SECONDS
    #: Minimum seconds between forced JWKS refreshes after an unknown `kid`. This is the
    #: storm guard: PyJWT's own default is 30, too short for an unauthenticated endpoint.
    jwks_cooldown_seconds: float = DEFAULT_JWKS_COOLDOWN_SECONDS
    #: An unknown `kid` is refused from a negative cache for this long, so repeating the
    #: same bogus key never even reaches the JWKS client.
    unknown_kid_ttl_seconds: float = DEFAULT_UNKNOWN_KID_TTL_SECONDS


def _claim_at(payload: Mapping[str, Any], path: str) -> Any:
    node: Any = payload
    for segment in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(segment)
    return node


def _is_https_origin(value: str) -> bool:
    return value.startswith("https://") and "/" not in value[len("https://") :]


def _is_loopback_http(value: str) -> bool:
    # Tests run a local JWKS over plain HTTP. Production configuration is always https.
    return (
        value.startswith("http://127.0.0.1")
        or value.startswith("http://localhost")
        or value.startswith("http://[::1]")
    )


def _audience_list(audience: str | Sequence[str]) -> list[str]:
    values = [audience] if isinstance(audience, str) else list(audience)
    return [value for value in values if value]


class AccessJwtVerifier:
    """Verifies `Cf-Access-Jwt-Assertion` tokens for one Access application."""

    def __init__(
        self, config: AccessJwtConfig, *, jwk_client: PyJWKClient | None = None
    ) -> None:
        team_domain = config.team_domain.strip().rstrip("/")
        if not (_is_https_origin(team_domain) or _is_loopback_http(team_domain)):
            raise AccessJwtError(
                "malformed",
                f"team_domain must be an https origin, got {config.team_domain!r}",
            )
        audience = _audience_list(config.audience)
        if not audience:
            raise AccessJwtError(
                "malformed",
                "audience (the Access app AUD tag) is required",
            )
        if not config.oid_claim_path:
            raise AccessJwtError("malformed", "oid_claim_path must not be empty")

        self._team_domain = team_domain
        self._audience = audience
        self._oid_claim_path = config.oid_claim_path
        self._allow_email_fallback = config.allow_email_fallback
        self._clock_tolerance = config.clock_tolerance_seconds
        self._unknown_kid_ttl = config.unknown_kid_ttl_seconds
        if self._unknown_kid_ttl < 0:
            raise AccessJwtError("malformed", "unknown_kid_ttl_seconds must be >= 0")
        if config.jwks_cooldown_seconds < 0:
            raise AccessJwtError("malformed", "jwks_cooldown_seconds must be >= 0")
        self._unknown_kids: dict[str, float] = {}
        self._unknown_kids_lock = threading.Lock()
        self._jwk_client = jwk_client or PyJWKClient(
            f"{team_domain}/cdn-cgi/access/certs",
            cache_keys=True,
            timeout=config.jwks_timeout_seconds,
            # PyJWT's default is 30 s; an unauthenticated endpoint needs the longer
            # cooldown so a stream of unknown kids cannot force a refresh per request.
            cooldown_duration=config.jwks_cooldown_seconds,
        )

    def _known_unknown_kid(self, kid: str) -> bool:
        now = time.monotonic()
        with self._unknown_kids_lock:
            expires = self._unknown_kids.get(kid)
            if expires is None:
                return False
            if expires <= now:
                del self._unknown_kids[kid]
                return False
            return True

    def _remember_unknown_kid(self, kid: str) -> None:
        if self._unknown_kid_ttl <= 0:
            return
        now = time.monotonic()
        with self._unknown_kids_lock:
            if len(self._unknown_kids) >= MAX_NEGATIVE_KID_ENTRIES:
                # Drop expired entries first; if the cache is still full of live ones, the
                # oldest is evicted. Bounded memory is the point: `kid` is caller input.
                for key, expiry in list(self._unknown_kids.items()):
                    if expiry <= now:
                        del self._unknown_kids[key]
                if len(self._unknown_kids) >= MAX_NEGATIVE_KID_ENTRIES:
                    self._unknown_kids.pop(next(iter(self._unknown_kids)))
            self._unknown_kids[kid] = now + self._unknown_kid_ttl

    def _forget_unknown_kid(self, kid: str) -> None:
        with self._unknown_kids_lock:
            self._unknown_kids.pop(kid, None)

    def verify(self, token: str) -> AccessIdentity:
        """Verify a raw Access JWT and return the person it names."""
        if not isinstance(token, str) or not token.strip():
            raise AccessJwtError("malformed", "the Access JWT is empty")

        try:
            header = jwt.get_unverified_header(token)
        except PyJWTError:
            raise AccessJwtError("malformed", "the Access JWT is malformed") from None
        if header.get("alg") != "RS256":
            raise AccessJwtError("wrong_algorithm", "the Access JWT `alg` is not RS256")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise AccessJwtError("malformed", "the Access JWT carries no `kid`")

        if self._known_unknown_kid(kid):
            # No JWKS client call at all: the same bogus key was already refused recently.
            raise AccessJwtError(
                "invalid_signature", "the Access JWT signature is invalid"
            )

        try:
            signing_key: PyJWK = self._jwk_client.get_signing_key(kid)
        except PyJWKClientConnectionError:
            raise AccessJwtError(
                "jwks_unavailable", "the team JWKS could not be fetched"
            ) from None
        except PyJWKSetError:
            # The JWKS itself is unusable, not this token. Do not negative-cache the kid:
            # the next request should be allowed to try again once the set recovers.
            raise AccessJwtError(
                "jwks_unavailable", "the team JWKS could not be used"
            ) from None
        except PyJWKClientError:
            self._remember_unknown_kid(kid)
            raise AccessJwtError(
                "invalid_signature", "the Access JWT signature is invalid"
            ) from None
        self._forget_unknown_kid(kid)

        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._team_domain,
                audience=self._audience,
                leeway=self._clock_tolerance,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except ExpiredSignatureError:
            raise AccessJwtError("expired", "the Access JWT has expired") from None
        except ImmatureSignatureError:
            raise AccessJwtError(
                "not_yet_valid", "the Access JWT is not yet valid (nbf)"
            ) from None
        except InvalidIssuerError:
            raise AccessJwtError(
                "bad_issuer", "the Access JWT `iss` is not the configured team domain"
            ) from None
        except InvalidAudienceError:
            raise AccessJwtError(
                "bad_audience", "the Access JWT `aud` is not the configured AUD tag"
            ) from None
        except InvalidAlgorithmError:
            raise AccessJwtError(
                "wrong_algorithm", "the Access JWT `alg` is not RS256"
            ) from None
        except MissingRequiredClaimError:
            raise AccessJwtError(
                "malformed", "the Access JWT is missing a required claim"
            ) from None
        except InvalidSignatureError:
            raise AccessJwtError(
                "invalid_signature", "the Access JWT signature is invalid"
            ) from None
        except PyJWTError:
            raise AccessJwtError("malformed", "the Access JWT was rejected") from None

        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise AccessJwtError("missing_principal", "the Access JWT carries no `sub`")

        email = claims.get("email")
        email_value = email if isinstance(email, str) and email else None

        oid = _claim_at(claims, self._oid_claim_path)
        if isinstance(oid, str) and oid:
            return AccessIdentity(sub=sub, email=email_value, oid=oid)

        if self._allow_email_fallback and email_value:
            return AccessIdentity(sub=sub, email=email_value, oid=None)

        raise AccessJwtError(
            "missing_principal",
            f"the Access JWT carries no `{self._oid_claim_path}` claim and "
            "the email fallback is not enabled",
        )
