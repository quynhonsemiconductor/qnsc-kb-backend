"""Email delivery boundary used by invitations and notification workers.

Providers (EMAIL_PROVIDER):

* ``graph``   Microsoft 365, Microsoft Graph ``sendMail`` with the **application** permission, authenticating
              as a dedicated Entra app with a **certificate** (a signed client assertion, no client secret).
              This is the provider for every deployed environment.
* ``console`` development only: writes the message to the log instead of sending it, so the invitation and
              password-reset links are visible without a mail server. Refused outside development.
* ``fake``    in-memory list, for unit tests.

Why a certificate and a dedicated app. ``Mail.Send`` as an *application* permission is tenant-wide unless it is
limited with Exchange Online RBAC for Applications, and a client secret is a bearer string that anything which
can read the environment can replay. The mail app holds no Entra Graph consent at all: its only authority is an
Exchange role assignment (``Application Mail.Send``) scoped to the one sender mailbox, so a stolen key can send
as that mailbox and as nobody else. See docs/runbooks/email-graph.md.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from src.core.config import settings

logger = structlog.get_logger()

LOGIN_AUTHORITY = "https://login.microsoftonline.com"
GRAPH_API = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
# Refresh a token this long before it expires, so a request never carries one that lapses in flight.
_TOKEN_REFRESH_MARGIN_SECONDS = 120
_CERT_EXPIRY_WARNING_DAYS = 30


class EmailSender(ABC):
    @abstractmethod
    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        raise NotImplementedError


class FakeEmailSender(EmailSender):
    """Deterministic sender for unit tests."""

    sent: list[dict[str, str]] = []

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        self.sent.append({"to": to, "subject": subject, "text": text, "html": html or ""})


class ConsoleEmailSender(EmailSender):
    """Development: log the message so invite and reset links can be copied from the terminal.

    The BODY is logged on purpose (that is the whole point locally), which is exactly why this refuses to be
    selected anywhere but development: in a deployed environment it would write one-time links to the logs.
    """

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        logger.info("Email (console provider, not sent)", to=to, subject=subject, text=text)


class MailSendError(RuntimeError):
    """Graph (or the token endpoint) refused the request. Carries what is needed to diagnose it, never a secret."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


# ── certificate credential ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Credential:
    private_key: Any
    thumbprint_sha1: str
    thumbprint_sha256: str
    not_after: datetime


def _normalise_pem(value: str) -> str:
    """Secrets Manager and shell exports often carry a PEM with literal backslash-n instead of newlines."""
    return value.replace("\\n", "\n").strip()


_credential_cache: dict[tuple[str, str], _Credential] = {}


def _credential() -> _Credential:
    cert_pem = _normalise_pem(settings.MAIL_GRAPH_CERT_PEM or "")
    key_pem = _normalise_pem(settings.MAIL_GRAPH_PRIVATE_KEY_PEM or "")
    if not cert_pem or not key_pem:
        raise MailSendError("MAIL_GRAPH_CERT_PEM and MAIL_GRAPH_PRIVATE_KEY_PEM are required for EMAIL_PROVIDER=graph")
    cache_key = (hashlib.sha256(cert_pem.encode()).hexdigest(), hashlib.sha256(key_pem.encode()).hexdigest())
    cached = _credential_cache.get(cache_key)
    if cached:
        return cached
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    try:
        cert = x509.load_pem_x509_certificate(cert_pem.encode())
        key = serialization.load_pem_private_key(key_pem.encode(), password=None)
    except Exception as exc:  # the message names the setting, never the material
        raise MailSendError(f"MAIL_GRAPH_CERT_PEM / MAIL_GRAPH_PRIVATE_KEY_PEM could not be parsed: {type(exc).__name__}") from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise MailSendError("MAIL_GRAPH_PRIVATE_KEY_PEM must be an RSA key (RS256 client assertion)")
    public = cert.public_key()
    if not isinstance(public, rsa.RSAPublicKey) or public.public_numbers() != key.public_key().public_numbers():
        raise MailSendError("MAIL_GRAPH_CERT_PEM does not belong to MAIL_GRAPH_PRIVATE_KEY_PEM")
    not_after = cert.not_valid_after_utc
    now = datetime.now(timezone.utc)
    if not_after <= now:
        raise MailSendError(f"The mail certificate expired on {not_after.date().isoformat()}; rotate it (docs/runbooks/email-graph.md)")
    days_left = (not_after - now).days
    if days_left <= _CERT_EXPIRY_WARNING_DAYS:
        logger.warning("Mail certificate expires soon", days_left=days_left, expires=not_after.date().isoformat())
    der = cert.public_bytes(serialization.Encoding.DER)
    credential = _Credential(
        private_key=key,
        thumbprint_sha1=_b64url(hashlib.sha1(der).digest()),  # noqa: S324 - the x5t header is defined as SHA-1
        thumbprint_sha256=_b64url(hashlib.sha256(der).digest()),
        not_after=not_after,
    )
    _credential_cache.clear()
    _credential_cache[cache_key] = credential
    return credential


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_client_assertion(*, tenant: str, client_id: str, now: float | None = None) -> str:
    """The signed JWT that proves possession of the certificate's private key to Entra (RFC 7523)."""
    import jwt

    credential = _credential()
    issued = int(now if now is not None else time.time())
    claims = {
        "aud": f"{LOGIN_AUTHORITY}/{tenant}/oauth2/v2.0/token",
        "iss": client_id,
        "sub": client_id,
        "jti": str(uuid.uuid4()),
        "nbf": issued - 30,  # tolerate a little clock skew
        "exp": issued + 600,
    }
    headers = {"x5t": credential.thumbprint_sha1, "x5t#S256": credential.thumbprint_sha256}
    return jwt.encode(claims, credential.private_key, algorithm="RS256", headers=headers)


# ── token ────────────────────────────────────────────────────────────────────────────────────
_token: tuple[str, float] | None = None
# One lock per event loop: a Celery task runs each coroutine on a fresh loop, and an asyncio.Lock belongs to one.
_token_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def reset_token_cache() -> None:
    global _token
    _token = None


def _error_from(response: httpx.Response, what: str) -> MailSendError:
    code = message = None
    try:
        body = response.json()
        error = body.get("error")
        if isinstance(error, dict):
            code, message = error.get("code"), error.get("message")
        else:  # the token endpoint: {"error": "invalid_client", "error_description": "..."}
            code, message = error, body.get("error_description")
    except Exception:
        message = response.text[:200]
    request_id = response.headers.get("request-id") or response.headers.get("x-ms-request-id")
    text = f"{what} failed: HTTP {response.status_code}" + (f" {code}" if code else "") + (f": {str(message)[:300]}" if message else "")
    return MailSendError(text, status=response.status_code, code=str(code) if code else None, request_id=request_id)


async def _access_token(client: httpx.AsyncClient) -> str:
    global _token
    if _token and _token[1] - time.monotonic() > _TOKEN_REFRESH_MARGIN_SECONDS:
        return _token[0]
    loop = asyncio.get_running_loop()
    lock = _token_locks.get(loop)
    if lock is None:
        lock = _token_locks[loop] = asyncio.Lock()
    async with lock:
        if _token and _token[1] - time.monotonic() > _TOKEN_REFRESH_MARGIN_SECONDS:
            return _token[0]
        tenant = (settings.MICROSOFT_TENANT_ID or "").strip()
        client_id = (settings.MAIL_GRAPH_CLIENT_ID or "").strip()
        if not client_id:
            raise MailSendError("MAIL_GRAPH_CLIENT_ID is not configured")
        if tenant.lower() in {"", "common", "organizations", "consumers"}:
            raise MailSendError("MICROSOFT_TENANT_ID must be the tenant GUID for application mail (not 'common')")
        response = await client.post(
            f"{LOGIN_AUTHORITY}/{tenant}/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "scope": GRAPH_SCOPE,
                "grant_type": "client_credentials",
                "client_assertion_type": _ASSERTION_TYPE,
                "client_assertion": build_client_assertion(tenant=tenant, client_id=client_id),
            },
        )
        if response.status_code != 200:
            raise _error_from(response, "Requesting the Graph token")
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise MailSendError("The token endpoint returned no access_token")
        _token = (str(token), time.monotonic() + float(data.get("expires_in") or 3600))
        return _token[0]


class MicrosoftGraphEmailSender(EmailSender):
    """Graph ``POST /users/{mailbox}/sendMail``, application permission, certificate-authenticated."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport  # tests inject a MockTransport; production leaves it None

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None, mailbox: str | None = None) -> None:
        sender = (mailbox or settings.MICROSOFT_GRAPH_SENDER or "").strip()
        if not sender:
            raise MailSendError("MICROSOFT_GRAPH_SENDER is not configured")
        body: dict[str, Any] = {
            "message": {
                "subject": subject,
                "body": {"contentType": "HTML" if html else "Text", "content": html or text},
                "toRecipients": [{"emailAddress": {"address": to}}],
            },
            "saveToSentItems": False,
        }
        async with httpx.AsyncClient(timeout=20, trust_env=False, transport=self._transport) as client:
            token = await _access_token(client)
            response = await client.post(
                f"{GRAPH_API}/users/{sender}/sendMail",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
        if response.status_code == 401:
            reset_token_cache()  # a revoked or rotated credential must not be retried with a cached token
        if response.status_code != 202:
            raise _error_from(response, f"Graph sendMail as {sender}")
        logger.info("Email sent through Microsoft Graph", sender=sender, request_id=response.headers.get("request-id"))


def get_email_sender() -> EmailSender:
    provider = (settings.EMAIL_PROVIDER or "graph").strip().lower()
    if provider == "fake":
        return FakeEmailSender()
    if provider == "console":
        if not settings.is_development_environment:
            raise RuntimeError("EMAIL_PROVIDER=console is for local development only; it would log one-time links")
        return ConsoleEmailSender()
    if provider != "graph":
        raise RuntimeError(f"Unknown EMAIL_PROVIDER '{provider}'; expected graph, console, or fake")
    if not settings.MAIL_GRAPH_CLIENT_ID:
        environment = (settings.ENVIRONMENT or "").strip().lower()
        # No Entra mail app configured: a local run prints, a test run records. Anywhere else this is a
        # misconfiguration, and validate_production() refuses to boot on it before a message is ever queued.
        if settings.is_development_environment:
            return ConsoleEmailSender()
        if environment == "test":
            return FakeEmailSender()
    return MicrosoftGraphEmailSender()
