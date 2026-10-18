"""Email delivery boundary used by invitations and notification workers.

Providers (EMAIL_PROVIDER):

* ``graph``   Microsoft 365, Microsoft Graph ``sendMail`` with the **application** permission, authenticating
              as a dedicated Entra app with a **certificate** (a signed client assertion, no client secret).
              This is the provider for every deployed environment.
* ``console`` development only: writes the message to the log instead of sending it, so the invitation and
              password-reset links are visible without a mail server. Refused outside development.
* ``fake``    in-memory list, for unit tests.
* ``ses``     AWS SES; kept only until Graph is proven in develop, then removed.

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
import re
import time
import uuid
import weakref
from email.utils import parsedate_to_datetime
from urllib.parse import quote
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
    """Graph (or the token endpoint) refused the request. Carries what is needed to diagnose it, never a secret.

    ``stage`` says WHERE it failed, because the right response differs: ``config`` and ``token`` failures affect
    every message, so the delivery loop stops the batch instead of failing each row in turn; ``send`` failures
    belong to one message. ``permanent`` is True only for statuses no retry can fix because the MESSAGE is the problem
    (400 malformed, 413 too large). 403 and 404 at the send stage are NOT permanent: they are the SENDER being
    refused (``sender_refused``: scope missing or propagating, mailbox missing) and stop the batch. ``retry_after``
    is the server's own pacing for 429/503.
    """

    # What no retry can fix because the MESSAGE is the problem: malformed (400) or too large (413).
    PERMANENT_STATUSES = frozenset({400, 413})
    # 403 and 404 at the send stage are problems with the SENDER, not the message: the Exchange scope or the
    # role assignment is missing or still propagating (up to ~2 hours), or the mailbox does not exist or is not
    # REST-enabled yet. Every queued message fails identically, so these stop the batch, do not kill the rows,
    # and are retried on a long fixed backoff while an alarm tells a human.
    SENDER_STATUSES = frozenset({403, 404})
    SENDER_BACKOFF_SECONDS = 15 * 60

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        stage: str = "send",
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id
        self.stage = stage
        self.retry_after = retry_after

    @property
    def permanent(self) -> bool:
        return self.stage == "send" and self.status in self.PERMANENT_STATUSES

    @property
    def sender_refused(self) -> bool:
        return self.stage == "send" and self.status in self.SENDER_STATUSES

    @property
    def stops_the_batch(self) -> bool:
        """Nothing else in this batch can succeed: bad config, a rejected token request, throttling, or a
        sender Exchange will not let the app use. The rows are not charged an attempt for it."""
        return self.stage in {"config", "token"} or self.status in {429, 503} or self.sender_refused

    @property
    def minimum_wait(self) -> int:
        """Seconds before ANY retry, from the server's Retry-After or, for a refused sender, a quarter hour."""
        return max(self.retry_after or 0, self.SENDER_BACKOFF_SECONDS if self.sender_refused else 0)


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


def _reveal(value: Any) -> str:
    """The PEM settings are SecretStr (so they never appear in a repr or a log); tests may hand a plain str."""
    if value is None:
        return ""
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


_CERT_BLOCK = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", re.DOTALL)
_KEY_BLOCK = re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----.+?-----END (?:RSA )?PRIVATE KEY-----", re.DOTALL)


def _pems() -> tuple[str, str]:
    """(certificate PEM, private key PEM) from the configuration, or empty strings for what is missing.

    Deployed environments get ONE standalone secret, MAIL_GRAPH_PEM, holding the private key and the certificate
    together in a single PEM, so the pair is written and read atomically and cannot drift apart. The two separate
    settings are for a local test and for scripts/graph_mail_check.py; when both are set they win.
    """
    cert_pem = _normalise_pem(_reveal(settings.MAIL_GRAPH_CERT_PEM))
    key_pem = _normalise_pem(_reveal(settings.MAIL_GRAPH_PRIVATE_KEY_PEM))
    if cert_pem and key_pem:
        return cert_pem, key_pem
    combined = _normalise_pem(_reveal(settings.MAIL_GRAPH_PEM))
    if combined:
        cert = _CERT_BLOCK.search(combined)
        key = _KEY_BLOCK.search(combined)
        return (cert_pem or (cert.group(0) if cert else "")), (key_pem or (key.group(0) if key else ""))
    return cert_pem, key_pem


def _credential() -> _Credential:
    cert_pem, key_pem = _pems()
    if not cert_pem or not key_pem:
        raise MailSendError(
            "MAIL_GRAPH_PEM (a private key and its certificate in one PEM) is required for EMAIL_PROVIDER=graph",
            stage="config",
        )
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
        raise MailSendError(
            f"MAIL_GRAPH_PEM could not be parsed: {type(exc).__name__}", stage="config"
        ) from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise MailSendError("The mail private key must be an RSA key (RS256 client assertion)", stage="config")
    public = cert.public_key()
    if not isinstance(public, rsa.RSAPublicKey) or public.public_numbers() != key.public_key().public_numbers():
        raise MailSendError("The mail certificate does not belong to the mail private key", stage="config")
    not_after = cert.not_valid_after_utc
    now = datetime.now(timezone.utc)
    if not_after <= now:
        raise MailSendError(
            f"The mail certificate expired on {not_after.date().isoformat()}; rotate it (docs/runbooks/email-graph.md)",
            stage="config",
        )
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


def _retry_after(response: httpx.Response) -> int | None:
    """Seconds the server asked us to wait (429/503), capped: a header of "86400" must not park mail for a day.

    Both forms RFC 9110 allows: delta-seconds ("120") and an HTTP-date. Anything else is ignored.
    """
    raw = (response.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return min(int(raw), 3600)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return min(max(int((when - datetime.now(timezone.utc)).total_seconds()), 0), 3600)


def _error_from(response: httpx.Response, what: str, stage: str = "send") -> MailSendError:
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
    return MailSendError(
        text,
        status=response.status_code,
        code=str(code) if code else None,
        request_id=request_id,
        stage=stage,
        retry_after=_retry_after(response),
    )


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
            raise MailSendError("MAIL_GRAPH_CLIENT_ID is not configured", stage="config")
        if tenant.lower() in {"", "common", "organizations", "consumers"}:
            raise MailSendError("MICROSOFT_TENANT_ID must be the tenant GUID for application mail (not 'common')", stage="config")
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
            raise _error_from(response, "Requesting the Graph token", stage="token")
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise MailSendError("The token endpoint returned no access_token", stage="token")
        _token = (str(token), time.monotonic() + float(data.get("expires_in") or 3600))
        return _token[0]


class MicrosoftGraphEmailSender(EmailSender):
    """Graph ``POST /users/{mailbox}/sendMail``, application permission, certificate-authenticated."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport  # tests inject a MockTransport; production leaves it None

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None, mailbox: str | None = None) -> None:
        sender = (mailbox or settings.MICROSOFT_GRAPH_SENDER or "").strip()
        if not sender:
            raise MailSendError("MICROSOFT_GRAPH_SENDER is not configured", stage="config")
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
                # quoted: the mailbox is configuration (or a script argument), and it becomes a URL path segment
                f"{GRAPH_API}/users/{quote(sender, safe='@')}/sendMail",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
        if response.status_code == 401:
            reset_token_cache()  # a revoked or rotated credential must not be retried with a cached token
        if response.status_code != 202:
            raise _error_from(response, f"Graph sendMail as {sender}")
        logger.info("Email sent through Microsoft Graph", sender=sender, request_id=response.headers.get("request-id"))


class SesEmailSender(EmailSender):
    """Sends through AWS SES. Opt in with EMAIL_PROVIDER=ses; boto3 resolves AWS
    credentials through its normal chain (env vars, ~/.aws, or an IAM role), same
    as source_storage.py does for R2."""

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        sender = settings.MAIL_FROM_EMAIL
        if not sender:
            raise RuntimeError("MAIL_FROM_EMAIL is not configured")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for EMAIL_PROVIDER=ses") from exc
        from_address = f"{settings.MAIL_FROM_NAME} <{sender}>" if settings.MAIL_FROM_NAME else sender
        body: dict[str, Any] = {"Subject": {"Data": subject}, "Body": {}}
        if html:
            body["Body"]["Html"] = {"Data": html}
        else:
            body["Body"]["Text"] = {"Data": text}
        client = boto3.client("sesv2", region_name=settings.AWS_REGION)
        # boto3 has no async client; keep the event loop unblocked for the HTTP call.
        await asyncio.to_thread(
            client.send_email,
            FromEmailAddress=from_address,
            Destination={"ToAddresses": [to]},
            Content={"Simple": body},
        )


def configuration_problems() -> tuple[list[str], bool]:
    """What is wrong with the mail configuration, for the WORKER to act on at start-up.

    Returns ``(fatal, unconfigured)``. ``fatal`` lists settings that make this process unable to send even
    though it was told to (a client id with a missing certificate or key, or a provider that must not run
    here): the worker refuses to start. ``unconfigured`` is True for EMAIL_PROVIDER=graph with no mail app at
    all: the worker keeps running (the knowledge base works without mail), logs an error on every delivery
    pass, and an alarm watches that line. The API never calls this: it must not refuse to boot over mail.
    """
    provider = (settings.EMAIL_PROVIDER or "graph").strip().lower()
    if provider == "console" and not settings.is_development_environment:
        return ["EMAIL_PROVIDER=console is for local development only"], False
    if provider != "graph":
        return [], False
    if not (settings.MAIL_GRAPH_CLIENT_ID or "").strip():
        return [], True
    if settings.microsoft_tenant is None:
        # Without a pinned tenant every token request fails at the config stage and stops every batch for ever.
        # Say so once, at start, instead.
        return ["MICROSOFT_TENANT_ID must be the Entra tenant GUID (not blank, common, organizations or consumers)"], False
    cert_pem, key_pem = _pems()
    required = {
        "MICROSOFT_GRAPH_SENDER": settings.MICROSOFT_GRAPH_SENDER,
        "a certificate in MAIL_GRAPH_PEM": cert_pem,
        "a private key in MAIL_GRAPH_PEM": key_pem,
    }
    missing = sorted(name for name, value in required.items() if not (value or "").strip())
    return ([f"MAIL_GRAPH_CLIENT_ID is set, so {', '.join(missing)} must be too"] if missing else []), False


def certificate_status(now: datetime | None = None) -> dict[str, Any] | None:
    """Days left on the mail certificate, or None when no certificate is configured (nothing to watch).

    Run daily by the worker (a once-per-process warning at the first send is invisible in a process that lives
    for weeks). Expired is reported, not raised: the daily task must always get to log it.
    """
    cert_pem, _ = _pems()
    if not cert_pem:
        return None
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_pem.encode())
    except Exception:
        return {"valid": False, "days_left": None, "expires": None}
    not_after = cert.not_valid_after_utc
    days_left = (not_after - (now or datetime.now(timezone.utc))).days
    return {"valid": True, "days_left": days_left, "expires": not_after.date().isoformat(), "expired": days_left < 0}


def get_email_sender() -> EmailSender:
    provider = (settings.EMAIL_PROVIDER or "graph").strip().lower()
    if provider == "fake":
        return FakeEmailSender()
    if provider == "console":
        if not settings.is_development_environment:
            raise RuntimeError("EMAIL_PROVIDER=console is for local development only; it would log one-time links")
        return ConsoleEmailSender()
    if provider == "ses":
        return SesEmailSender()
    if provider != "graph":
        raise RuntimeError(f"Unknown EMAIL_PROVIDER '{provider}'; expected graph, console, ses, or fake")
    if not settings.MAIL_GRAPH_CLIENT_ID:
        environment = (settings.ENVIRONMENT or "").strip().lower()
        # No Entra mail app configured: a local run prints, a test run records. Anywhere else this is a
        # misconfiguration, and validate_production() refuses to boot on it before a message is ever queued.
        if settings.is_development_environment:
            return ConsoleEmailSender()
        if environment == "test":
            return FakeEmailSender()
    return MicrosoftGraphEmailSender()
