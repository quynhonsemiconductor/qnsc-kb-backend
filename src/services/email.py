"""Email delivery boundary used by invitations and notification workers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from src.core.config import settings


class EmailSender(ABC):
    @abstractmethod
    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        raise NotImplementedError


class FakeEmailSender(EmailSender):
    """Deterministic sender for development and unit tests."""

    sent: list[dict[str, str]] = []

    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        self.sent.append({"to": to, "subject": subject, "text": text, "html": html or ""})


async def _graph_access_token() -> str:
    if not settings.MICROSOFT_CLIENT_ID or not settings.MICROSOFT_CLIENT_SECRET:
        raise RuntimeError("Microsoft app credentials are not configured")
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        response = await client.post(
            f"https://login.microsoftonline.com/{settings.MICROSOFT_TENANT_ID}/oauth2/v2.0/token",
            data={
                "client_id": settings.MICROSOFT_CLIENT_ID,
                "client_secret": settings.MICROSOFT_CLIENT_SECRET,
                "grant_type": "client_credentials",
                "scope": settings.MICROSOFT_GRAPH_SCOPE,
            },
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("Microsoft Graph did not return an access token")
        return str(token)


class MicrosoftGraphEmailSender(EmailSender):
    async def send(self, *, to: str, subject: str, text: str, html: str | None = None) -> None:
        sender = settings.MICROSOFT_GRAPH_SENDER
        if not sender:
            raise RuntimeError("MICROSOFT_GRAPH_SENDER is not configured")
        token = await _graph_access_token()
        body: dict[str, Any] = {
            "message": {
                "subject": subject,
                "body": {"contentType": "HTML" if html else "Text", "content": html or text},
                "toRecipients": [{"emailAddress": {"address": to}}],
            },
            "saveToSentItems": "false",
        }
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            response = await client.post(
                f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
            response.raise_for_status()


def get_email_sender() -> EmailSender:
    if settings.ENVIRONMENT.lower() in {"development", "dev", "local", "test"} and not settings.MICROSOFT_GRAPH_SENDER:
        return FakeEmailSender()
    return MicrosoftGraphEmailSender()
