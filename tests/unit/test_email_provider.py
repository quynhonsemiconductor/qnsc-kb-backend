"""get_email_sender() picks a transport from EMAIL_PROVIDER.

The Graph and console cases are covered in test_graph_mail.py. This file keeps the legacy providers honest
until the SES path is removed: ``ses`` and ``fake`` are explicit opt-ins, and an unknown name fails loudly.
"""
from __future__ import annotations

import pytest

from src.core.config import Settings
from src.services import email


def test_default_provider_in_development_prints_instead_of_sending(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, ENVIRONMENT="development"))
    assert isinstance(email.get_email_sender(), email.ConsoleEmailSender)


def test_explicit_fake_provider_wins_outside_development(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, ENVIRONMENT="production", EMAIL_PROVIDER="fake"))
    assert isinstance(email.get_email_sender(), email.FakeEmailSender)


def test_explicit_ses_provider(monkeypatch):
    monkeypatch.setattr(
        email,
        "settings",
        Settings(_env_file=None, ENVIRONMENT="development", EMAIL_PROVIDER="ses", MAIL_FROM_EMAIL="no-reply@qnsc.vn"),
    )
    assert isinstance(email.get_email_sender(), email.SesEmailSender)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, EMAIL_PROVIDER="mailgun"))
    with pytest.raises(RuntimeError, match="Unknown EMAIL_PROVIDER"):
        email.get_email_sender()


@pytest.mark.asyncio
async def test_ses_sender_requires_mail_from_email(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, MAIL_FROM_EMAIL=None))
    sender = email.SesEmailSender()
    with pytest.raises(RuntimeError, match="MAIL_FROM_EMAIL is not configured"):
        await sender.send(to="user@qnsc.vn", subject="hi", text="hi")
