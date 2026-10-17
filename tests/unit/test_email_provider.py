"""get_email_sender() picks a transport from EMAIL_PROVIDER.

The Graph and console cases are covered in test_graph_mail.py. This file keeps the simple cases honest:
``fake`` is an explicit opt-in, an unknown name fails loudly, and ``ses`` (removed) is now one of them.
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


def test_ses_is_gone_and_fails_loudly(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, ENVIRONMENT="development", EMAIL_PROVIDER="ses"))
    with pytest.raises(RuntimeError, match="Unknown EMAIL_PROVIDER 'ses'"):
        email.get_email_sender()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(email, "settings", Settings(_env_file=None, EMAIL_PROVIDER="mailgun"))
    with pytest.raises(RuntimeError, match="Unknown EMAIL_PROVIDER"):
        email.get_email_sender()
