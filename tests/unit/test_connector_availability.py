"""A provider with no credentials must not be offered as a choice.

Before this, every declared provider was rendered as a card. Picking an
unconfigured one created a connector row that could never authorize: the failure
arrived later, at `oauth/start`, as a 422 telling the administrator to go set
environment variables — and the useless row stayed in the list looking like a
source that merely needed a click.

`available_providers` is the single answer the UI hides on and the creation
endpoint rejects on, so the two cannot disagree.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.domain.connector_availability import (
    available_providers,
    provider_availability,
)
from src.domain.connector_providers import CONNECTOR_PROVIDERS


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment with no connector credentials at all."""
    for name in (
        "MICROSOFT_CLIENT_ID",
        "MICROSOFT_CLIENT_SECRET",
        "MICROSOFT_REDIRECT_URI",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REDIRECT_URI",
    ):
        monkeypatch.setattr(settings, name, None)
    monkeypatch.setattr(settings, "MICROSOFT_CONNECTOR_AUTH_MODE", "delegated")
    monkeypatch.setattr(settings, "MICROSOFT_SHAREPOINT_SITE_IDS", "")
    monkeypatch.setattr(settings, "MICROSOFT_ONEDRIVE_USER_IDS", "")


def _configure_microsoft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_ID", "client-id")
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(settings, "MICROSOFT_REDIRECT_URI", "https://api.acme.test/cb")


def test_every_declared_provider_is_reported(unconfigured):
    """The list is the catalogue, not just the usable part.

    The UI hides what it cannot offer, but an operator still needs to see that a
    provider exists and what it is waiting for.
    """
    assert {item.system for item in available_providers()} == set(CONNECTOR_PROVIDERS)


def test_an_unconfigured_provider_is_unavailable_and_names_what_it_needs(unconfigured):
    google = provider_availability("google_drive")
    assert google.available is False
    assert google.missing == (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REDIRECT_URI",
    )


def test_a_local_folder_needs_no_credentials(unconfigured):
    """The fallback that keeps ingestion possible with no cloud setup at all."""
    local = provider_availability("local_folder")
    assert local.available is True
    assert local.missing == ()


def test_configuring_microsoft_does_not_make_google_available(unconfigured, monkeypatch):
    """Availability is per provider. One vendor's credentials are not another's."""
    _configure_microsoft(monkeypatch)
    by_system = {item.system: item for item in available_providers()}

    assert by_system["sharepoint"].available is True
    assert by_system["onedrive"].available is True
    assert by_system["google_drive"].available is False


def test_delegated_mode_requires_the_redirect_uri(unconfigured, monkeypatch):
    """Delegated consent sends a human to the provider and back."""
    _configure_microsoft(monkeypatch)
    monkeypatch.setattr(settings, "MICROSOFT_REDIRECT_URI", "")

    assert provider_availability("sharepoint").missing == ("MICROSOFT_REDIRECT_URI",)


def test_application_mode_needs_a_real_tenant_and_its_own_allowlist(unconfigured, monkeypatch):
    """App-only mode has different prerequisites, not merely more of them.

    There is no redirect (no human), `common` cannot issue client credentials, and
    `discover_scopes` refuses to run without the allowlist — so offering the card
    would promise a flow that cannot complete.
    """
    _configure_microsoft(monkeypatch)
    monkeypatch.setattr(settings, "MICROSOFT_CONNECTOR_AUTH_MODE", "application")
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", "common")

    sharepoint = provider_availability("sharepoint")
    onedrive = provider_availability("onedrive")

    assert "MICROSOFT_REDIRECT_URI" not in sharepoint.missing
    assert sharepoint.missing == ("MICROSOFT_TENANT_ID", "MICROSOFT_SHAREPOINT_SITE_IDS")
    # Each Graph provider reads its OWN allowlist: allowlisting sites must not make
    # OneDrive look ready, and vice versa.
    assert onedrive.missing == ("MICROSOFT_TENANT_ID", "MICROSOFT_ONEDRIVE_USER_IDS")


def test_one_graph_allowlist_does_not_satisfy_the_other(unconfigured, monkeypatch):
    _configure_microsoft(monkeypatch)
    monkeypatch.setattr(settings, "MICROSOFT_CONNECTOR_AUTH_MODE", "application")
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", "tenant-guid")
    monkeypatch.setattr(settings, "MICROSOFT_ONEDRIVE_USER_IDS", "alice@acme.test")

    assert provider_availability("onedrive").available is True
    assert provider_availability("sharepoint").missing == ("MICROSOFT_SHAREPOINT_SITE_IDS",)


def test_whitespace_is_not_configuration(unconfigured, monkeypatch):
    """A quoted-empty value in an env file is absent, not present."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "   ")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "https://api.acme.test/cb")

    assert provider_availability("google_drive").missing == ("GOOGLE_CLIENT_ID",)


def test_an_unknown_provider_is_never_available(unconfigured):
    """Availability must not become a way to smuggle in an unsupported system."""
    unknown = provider_availability("dropbox")
    assert unknown.available is False
