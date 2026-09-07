"""Which source providers this deployment can actually offer.

A provider whose credentials are absent is not a choice, it is a dead end: the
card was offered, the connector was created, and the failure only surfaced later
at `POST /connectors/{id}/oauth/start` as a 422 telling the administrator to go
set environment variables. The connector row survived that failure, so the list
filled up with sources that could never authorize.

Availability is decided HERE and enforced at creation, so the API and the UI
cannot disagree about it. Deliberately kept out of ``connector_providers.py``:
that module is imported by ``models/article.py`` and must stay free of imports.

`missing` names the settings to supply. Hiding a provider is a UI decision;
being unable to say WHY it is hidden is an operations problem, so the reason
travels with the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.config import settings
from src.domain.connector_providers import CONNECTOR_PROVIDERS, is_microsoft_graph


@dataclass(frozen=True)
class ProviderAvailability:
    system: str
    available: bool
    #: Setting names that must be supplied before this provider can be used.
    missing: tuple[str, ...] = field(default=())


def _blank(value: str | None) -> bool:
    return not (value or "").strip()


def _microsoft_missing(system: str) -> list[str]:
    """Settings a Graph connector needs, which differ by auth mode.

    Delegated mode redirects a human through consent, so it needs the redirect
    URI. Application mode has no human and no redirect; it needs a real tenant
    (``common`` cannot issue client credentials) and the scope allowlist that
    ``discover_scopes`` refuses to run without.
    """

    missing: list[str] = []
    if _blank(settings.MICROSOFT_CLIENT_ID):
        missing.append("MICROSOFT_CLIENT_ID")
    if _blank(settings.MICROSOFT_CLIENT_SECRET):
        missing.append("MICROSOFT_CLIENT_SECRET")
    if settings.microsoft_connector_auth_mode == "application":
        if _blank(settings.MICROSOFT_TENANT_ID) or settings.MICROSOFT_TENANT_ID.strip() == "common":
            missing.append("MICROSOFT_TENANT_ID")
        allowlist = (
            "MICROSOFT_SHAREPOINT_SITE_IDS"
            if system == "sharepoint"
            else "MICROSOFT_ONEDRIVE_USER_IDS"
        )
        if _blank(getattr(settings, allowlist, "")):
            missing.append(allowlist)
    elif _blank(settings.MICROSOFT_REDIRECT_URI):
        missing.append("MICROSOFT_REDIRECT_URI")
    return missing


def _google_missing() -> list[str]:
    return [
        name
        for name, value in (
            ("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID),
            ("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET),
            ("GOOGLE_REDIRECT_URI", settings.GOOGLE_REDIRECT_URI),
        )
        if _blank(value)
    ]


def provider_availability(system: str) -> ProviderAvailability:
    """Whether one provider can be created and authorized right now."""

    if system not in CONNECTOR_PROVIDERS:
        return ProviderAvailability(system=system, available=False)
    # A local folder needs no credentials, so it is always offered. It is the
    # fallback that keeps ingestion possible on a deployment with no cloud setup.
    if system == "local_folder":
        return ProviderAvailability(system=system, available=True)
    missing = _microsoft_missing(system) if is_microsoft_graph(system) else _google_missing()
    return ProviderAvailability(
        system=system, available=not missing, missing=tuple(missing)
    )


def available_providers() -> list[ProviderAvailability]:
    """Every declared provider with its current availability, name-ordered.

    Returns unavailable entries too: the caller decides whether to hide them,
    and the creation guard needs the reason to explain a rejection.
    """

    return [provider_availability(system) for system in sorted(CONNECTOR_PROVIDERS)]
