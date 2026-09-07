"""One place that answers "what is this connector's provider?".

Provider identity used to be decided by comparing ``connector.system`` to the
literal ``"sharepoint"`` at roughly a dozen call sites: cursor type, identity
provider, ACL metadata key, webhook lifecycle support, OAuth configuration, and
— critically — the source-ACL intersection in ``domain/permissions.py`` and
``repositories/article.py``.

That is safe while SharePoint is the only Microsoft Graph provider and stops
being safe the moment a second one exists. ``_apply_mapped_groups`` stamps
provenance rows with ``connector.system``, so a ``onedrive`` connector would
write ``source="onedrive"`` permissions that neither ACL predicate recognises:
the provider ACL would be ignored and a company-wide reader would see a
document OneDrive never shared with them. Fail-open, not fail-closed.

So the predicates ask this module instead of comparing strings.
"""
from __future__ import annotations

#: Providers whose ingestion runs through a real remote API.
#: ``local_folder`` is deliberately absent: it has no OAuth, no delta cursor,
#: and no provider ACL.
REMOTE_PROVIDERS: frozenset[str] = frozenset({"sharepoint", "onedrive", "google_drive"})

#: Every value accepted in ``Connector.system``.
CONNECTOR_PROVIDERS: frozenset[str] = REMOTE_PROVIDERS | {"local_folder"}

#: Providers served by Microsoft Graph. They share the adapter, the Entra
#: application registration, the delta cursor semantics, and the subscription
#: lifecycle endpoint.
MICROSOFT_GRAPH_PROVIDERS: frozenset[str] = frozenset({"sharepoint", "onedrive"})

#: Provider values that stamp ``ArticleUserPermission.source`` and
#: ``DocumentSource.source_system`` for source-managed ACL rows. A row carrying
#: one of these is provider-governed and MUST be intersected with the internal
#: policy rather than treated as an ordinary internal grant.
SOURCE_ACL_PROVIDERS: tuple[str, ...] = tuple(sorted(REMOTE_PROVIDERS))


def is_microsoft_graph(system: str | None) -> bool:
    """True when this provider is reached through Microsoft Graph."""

    return system in MICROSOFT_GRAPH_PROVIDERS


def identity_provider(system: str | None) -> str:
    """The ``ExternalIdentity.provider`` value ACL principals resolve through.

    Graph providers authenticate against Entra, so a SharePoint and a OneDrive
    connector must resolve a user principal through the same identity rows.
    """

    return "microsoft_entra" if is_microsoft_graph(system) else str(system)


def cursor_type(system: str | None) -> str:
    """The delta-cursor flavour stored for this provider's scopes.

    Graph exposes ``/delta`` with a ``deltaLink``; Google Drive exposes a
    changes feed with a page token. The distinction is the shape of the stored
    cursor, not the vendor.
    """

    return "delta" if is_microsoft_graph(system) else "changes"


def supports_lifecycle_webhooks(system: str | None) -> bool:
    """Only Graph sends subscriptionRemoved / reauthorizationRequired events."""

    return is_microsoft_graph(system)


def acl_present_key(system: str | None) -> str:
    """Legacy per-provider ACL-present metadata key.

    ``provider_acl_present`` is written for every provider and is what new code
    reads. SharePoint additionally keeps ``sharepoint_acl_present`` because
    documents synced before the generic key existed only carry that one, and
    the fail-closed intersection reads both.
    """

    return "sharepoint_acl_present" if system == "sharepoint" else "provider_acl_present"
