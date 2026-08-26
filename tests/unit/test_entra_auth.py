from urllib.parse import parse_qs, urlparse
import pytest

from src.core.config import settings
from src.domain import entra_auth


def test_entra_authorization_url_contains_only_non_secret_parameters(monkeypatch):
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_ID", "client-id")
    monkeypatch.setattr(settings, "MICROSOFT_CLIENT_SECRET", "secret-value")
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", "tenant-id")
    monkeypatch.setattr(settings, "MICROSOFT_LOGIN_REDIRECT_URI", "https://kb.example/auth/entra/callback")

    url = entra_auth.authorization_url("signed-state", "nonce-value")
    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["client-id"]
    assert query["state"] == ["signed-state"]
    assert query["nonce"] == ["nonce-value"]
    assert "secret-value" not in url
    assert entra_auth.configured()


def test_entra_token_metadata_rejects_algorithm_confusion_and_wrong_tenant(monkeypatch):
    """The header check and the claim check are separate because they run at different times.

    Only `alg`/`kid` may be read before the signature is verified; tenant and issuer are
    pinned afterwards, on claims the signature already vouches for.
    """
    monkeypatch.setattr(settings, "MICROSOFT_TENANT_ID", "tenant-guid")
    claims = {"iss": "https://login.microsoftonline.com/tenant-guid/v2.0", "tid": "tenant-guid"}

    with pytest.raises(ValueError, match="algorithm"):
        entra_auth._validated_algorithm({"alg": "HS256"})

    assert entra_auth._validated_algorithm({"alg": "RS256"}) == "RS256"
    assert entra_auth._validate_tenant_claims(claims) == "tenant-guid"

    with pytest.raises(ValueError, match="tenant"):
        entra_auth._validate_tenant_claims(
            {"iss": "https://login.microsoftonline.com/other-tenant/v2.0", "tid": "other-tenant"},
        )

    with pytest.raises(ValueError, match="issuer"):
        entra_auth._validate_tenant_claims(
            {"iss": "https://login.microsoftonline.com/evil/v2.0", "tid": "tenant-guid"},
        )
