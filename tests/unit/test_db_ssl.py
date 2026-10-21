"""Unit tests for the DATABASE_SSL_CA / verify-full decision (src/core/db_ssl.py)."""

import ssl
from pathlib import Path

import pytest

from src.core import db_ssl
from src.core.db_ssl import asyncpg_ssl, ssl_connect_args

# A real, valid PEM certificate (a throwaway self-signed test CA) — ssl.load_verify_locations
# parses the file eagerly, so a fake body would fail before the code under test runs.
_CA = """-----BEGIN CERTIFICATE-----
MIIC8jCCAdqgAwIBAgIUDBLMQhIwgEVRBtwK84ofDMLSs4wwDQYJKoZIhvcNAQEL
BQAwGTEXMBUGA1UEAwwOZGItdGxzLXRlc3QtY2EwHhcNMjYxMDIwMTQ1OTE4WhcN
MjYxMDIyMTQ1OTE4WjAZMRcwFQYDVQQDDA5kYi10bHMtdGVzdC1jYTCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBAMaP3HNVcKh9iCSs+xZu6SNLuWm74GcT
whTcZBxLXV5NORB1X951RzNXKQSAPY0vQXR+eTszBRApsjylpQZLd21+sQtluCsm
9C9yCuR7ENR09zzbfLjkdvDQZiqDW4lW/VmWgW2htwiM8DL4/xAGrJ5yR2xdtMau
QnImQK2KpMJDj+eWqbnXGbXO1rnVvAU2hAW/TUlq+4M1H3tlwbTIbtSaS1dZ8SMY
ziKnmOsdjLLpVtvZMs9Pi03KD9f8zLsgoWyrmFWE5VudPiDnb2QoMe8VfDZw2JCD
tuVxEUJHPd4ih45AT6GoqI5beZeSZe9ZpBCuWCVO4ntwMci8ip4iirkCAwEAAaMy
MDAwHQYDVR0OBBYEFL4m0Ve6N37ZMKAeNLCHSo1ZlpssMA8GA1UdEwEB/wQFMAMB
Af8wDQYJKoZIhvcNAQELBQADggEBALEUH0psSQOeG2Tygg3RtQcerY01tZWTyVlF
DKKYN/QrzOtLFpwqLTzvq4pRwIotPl9nnLYnlT6uhCxSFXH2x6sZYus/NulpzmsF
g06WYtFGpXHVCmObrSqTR2gV4NbiE1F96g70TNy94M1xDwg4dXIeJwL1XIab6CsH
AfpzhxDCvj26wW3nTmxAExJMZD4nJsMig8yZ2wRR4ZhSfdf+kL6jtKCfGaAhASqN
f1a/rvBIyL0F2e/5tfKn9xmn6XHbq9O/9n3/56DFqI4o9yz8CF/JXF8PiUrhaW/Z
KgFAxDqgv7WlkO6oySlrM6joCpavg6TiNFzBgJuxO3iWkBsix80=
-----END CERTIFICATE-----"""


@pytest.fixture()
def ca_file(tmp_path: Path) -> str:
    path = tmp_path / "ca-bundle.pem"
    path.write_text(_CA)
    return str(path)


def test_ca_set_gives_a_verify_full_context(ca_file: str) -> None:
    context = asyncpg_ssl(ca_file, "production")
    assert context is not None
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_no_ca_in_production_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # BAKED_CA_PATH pinned to a path that does not exist: on a host that happens to
    # have the baked bundle the fallback would silently make this test pass for the
    # wrong reason (it would test the fallback, not the fail-closed rule).
    monkeypatch.setattr(db_ssl, "BAKED_CA_PATH", "/nonexistent/bundle.pem")
    monkeypatch.setattr("src.core.config.settings.DATABASE_SSL_CA", None)
    with pytest.raises(RuntimeError, match="DATABASE_SSL_CA"):
        asyncpg_ssl(None, "production")


def test_no_ca_outside_production_means_no_tls(ca_file: str) -> None:
    assert asyncpg_ssl(None, "development") is None
    assert (
        asyncpg_ssl(None, "Development ") is None
    )  # the deployed value is normalized away


def test_unreadable_ca_file_is_an_error_not_a_downgrade() -> None:
    with pytest.raises(OSError):
        asyncpg_ssl("/nonexistent/ca-bundle.pem", "production")


def test_connect_args_shape(ca_file: str) -> None:
    kwargs = ssl_connect_args(ca_file, "production")
    context = kwargs["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert ssl_connect_args(None, "development") == {}


def test_production_falls_back_to_the_baked_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The image bakes the RDS bundle; production with no explicit DATABASE_SSL_CA
    # uses it. Deliberately NOT an ENV default: development must never pick it up.
    baked = tmp_path / "rds-global-bundle.pem"
    baked.write_text(_CA)
    monkeypatch.setattr("src.core.config.settings.DATABASE_SSL_CA", None)
    monkeypatch.setattr(db_ssl, "BAKED_CA_PATH", str(baked))
    context = asyncpg_ssl(None, "production")
    assert context is not None and context.verify_mode == ssl.CERT_REQUIRED


def test_development_never_picks_up_the_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exact regression the baked path could cause: a dev compose container sits
    # on an image that HAS the bundle, but its database has no TLS.
    monkeypatch.setattr("src.core.config.settings.DATABASE_SSL_CA", None)
    monkeypatch.setattr(db_ssl, "BAKED_CA_PATH", "/baked/rds-bundle.pem")
    monkeypatch.setattr(db_ssl.os.path, "exists", lambda p: True)
    assert asyncpg_ssl(None, "development") is None


def test_production_with_no_ca_anywhere_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.core.config.settings.DATABASE_SSL_CA", None)
    monkeypatch.setattr(db_ssl, "BAKED_CA_PATH", "/nonexistent/bundle.pem")
    monkeypatch.setattr(db_ssl.os.path, "exists", lambda p: False)
    with pytest.raises(RuntimeError, match="DATABASE_SSL_CA"):
        asyncpg_ssl(None, "production")


def test_defaults_read_settings(ca_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.core.config.settings.DATABASE_SSL_CA", ca_file)
    monkeypatch.setattr("src.core.config.settings.ENVIRONMENT", "production")
    context = asyncpg_ssl()
    assert context is not None
    assert context.verify_mode == ssl.CERT_REQUIRED
