"""Local LLM-free stack: LocalStack stands in for R2 in development, and only there.

The browser uploads to a presigned URL, so the URL must carry the host the browser calls (S3_PUBLIC_ENDPOINT_URL), while the
API and worker reach LocalStack by its compose name. Production stays on Cloudflare R2 with no local endpoint.
"""

import pytest

from src.core.config import is_local_object_endpoint, settings
from src.domain import source_storage


@pytest.mark.parametrize(
    "value",
    ["http://localhost:4567", "http://127.0.0.1:4566", "http://localstack:4566"],
)
def test_local_object_endpoints_are_plain_http_on_this_machine_or_the_compose_network(
    value,
):
    assert is_local_object_endpoint(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://localhost:4567",  # local emulators are plain HTTP
        "http://localhost",  # no port: not the emulator this stack publishes
        "http://evil.example:4566",
        "http://user:pass@localhost:4567",  # never carry credentials in the URL
        "https://69e52835cf2d08edde5b6ebd741d30fa.r2.cloudflarestorage.com",
        "",
        None,
    ],
)
def test_anything_else_is_not_a_local_object_endpoint(value):
    assert not is_local_object_endpoint(value)


def test_the_local_endpoint_is_honoured_only_in_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    assert settings.allows_local_object_endpoints
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert not settings.allows_local_object_endpoints
    monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
    assert not settings.allows_local_object_endpoints


def _configure_local_stack(monkeypatch, *, internal, public=None):
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "qnsc-kb-sources-local")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_PREFIX", "qnsc-sources")
    monkeypatch.setattr(settings, "R2_ACCESS_KEY_ID", "test")
    monkeypatch.setattr(settings, "R2_SECRET_ACCESS_KEY", "test")
    monkeypatch.setattr(settings, "S3_ENDPOINT_URL", internal)
    monkeypatch.setattr(settings, "S3_PUBLIC_ENDPOINT_URL", public)


def test_the_api_and_worker_client_talks_to_localstack_in_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    _configure_local_stack(monkeypatch, internal="http://localstack:4566")
    client = source_storage._s3_client()
    assert client.meta.endpoint_url == "http://localstack:4566"


def test_a_presigned_url_is_built_for_the_browser_origin_not_the_compose_name(
    monkeypatch,
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    _configure_local_stack(
        monkeypatch, internal="http://localstack:4566", public="http://localhost:4567"
    )
    url = source_storage.create_presigned_source_url(
        "s3://qnsc-kb-sources-local/qnsc-sources/acme/doc.pdf", operation="put_object"
    )
    assert url.startswith("http://localhost:4567/qnsc-kb-sources-local/qnsc-sources/")
    assert "localstack" not in url


def test_a_local_endpoint_is_refused_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    _configure_local_stack(monkeypatch, internal="http://localstack:4566")
    with pytest.raises(RuntimeError, match="Cloudflare R2 endpoint is required"):
        source_storage._s3_client()
