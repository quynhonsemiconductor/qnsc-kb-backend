import pytest

from src.core.config import settings
from src.domain import source_storage
from src.domain.source_storage import (
    create_presigned_source_url,
    head_source,
    list_source_objects,
    load_source,
    safe_source_media_type,
    save_source,
    source_should_display_inline,
    source_storage_key,
)


def test_local_source_storage_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "local")
    with pytest.raises(RuntimeError, match="Cloudflare R2"):
        save_source("abc123", "../../unsafe name.pdf", b"document")


def test_r2_client_rejects_non_cloudflare_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(settings, "S3_ENDPOINT_URL", "https://s3.amazonaws.com")
    monkeypatch.setattr(settings, "R2_ACCOUNT_ID", None)
    monkeypatch.setattr(settings, "R2_ACCESS_KEY_ID", "access-key")
    monkeypatch.setattr(settings, "R2_SECRET_ACCESS_KEY", "secret-key")

    with pytest.raises(RuntimeError, match="Cloudflare R2 endpoint"):
        source_storage._s3_client()


def test_r2_client_requires_explicit_r2_credentials(monkeypatch):
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(
        settings, "S3_ENDPOINT_URL", "https://account-id.r2.cloudflarestorage.com"
    )
    monkeypatch.setattr(settings, "R2_ACCOUNT_ID", None)
    monkeypatch.setattr(settings, "R2_ACCESS_KEY_ID", " ")
    monkeypatch.setattr(settings, "R2_SECRET_ACCESS_KEY", " \t")

    with pytest.raises(RuntimeError, match="R2_ACCESS_KEY_ID"):
        source_storage._s3_client()


def test_r2_upload_uses_private_tenant_scoped_object_key(monkeypatch):
    captured = {}

    class Client:
        def put_object(self, **kwargs):
            captured.update(kwargs)

        def get_object(self, **kwargs):
            return {"Body": type("Body", (), {"read": lambda self: captured["Body"]})()}

        def head_object(self, **kwargs):
            return {
                "ContentLength": len(captured["Body"]),
                "Metadata": {"sha256": "a" * 64},
            }

        def generate_presigned_url(self, **kwargs):
            captured["presign"] = kwargs
            return "https://private-r2.example/signed"

    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_PREFIX", "sources")
    monkeypatch.setattr(source_storage, "_s3_client", lambda: Client())

    source_hash = "a" * 64
    key = save_source(source_hash, "report.pdf", b"document", "ACME.test")
    second_key = save_source(source_hash, "report.pdf", b"document", "ACME.test")

    assert key.startswith("s3://private-kb/sources/acme.test/")
    assert key == second_key
    assert captured["Bucket"] == "private-kb"
    assert captured["ContentType"] == "application/octet-stream"
    assert captured["Metadata"] == {"sha256": source_hash}
    assert "ACL" not in captured
    assert load_source(key) == b"document"
    assert source_storage_key(source_hash, "report.pdf", "ACME.test") == key
    assert head_source(key)["content_length"] == len(b"document")
    assert create_presigned_source_url(key) == "https://private-r2.example/signed"
    assert captured["presign"]["ClientMethod"] == "get_object"
    assert captured["presign"]["ExpiresIn"] == 300


def test_source_media_type_is_derived_from_a_safe_allow_list():
    assert safe_source_media_type("report.pdf") == "application/pdf"
    assert safe_source_media_type("unsafe.svg") == "application/octet-stream"
    assert source_should_display_inline("report.pdf")
    assert not source_should_display_inline("unsafe.svg")
    # .txt and .md are now served as text/plain and shown inline: they make up most of
    # this corpus, and text/plain plus nosniff cannot be parsed as markup. The rule the
    # allow-list enforces is "nothing a browser will execute", not "nothing textual" —
    # see tests/unit/test_source_media_types.py.
    assert source_should_display_inline("notes.txt")


def test_r2_source_listing_is_limited_to_the_configured_private_prefix(monkeypatch):
    captured = {}

    class Paginator:
        def paginate(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "Contents": [
                        {
                            "Key": "sources/acme.test/aa/orphan.pdf",
                            "LastModified": "timestamp",
                        },
                    ]
                }
            ]

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return Paginator()

    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_PREFIX", "sources")
    monkeypatch.setattr(source_storage, "_s3_client", lambda: Client())

    objects = list_source_objects()

    assert captured == {"Bucket": "private-kb", "Prefix": "sources/"}
    assert objects == [
        {
            "storage_key": "s3://private-kb/sources/acme.test/aa/orphan.pdf",
            "last_modified": "timestamp",
        }
    ]
