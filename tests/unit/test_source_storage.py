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


#: Cloudflare R2 accepts only these as a region hint. An AWS region is not one of them,
#: and R2 rejects the signed request with InvalidRegionName rather than ignoring it.
R2_VALID_REGIONS = {"wnam", "enam", "weur", "eeur", "apac", "oc", "auto"}


def test_the_r2_client_never_signs_with_an_aws_region(monkeypatch):
    """Every R2 call failed in the deployment because of this one value.

    `_s3_client` passed `settings.AWS_REGION or "auto"`. The fallback looks like it
    covers R2, but AWS_REGION is set — legitimately and unavoidably — to the real AWS
    region for Secrets Manager and ECS, so the fallback never fired and R2 answered:

        ClientError: An error occurred (InvalidRegionName) when calling the PutObject
        operation: The region name 'ap-southeast-1' is not valid.

    The regression is only visible with AWS_REGION SET, which is why it survived: with
    it unset the fallback produces the right answer and everything passes.
    """
    captured = {}

    def fake_client(service, **kwargs):
        captured.update(kwargs)
        captured["service"] = service
        return object()

    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(
        settings, "S3_ENDPOINT_URL", "https://account-id.r2.cloudflarestorage.com"
    )
    monkeypatch.setattr(settings, "R2_ACCOUNT_ID", None)
    monkeypatch.setattr(settings, "R2_ACCESS_KEY_ID", "access-key")
    monkeypatch.setattr(settings, "R2_SECRET_ACCESS_KEY", "secret-key")
    # The deployed value, set from var.region by infra/modules/stack/main.tf.
    monkeypatch.setattr(settings, "AWS_REGION", "ap-southeast-1")

    import boto3

    monkeypatch.setattr(boto3, "client", fake_client)
    source_storage._s3_client()

    assert captured["region_name"] in R2_VALID_REGIONS, captured["region_name"]
    assert captured["region_name"] != "ap-southeast-1"
    assert captured["endpoint_url"] == "https://account-id.r2.cloudflarestorage.com"


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


def test_a_backend_failure_is_raised_as_a_catchable_storage_error(monkeypatch):
    """botocore's ClientError is neither FileNotFoundError nor RuntimeError, so it slipped
    through every R2 guard in articles.py and reached the client as a bare 500."""
    from botocore.exceptions import ClientError

    error = ClientError(
        {"Error": {"Code": "InvalidRegionName", "Message": "not valid"}}, "PutObject"
    )

    class Client:
        def put_object(self, **kwargs):
            raise error

    monkeypatch.setattr(settings, "SOURCE_STORAGE_BACKEND", "r2")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_BUCKET", "private-kb")
    monkeypatch.setattr(settings, "SOURCE_STORAGE_PREFIX", "sources")
    monkeypatch.setattr(source_storage, "_s3_client", lambda: Client())

    with pytest.raises(source_storage.SourceStorageError) as raised:
        save_source("a" * 64, "report.pdf", b"document", "acme.test")

    # RuntimeError is what the call sites in articles.py already catch, so subclassing it
    # is what makes them return 503 instead of leaking a 500.
    assert isinstance(raised.value, RuntimeError)
    assert raised.value.__cause__ is error


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
