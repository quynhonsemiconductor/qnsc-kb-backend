"""Private Cloudflare R2 source storage.

R2 exposes an S3-compatible API, so the implementation uses boto3 as the
client.  The application intentionally does not retain a local-disk backend:
source retrieval must always pass through the authorized API route.
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

from src.core.config import is_cloudflare_r2_endpoint, settings

#: Types a browser may be handed for an ORIGINAL upload, chosen for being inert as
#: documents: images cannot script, a PDF is drawn by the browser's own isolated viewer,
#: and text/plain is never parsed as markup — the responses also carry
#: ``X-Content-Type-Options: nosniff``, so a .md full of HTML stays text.
#:
#: Markdown and text are here because most of this corpus is exactly that, and without
#: them every such source came back as an undisplayable octet-stream download.
_SAFE_SOURCE_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".csv": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
}


class SourceStorageError(RuntimeError):
    """The storage backend refused or could not complete the request.

    Subclasses RuntimeError deliberately. The R2 call sites in articles.py already catch
    ``(FileNotFoundError, RuntimeError)`` for CONFIGURATION failures raised by
    ``_s3_client``, but botocore's ClientError is neither, so a backend failure escaped
    every one of those guards as a bare 500 — which is exactly how an InvalidRegionName
    on every upload reached the browser as an opaque error. Raising this instead means
    those existing guards cover backend failures too, without editing them.
    """


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "source.bin"


def safe_source_media_type(filename: str | None) -> str:
    """Return an allow-listed type for an original source response.

    Multipart ``Content-Type`` is supplied by the uploader. Never reuse it
    for a browser response: an HTML type would execute in the application's
    origin when a reviewer opens the original source.
    """
    return _SAFE_SOURCE_MEDIA_TYPES.get(
        Path(filename or "").suffix.lower(), "application/octet-stream"
    )


def source_should_display_inline(filename: str | None) -> bool:
    return safe_source_media_type(filename) != "application/octet-stream"


def _is_object_storage() -> bool:
    return settings.SOURCE_STORAGE_BACKEND.lower() in {"r2", "cloudflare_r2"}


def _safe_namespace(company_domain: str | None) -> str:
    value = re.sub(
        r"[^A-Za-z0-9._-]+", "_", (company_domain or "local").strip().lower()
    )
    return value[:120] or "local"


def save_source(
    source_hash: str, filename: str, data: bytes, company_domain: str | None = None
) -> str:
    """Store a source privately; callers expose it only through authorized API routes."""
    if not _is_object_storage():
        raise RuntimeError(
            "Cloudflare R2 is required; local-disk source storage is disabled"
        )
    return _s3_put(source_hash, filename, data, company_domain)


def source_storage_key(
    source_hash: str, filename: str, company_domain: str | None = None
) -> str:
    """Return the validated internal R2 URI used by a source upload."""
    if not _is_object_storage():
        raise RuntimeError(
            "Cloudflare R2 is required; local-disk source storage is disabled"
        )
    return f"s3://{settings.SOURCE_STORAGE_BUCKET}/{_s3_key(source_hash, filename, company_domain)}"


def source_path(storage_key: str) -> Path:
    raise FileNotFoundError("R2-backed source has no local filesystem path")


def delete_source(storage_key: str) -> None:
    """Best-effort deletion for rejected drafts and failed transactions."""
    if not storage_key.startswith("s3://"):
        raise FileNotFoundError("Invalid R2 source key")
    prefix = "s3://"
    bucket, _, key = storage_key[len(prefix) :].partition("/")
    expected_prefix = f"{settings.SOURCE_STORAGE_PREFIX.strip('/')}/"
    if bucket != settings.SOURCE_STORAGE_BUCKET or not key.startswith(expected_prefix):
        raise FileNotFoundError("Invalid R2 source key")
    _s3_client().delete_object(Bucket=bucket, Key=key)


def list_source_objects() -> list[dict[str, Any]]:
    """List private source objects for the scheduled orphan sweep.

    Only the configured source prefix is enumerated. Callers must still
    compare each returned key with database references and apply a grace
    period before deleting anything.
    """
    if not _is_object_storage():
        raise RuntimeError(
            "Cloudflare R2 is required; local-disk source storage is disabled"
        )
    bucket, key_prefix = _validated_storage_key(
        f"s3://{settings.SOURCE_STORAGE_BUCKET}/{settings.SOURCE_STORAGE_PREFIX.strip('/')}/_prefix"
    )
    paginator = _s3_client().get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(
        Bucket=bucket, Prefix=key_prefix.rsplit("/", 1)[0] + "/"
    ):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if key:
                objects.append(
                    {
                        "storage_key": f"s3://{bucket}/{key}",
                        "last_modified": item.get("LastModified"),
                    }
                )
    return objects


def create_presigned_source_url(
    storage_key: str, *, operation: str = "get_object", expires_in: int = 300
) -> str:
    """Create a short-lived URL for a validated private R2 object.

    The bucket remains private; callers must first authorize the Article. Only
    GET and PUT object operations are exposed by this helper, and the object
    key must remain under the configured tenant prefix.
    """
    if operation not in {"get_object", "put_object"}:
        raise ValueError("Unsupported R2 presign operation")
    if not 1 <= expires_in <= 900:
        raise ValueError("expires_in must be between 1 and 900 seconds")
    if not storage_key.startswith("s3://"):
        raise FileNotFoundError("Invalid R2 source key")
    bucket, _, key = storage_key[len("s3://") :].partition("/")
    expected_prefix = f"{settings.SOURCE_STORAGE_PREFIX.strip('/')}/"
    if bucket != settings.SOURCE_STORAGE_BUCKET or not key.startswith(expected_prefix):
        raise FileNotFoundError("Invalid R2 source key")
    return _s3_client().generate_presigned_url(
        ClientMethod=operation,
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def head_source(storage_key: str) -> dict[str, Any]:
    """Read object metadata without making the object publicly reachable."""
    bucket, key = _validated_storage_key(storage_key)
    response = _s3_client().head_object(Bucket=bucket, Key=key)
    return {
        "content_length": int(response.get("ContentLength") or 0),
        "metadata": response.get("Metadata") or {},
        "etag": response.get("ETag"),
    }


def _s3_client() -> Any:
    bucket = (settings.SOURCE_STORAGE_BUCKET or "").strip()
    access_key = (settings.R2_ACCESS_KEY_ID or "").strip()
    secret_key = (settings.R2_SECRET_ACCESS_KEY or "").strip()
    if not bucket:
        raise RuntimeError("SOURCE_STORAGE_BUCKET is required for object storage")
    if not access_key or not secret_key:
        raise RuntimeError(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required for Cloudflare R2"
        )
    endpoint_url = (settings.S3_ENDPOINT_URL or "").strip()
    account_or_endpoint = (settings.R2_ACCOUNT_ID or "").strip().rstrip("/")
    if not endpoint_url and account_or_endpoint:
        # Accept the dashboard's account ID as well as a full endpoint pasted
        # into the setting.  The latter is common in local .env files.
        endpoint_url = (
            account_or_endpoint
            if account_or_endpoint.startswith(("https://", "http://"))
            else f"https://{account_or_endpoint}.r2.cloudflarestorage.com"
        )
    if not endpoint_url or not is_cloudflare_r2_endpoint(endpoint_url):
        raise RuntimeError("A Cloudflare R2 endpoint is required for object storage")
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3-backed source storage") from exc
    kwargs: dict[str, Any] = {
        # "auto", NEVER settings.AWS_REGION. This client only ever talks to Cloudflare
        # R2 — the endpoint check above rejects anything else — and R2 accepts only its
        # own region hints: wnam, enam, weur, eeur, apac, oc, auto. An AWS region is not
        # one of them, so R2 rejects the signed request outright:
        #
        #   ClientError: An error occurred (InvalidRegionName) when calling the
        #   PutObject operation: The region name 'ap-southeast-1' is not valid.
        #
        # This read `settings.AWS_REGION or "auto"`, and the fallback looks like it
        # covers this — but AWS_REGION is set, legitimately and unavoidably, to the real
        # AWS region for Secrets Manager and ECS (infra sets it from var.region). So the
        # fallback never fired in a deployment and every R2 call failed. Not just
        # uploads: this client backs put, get, delete, head, list and presign.
        "region_name": "auto",
        "endpoint_url": endpoint_url,
    }
    kwargs.update(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return boto3.client("s3", **kwargs)


def _s3_key(source_hash: str, filename: str, company_domain: str | None = None) -> str:
    # The object remains private, so deterministic naming is safe and makes
    # retries/idempotent connector syncs address the same object. The tenant,
    # hash shard, and content hash prevent cross-tenant collisions.
    normalized_hash = re.sub(r"[^A-Fa-f0-9]", "", source_hash).lower()
    if len(normalized_hash) != 64:
        raise ValueError("source_hash must be a 64-character SHA-256 hex digest")
    return f"{settings.SOURCE_STORAGE_PREFIX.strip('/')}/{_safe_namespace(company_domain)}/{normalized_hash[:2]}/{normalized_hash}_{_safe_name(filename)}"


def _s3_put(
    source_hash: str, filename: str, data: bytes, company_domain: str | None = None
) -> str:
    from botocore.exceptions import BotoCoreError, ClientError

    key = _s3_key(source_hash, filename, company_domain)
    try:
        _s3_client().put_object(
            Bucket=settings.SOURCE_STORAGE_BUCKET,
            Key=key,
            Body=data,
            ContentType="application/octet-stream",
            Metadata={"sha256": source_hash},
        )
    except (BotoCoreError, ClientError) as exc:
        raise SourceStorageError(f"R2 rejected the source upload: {exc}") from exc
    return f"s3://{settings.SOURCE_STORAGE_BUCKET}/{key}"


def _validated_storage_key(storage_key: str) -> tuple[str, str]:
    if not storage_key.startswith("s3://"):
        raise FileNotFoundError("Invalid R2 source key")
    bucket, _, key = storage_key[len("s3://") :].partition("/")
    expected_prefix = f"{settings.SOURCE_STORAGE_PREFIX.strip('/')}/"
    if (
        not bucket
        or not key
        or bucket != settings.SOURCE_STORAGE_BUCKET
        or not key.startswith(expected_prefix)
    ):
        raise FileNotFoundError("Invalid R2 source key")
    return bucket, key


def load_source(storage_key: str) -> bytes:
    bucket, key = _validated_storage_key(storage_key)
    response = _s3_client().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()
