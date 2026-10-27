"""Which build is running: the release number and the commit it was built from.

The release number is a constant in `src/version.py`. The commit is not known to the code:
the deploy pipeline (the shared `ci` reusable) builds the image without build arguments and
tags it `sha-<commit>`, and the ECS task definition points at that tag. So the commit is read
back from the image reference the container is running, which the ECS agent publishes at the
task metadata endpoint. `APP_COMMIT_SHA` overrides it for any host that is not ECS.

Nothing here can fail a request: a metadata endpoint that is absent (local run, tests) or
slow gives `commit=None`, and the answer says "unknown" instead of guessing.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import structlog

from src.core.config import settings
from src.version import __version__

logger = structlog.get_logger()

# `<registry>/qnsc-kb-api:sha-<40 hex>`, optionally followed by `@sha256:<digest>`.
_IMAGE_COMMIT = re.compile(r":sha-([0-9a-f]{7,40})(?:@|$)")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# The ECS agent serves task metadata on a link-local address; anything else in the variable
# is not the agent, and is not fetched.
_METADATA_HOSTS = {"169.254.170.2"}
_METADATA_TIMEOUT_SECONDS = 2.0
_RETRY_AFTER_SECONDS = 60.0

STARTED_AT = datetime.now(timezone.utc)


@dataclass(frozen=True)
class BuildInfo:
    service: str
    version: str
    commit: str | None
    environment: str
    started_at: str
    image_digest: str | None = None

    @property
    def commit_short(self) -> str | None:
        return self.commit[:7] if self.commit else None

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["commit_short"] = self.commit_short
        return data


def commit_from_image(image: str | None) -> str | None:
    """The commit in an image reference tagged `sha-<commit>`; None for any other tag."""
    match = _IMAGE_COMMIT.search(image or "")
    return match.group(1) if match else None


def digest_from_image(image_id: str | None) -> str | None:
    """The `sha256:` digest in an ECS `ImageID` (bare, or `repo@sha256:...`)."""
    match = _DIGEST.search(image_id or "")
    return match.group(0) if match else None


def _metadata_url(raw: str | None) -> str | None:
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme != "http" or parsed.hostname not in _METADATA_HOSTS:
        return None
    return raw


def _fetch_container_image(url: str) -> tuple[str | None, str | None]:
    """`(Image, ImageID)` of this container from the ECS agent.

    Uses `http.client` against the host/port parsed from a URL that
    `_metadata_url` already pinned to the link-local metadata address, so the
    request can never be redirected to another scheme (no `file://`) or host.
    """
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in _METADATA_HOSTS:
        return None, None
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=_METADATA_TIMEOUT_SECONDS
    )
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return None, None
        payload = json.loads(response.read(1_000_000))
    finally:
        connection.close()
    if not isinstance(payload, dict):
        return None, None
    image, image_id = payload.get("Image"), payload.get("ImageID")
    return (
        image if isinstance(image, str) else None,
        image_id if isinstance(image_id, str) else None,
    )


_cache: BuildInfo | None = None
_failed_at: float | None = None


def reset_cache() -> None:
    global _cache, _failed_at
    _cache = None
    _failed_at = None


def _build(commit: str | None, digest: str | None) -> BuildInfo:
    return BuildInfo(
        service="api",
        version=__version__,
        commit=commit,
        environment=settings.ENVIRONMENT,
        started_at=STARTED_AT.isoformat(timespec="seconds"),
        image_digest=digest,
    )


async def get_build_info(metadata_uri: str | None = None) -> BuildInfo:
    """The running build. A found commit is kept for the process's life (an image never
    changes under a running container); a miss is retried at most once a minute."""
    global _cache, _failed_at
    if _cache is not None:
        return _cache

    explicit = (settings.APP_COMMIT_SHA or "").strip().lower()
    if explicit:
        _cache = _build(explicit, None)
        return _cache

    url = _metadata_url(
        metadata_uri
        if metadata_uri is not None
        else os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    )
    if url is None:
        # Not on ECS: nothing to look up, and nothing will appear later.
        return _build(None, None)
    if _failed_at is not None and time.monotonic() - _failed_at < _RETRY_AFTER_SECONDS:
        return _build(None, None)
    try:
        image, image_id = await asyncio.to_thread(_fetch_container_image, url)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - version display must never break a request
        _failed_at = time.monotonic()
        logger.warning(
            "Could not read the running image from ECS metadata", error=str(exc)
        )
        return _build(None, None)
    commit = commit_from_image(image)
    if commit is None:
        _failed_at = time.monotonic()
        return _build(None, digest_from_image(image_id))
    _cache = _build(commit, digest_from_image(image_id))
    return _cache
