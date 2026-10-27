"""The build the API reports: release number, and the commit read back from the image tag."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from src.core import build_info
from src.core.build_info import commit_from_image, digest_from_image, get_build_info
from src.core.config import settings
from src.version import __version__

SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "ab" * 32
ECS = "http://169.254.170.2/v4/abc"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    build_info.reset_cache()
    monkeypatch.setattr(settings, "APP_COMMIT_SHA", None)
    yield
    build_info.reset_cache()


def _run(coro):
    return asyncio.run(coro)


def test_version_constant_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text("utf-8"), re.M)
    assert match and match.group(1) == __version__


def test_release_please_rewrites_the_constant():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "release-please-config.json").read_text("utf-8"))
    assert "src/version.py" in config["packages"]["."]["extra-files"]
    assert "x-release-please-version" in (root / "src/version.py").read_text("utf-8")


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (f"123.dkr.ecr.ap-southeast-1.amazonaws.com/qnsc-kb-api:sha-{SHA}", SHA),
        (f"repo/qnsc-kb-api:sha-{SHA}@{DIGEST}", SHA),
        ("repo/qnsc-kb-api:sha-abc1234", "abc1234"),
        ("repo/qnsc-kb-api:v1.2.3", None),
        ("repo/qnsc-kb-api:latest", None),
        ("repo/qnsc-kb-api", None),
        ("", None),
        (None, None),
    ],
)
def test_commit_is_read_from_the_image_tag(image, expected):
    assert commit_from_image(image) == expected


def test_digest_is_read_from_the_image_id():
    assert digest_from_image(DIGEST) == DIGEST
    assert digest_from_image(f"repo@{DIGEST}") == DIGEST
    assert digest_from_image("not a digest") is None
    assert digest_from_image(None) is None


def test_explicit_commit_wins_and_is_lowercased(monkeypatch):
    monkeypatch.setattr(settings, "APP_COMMIT_SHA", " ABC1234 ")
    info = _run(get_build_info(ECS))
    assert info.commit == "abc1234"
    assert info.commit_short == "abc1234"
    assert info.version == __version__


def test_outside_ecs_the_commit_is_unknown_not_guessed(monkeypatch):
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
    info = _run(get_build_info())
    assert info.commit is None and info.commit_short is None
    assert info.version == __version__


def test_only_the_link_local_agent_is_ever_fetched(monkeypatch):
    called = []
    monkeypatch.setattr(
        build_info,
        "_fetch_container_image",
        lambda url: called.append(url) or (None, None),
    )
    for url in (
        "http://evil.example/v4",
        "https://169.254.170.2/v4",
        "file:///etc/passwd",
    ):
        info = _run(get_build_info(url))
        assert info.commit is None
    assert called == []


def test_commit_comes_from_the_ecs_image_and_is_cached(monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        return f"reg/qnsc-kb-api:sha-{SHA}", f"reg/qnsc-kb-api@{DIGEST}"

    monkeypatch.setattr(build_info, "_fetch_container_image", fetch)
    first = _run(get_build_info(ECS))
    second = _run(get_build_info(ECS))
    assert first.commit == SHA and first.commit_short == SHA[:7]
    assert first.image_digest == DIGEST
    assert second is first
    assert calls == [ECS]


def test_a_failed_lookup_degrades_and_is_not_retried_every_request(monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        raise OSError("timed out")

    monkeypatch.setattr(build_info, "_fetch_container_image", fetch)
    assert _run(get_build_info(ECS)).commit is None
    assert _run(get_build_info(ECS)).commit is None
    assert len(calls) == 1


def test_an_image_without_a_sha_tag_has_no_commit(monkeypatch):
    monkeypatch.setattr(
        build_info,
        "_fetch_container_image",
        lambda url: ("reg/qnsc-kb-api:v1.2.3", DIGEST),
    )
    info = _run(get_build_info(ECS))
    assert info.commit is None
    assert info.image_digest == DIGEST


def test_public_shape_has_what_the_ui_reads(monkeypatch):
    monkeypatch.setattr(settings, "APP_COMMIT_SHA", SHA)
    data = _run(get_build_info()).public()
    assert set(data) == {
        "service",
        "version",
        "commit",
        "commit_short",
        "environment",
        "started_at",
        "image_digest",
    }
    assert data["service"] == "api"


def test_the_route_needs_a_signed_in_user_and_is_mounted():
    from src.api.routers import system

    route = next(r for r in system.router.routes if r.path == "/version")
    assert "get_current_user" in {d.call.__name__ for d in route.dependant.dependencies}
    # The app wraps included routers, so `app.routes` is not a flat list; read the source,
    # as the other wiring tests here do.
    main = (Path(__file__).resolve().parents[2] / "src/api/main.py").read_text("utf-8")
    assert 'system.router, prefix=f"{settings.API_V1_STR}/system"' in main
