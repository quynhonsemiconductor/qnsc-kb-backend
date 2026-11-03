"""The local compose stack and every CI service container run the same PostgreSQL and Valkey images.

One version per engine: a bump lands in every place at once or fails here. Terraform's engine_version joins
this check in the infra PR that upgrades develop.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
IMAGE_LINE = re.compile(r"""(?m)^\s*image:\s*["']?([^\s"']+)["']?\s*$""")


def _tag(image: str) -> str:
    """Compose pins may add a digest (`image:tag@sha256:...`); the version is the part before the @."""
    return image.split("@", 1)[0]


def _images(text: str, prefix: str) -> list[str]:
    return [image for image in IMAGE_LINE.findall(text) if image.startswith(prefix)]


def _compose() -> str:
    return (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def _ci_images(prefix: str) -> list[str]:
    return [
        image
        for workflow in WORKFLOWS
        for image in _images(workflow.read_text(encoding="utf-8"), prefix)
    ]


def test_postgres_image_is_one_exact_version_in_compose_and_every_workflow():
    pinned = _images(_compose(), "pgvector/pgvector:")
    assert len(pinned) == 1
    assert re.fullmatch(r"pgvector/pgvector:\d+\.\d+\.\d+-pg18", _tag(pinned[0]))
    in_ci = _ci_images("pgvector/pgvector:")
    assert in_ci, "no CI workflow runs pgvector: the pin check would pass vacuously"
    for image in in_ci:
        assert _tag(image) == _tag(pinned[0]), f"CI pins {image}, compose pins {pinned[0]}"


def test_valkey_image_is_one_exact_version_in_compose_and_every_workflow():
    pinned = _images(_compose(), "valkey/valkey:")
    assert len(pinned) == 1
    assert re.fullmatch(r"valkey/valkey:8\.1\.\d+-alpine", _tag(pinned[0]))
    in_ci = _ci_images("valkey/valkey:")
    assert in_ci, "no CI workflow runs valkey: the pin check would pass vacuously"
    for image in in_ci:
        assert _tag(image) == _tag(pinned[0]), f"CI pins {image}, compose pins {pinned[0]}"
