"""The local compose stack and every CI service container run the same PostgreSQL and Valkey images.

One version per engine: a bump lands in every place at once or fails here. Terraform's engine_version joins
this check in the infra PR that upgrades develop.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
IMAGE_LINE = re.compile(r"(?m)^\s*image:\s*(\S+)\s*$")


def _images(text: str, prefix: str) -> list[str]:
    return [image for image in IMAGE_LINE.findall(text) if image.startswith(prefix)]


def _compose() -> str:
    return (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_postgres_image_is_one_exact_version_in_compose_and_every_workflow():
    pinned = _images(_compose(), "pgvector/pgvector:")
    assert len(pinned) == 1
    assert re.fullmatch(r"pgvector/pgvector:\d+\.\d+\.\d+-pg18", pinned[0])
    for workflow in WORKFLOWS:
        for image in _images(
            workflow.read_text(encoding="utf-8"), "pgvector/pgvector:"
        ):
            assert (
                image == pinned[0]
            ), f"{workflow.name} pins {image}, compose pins {pinned[0]}"


def test_valkey_image_is_one_exact_version_in_compose_and_every_workflow():
    pinned = _images(_compose(), "valkey/valkey:")
    assert len(pinned) == 1
    assert re.fullmatch(r"valkey/valkey:8\.1\.\d+-alpine", pinned[0])
    for workflow in WORKFLOWS:
        for image in _images(workflow.read_text(encoding="utf-8"), "valkey/valkey:"):
            assert (
                image == pinned[0]
            ), f"{workflow.name} pins {image}, compose pins {pinned[0]}"
