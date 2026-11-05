"""The prod promote retags the same images this repository publishes, from the release commit.

kb prod is promoted by two repositories into one gitops values file: this repository retags
`kb-api` / `kb-worker` / `kb-migrator` from its own release tag, and `qnsc-kb-frontend` retags
`kb-web` from its own. Each repository's promote PR therefore has to carry *its* services and
nobody else's — a mismatch is not a style problem: `release-retag` fails the whole release when a
listed image was not published for the release commit (that is the "no partial promotions" rule),
and a service published here but missing from the promote list would simply never reach prod.

These are cheap static checks on the two workflow files, in the same spirit as
`test_local_compose.py`: they catch an edit to one workflow that forgets the other, without a
live GitHub run.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# What this repository builds. The Dockerfile has one stage per service; both workflows list them
# by name, and the promote list is what a release actually moves to prod.
BACKEND_SERVICES = ["api", "worker", "migrator"]


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _published_services() -> list[str]:
    """The `services-json` GHCR · Publish builds, in its declared order."""
    published = re.search(
        r"(?m)^\s+services-json: '(\[.*\])'$", _read("ghcr-publish.yml")
    )
    assert published, "ghcr-publish.yml has no services-json input"
    services = json.loads(published.group(1))
    assert all(isinstance(service, str) for service in services)
    return services


def _promoted_services() -> list[str]:
    """The `services:` input the promote job passes to the ci reusable, in its declared order."""
    promoted = re.search(r"(?m)^\s{6}services: (\S.*)$", _read("release-promote.yml"))
    assert promoted, "release-promote.yml passes no services input"
    return promoted.group(1).split()


def test_publish_and_promote_name_the_same_services():
    assert _published_services() == BACKEND_SERVICES
    assert _promoted_services() == BACKEND_SERVICES


def test_the_spa_is_promoted_by_its_own_repository():
    # The SPA image is built from qnsc-kb-frontend's commit; this repository's sha never
    # published a kb-web digest, so listing `web` here would fail every release at retag.
    assert "web" not in _published_services()
    assert "web" not in _promoted_services()
    assert "qnsc-kb-frontend" in _read("release-promote.yml")


def test_a_release_tag_retags_the_published_commit_after_waiting_for_its_publish():
    promote = _read("release-promote.yml")
    # The tag names the release; the commit that is retagged is the one the tag points at, whose
    # GHCR · Publish run the first job waits for. Without the wait, retag races the build.
    assert "release-tag: ${{ github.ref_name }}" in promote
    assert "commit: ${{ github.sha }}" in promote
    assert "workflow: ghcr-publish.yml" in promote
    assert re.search(r"(?m)^\s+tags: \['v\[0-9\]\+\.\[0-9\]\+\.\[0-9\]\+'\]$", promote)


def test_the_gate_is_the_repository_variable_and_skips_the_whole_run():
    promote = _read("release-promote.yml")
    # A gate on the promote job alone would still start (and wait on) the publish job. The first
    # job carries it, so an unarmed repository runs nothing at all for a release tag.
    assert re.search(
        r"(?ms)^  wait-for-publish:\n    if: \$\{\{ vars\.KB_PROD_PROMOTE_ENABLED == 'true' \}\}",
        promote,
    ), "the KB_PROD_PROMOTE_ENABLED gate must sit on wait-for-publish"
    assert "needs: wait-for-publish" in promote


def test_the_reusable_is_pinned_at_a_release_with_the_per_source_repository_branch():
    # ci#305 (v1.34.1) names the promotion branch per source repository. kb is the product that
    # needed it: with the older shared `promote/kb-<tag>` branch, the second repository's run
    # either found the first's PR and exited green without promoting, or failed on the branch.
    promote = _read("release-promote.yml")
    pinned = re.search(
        r"uses: quynhonsemiconductor/ci/\.github/workflows/release-promote\.yml@"
        r"([0-9a-f]{40})\s+# v(\d+)\.(\d+)\.(\d+)",
        promote,
    )
    assert (
        pinned
    ), "release-promote.yml must pin the reusable by sha and record its release"
    major, minor, patch = (int(part) for part in pinned.groups()[1:])
    assert (major, minor, patch) >= (1, 34, 1)
