"""Backend · Deploy is develop-only: a tag ref must run nothing.

The reusable selects production when the run's ref is a tag, so `gh workflow run
backend-deploy.yml --ref vX.Y.Z` would otherwise reach the prod promote with only
the reviewer gate in the way — and KB has no prod at all. The `develop-only` job
runs on EVERY trigger and fails the run from a tag ref; everything else `needs` it.

Reads the workflow as text, the way test_infra_apply_plan_guard.py does.
"""
from __future__ import annotations

import pathlib
import re

WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "backend-deploy.yml"

GATED = ("wait-for-infra", "wait-for-ci", "deploy")


def read() -> str:
    return WORKFLOW.read_text()


def job_block(text: str, name: str) -> str:
    start = text.index(f"\n  {name}:\n") + 1
    rest = text[start + 1 :]
    nxt = re.search(r"^  \S", rest, flags=re.MULTILINE)
    return text[start : start + 1 + (nxt.start() if nxt else len(rest))]


def test_the_gate_job_has_no_job_level_if():
    """A job-level `if` SKIPS the gate on the ordinary main push too, and the skip
    cascades through `needs` — which would kill every develop deploy."""
    assert "if:" not in job_block(read(), "develop-only")


def test_the_gate_fails_from_a_tag_ref():
    job = job_block(read(), "develop-only")
    assert "GITHUB_REF_TYPE" in job and "exit 1" in job


def test_every_other_job_needs_the_gate():
    text = read()
    for name in GATED:
        block = job_block(text, name)
        needs = next(line for line in block.splitlines() if line.strip().startswith("needs:"))
        assert "develop-only" in needs, name


def test_no_tag_trigger_and_no_tag_conditional_left():
    text = read()
    assert "refs/tags/" not in text
    assert "ref_type == 'tag'" not in text.replace("if [ \"$GITHUB_REF_TYPE\" = tag ]", "")


def test_mutation_re_adding_the_job_level_if_fails():
    """The exact bug this suite exists to catch: skipping the gate skips everything."""
    text = read()
    poisoned = text.replace(
        "  develop-only:\n    runs-on: ubuntu-latest",
        "  develop-only:\n    if: github.ref_type == 'tag'\n    runs-on: ubuntu-latest",
    )
    assert poisoned != text
    with_gate_if = poisoned
    assert "if:" in job_block(with_gate_if, "develop-only")
    assert not _gate_unconditional(with_gate_if)


def _gate_unconditional(text: str) -> bool:
    return "if:" not in job_block(text, "develop-only")
