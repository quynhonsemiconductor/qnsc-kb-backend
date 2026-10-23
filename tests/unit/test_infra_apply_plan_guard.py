"""The infra apply must read each saved plan before applying it, and refuse prod from anything but a release tag.

The guards live in the ci repo (plan-guard refuses a destroy of a secret, database or parameter group that is not in the
stack's .allow-data-destroy; destructive-guard refuses DNS, identity and CI-role changes). They only protect an apply if
the job runs them on the SAME plan file it then applies, in that order. A bare `tofu apply -auto-approve` reads nothing.

The preflight job refuses a prod apply that is not from a vX.Y.Z tag, before any apply runs.

This reads the workflow as text: step names must appear in order inside a job, and the job must not apply without a
saved plan. Parsing YAML is not needed for that, and it keeps the test free of new dependencies.
"""

from __future__ import annotations

import pathlib
import re

WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "infra-apply.yml"

PROD_ORDER = ["tofu plan (saved as tfplan)", "plan-guard (tfplan)", "destructive-guard (tfplan)", "tofu apply (the saved plan)"]
SHARED_ORDER = ["tofu plan (saved as tfplan)", "destructive-guard (tfplan)", "tofu apply (the saved plan)"]
DEVELOP_ORDER = SHARED_ORDER


def read() -> str:
    return WORKFLOW.read_text()


def job_block(text: str, name: str) -> str:
    start = text.index(f"\n  {name}:\n") + 1
    rest = text[start + 1 :]
    nxt = re.search(r"^  \S", rest, flags=re.MULTILINE)
    return text[start : start + 1 + (nxt.start() if nxt else len(rest))]


def step_names(job: str) -> list[str]:
    return re.findall(r"^      - name: (.+)$", job, flags=re.MULTILINE)


def in_order(names: list[str], order: list[str]) -> bool:
    if any(step not in names for step in order):
        return False
    positions = [names.index(step) for step in order]
    return positions == sorted(positions)


def test_prod_apply_runs_both_guards_on_the_saved_plan_before_applying():
    assert in_order(step_names(job_block(read(), "apply-prod")), PROD_ORDER)


def test_shared_apply_runs_the_destructive_guard_on_the_saved_plan_before_applying():
    """_shared holds the CI OIDC roles (module.iam_oidc), which destructive-guard protects."""
    assert in_order(step_names(job_block(read(), "apply-shared")), SHARED_ORDER)


def test_develop_apply_runs_the_destructive_guard_on_the_saved_plan_before_applying():
    assert in_order(step_names(job_block(read(), "apply-develop")), DEVELOP_ORDER)


def test_every_apply_runs_only_a_saved_plan():
    text = read()
    for name in ("apply-prod", "apply-shared", "apply-develop"):
        job = job_block(text, name)
        assert "-auto-approve" not in job, name
        assert "tofu apply -input=false tfplan" in job, name


def test_prod_apply_can_read_the_pull_request_labels_the_destructive_guard_needs():
    assert "pull-requests: read" in job_block(read(), "apply-prod")


def test_a_guard_missing_from_prod_fails_the_order_check():
    """Mutation check: dropping the plan-guard step, or moving it after the apply, must fail the order check."""
    job = job_block(read(), "apply-prod")
    dropped = job.replace("      - name: plan-guard (tfplan)\n", "      - name: something else\n")
    assert not in_order(step_names(dropped), PROD_ORDER)
    moved_after_apply = job.replace("      - name: plan-guard (tfplan)\n", "      - name: moved\n")
    moved_after_apply += "\n      - name: plan-guard (tfplan)\n"
    assert not in_order(step_names(moved_after_apply), PROD_ORDER)


def test_every_apply_job_waits_for_the_preflight():
    text = read()
    assert "\n  preflight:\n" in text
    assert "    needs: preflight\n" in job_block(text, "apply-shared")
    assert "    needs: apply-shared\n" in job_block(text, "apply-prod")
    assert "    needs: apply-shared\n" in job_block(text, "apply-develop")


def test_prod_runs_only_on_a_dispatch_that_asks_for_prod():
    """The tag trigger is gone (KB has no prod), so apply-prod fires on exactly one
    condition. A dispatch from a tag with target=develop must not reach apply-prod."""
    condition = next(
        line for line in job_block(read(), "apply-prod").splitlines() if line.strip().startswith("if:")
    )
    assert condition.strip() == "if: github.event_name == 'workflow_dispatch' && inputs.target == 'prod'"


def test_preflight_refuses_prod_from_anything_but_a_release_tag():
    text = read()
    preflight = job_block(text, "preflight")
    assert r"^v[0-9]+\.[0-9]+\.[0-9]+$" in preflight
    assert "exit 1" in preflight
