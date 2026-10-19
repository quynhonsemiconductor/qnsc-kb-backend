"""The prod infra apply must read the saved plan before applying it.

The guards live in the ci repo (plan-guard refuses a destroy of a secret, database or parameter group that is not in the
stack's .allow-data-destroy; destructive-guard refuses DNS and identity changes). They only protect a prod apply if the
job runs them on the SAME plan file it then applies, in that order. A bare `tofu apply -auto-approve` reads nothing.

This reads the workflow as text: the job's step names must appear in this order, and the job must not apply without a
saved plan. Parsing YAML is not needed for that, and it keeps the test free of new dependencies.
"""

from __future__ import annotations

import pathlib
import re

WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "infra-apply.yml"


def apply_prod_job(text: str) -> str:
    start = text.index("\n  apply-prod:\n") + 1
    rest = text[start + 1 :]
    nxt = re.search(r"^  \S", rest, flags=re.MULTILINE)
    return text[start : start + 1 + (nxt.start() if nxt else len(rest))]


def step_names(job: str) -> list[str]:
    return re.findall(r"^      - name: (.+)$", job, flags=re.MULTILINE)


ORDER = ["tofu plan (saved as tfplan)", "plan-guard (tfplan)", "destructive-guard (tfplan)", "tofu apply (the saved plan)"]


def in_order(names: list[str]) -> bool:
    if any(step not in names for step in ORDER):
        return False
    positions = [names.index(step) for step in ORDER]
    return positions == sorted(positions)


def test_prod_apply_runs_both_guards_on_the_saved_plan_before_applying():
    assert in_order(step_names(apply_prod_job(WORKFLOW.read_text())))


def test_prod_apply_never_plans_and_applies_in_one_step():
    job = apply_prod_job(WORKFLOW.read_text())
    assert "-auto-approve" not in job
    assert "tofu apply -input=false tfplan" in job


def test_prod_apply_can_read_the_pull_request_labels_the_destructive_guard_needs():
    job = apply_prod_job(WORKFLOW.read_text())
    assert "pull-requests: read" in job


def test_a_guard_missing_from_prod_fails_the_order_check():
    """Mutation check: dropping the plan-guard step, or moving it after the apply, must fail the order check."""
    job = apply_prod_job(WORKFLOW.read_text())
    dropped = job.replace("      - name: plan-guard (tfplan)\n", "      - name: something else\n")
    assert not in_order(step_names(dropped))
    moved_after_apply = job.replace("      - name: plan-guard (tfplan)\n", "      - name: moved\n")
    moved_after_apply += "\n      - name: plan-guard (tfplan)\n"
    assert not in_order(step_names(moved_after_apply))
