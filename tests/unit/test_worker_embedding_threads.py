"""The worker's ONNX thread count must fit its task and agree between environments.

EMBEDDING_ONNX_THREADS is a ceiling tied to the task's vCPU. Too low leaves cores idle through
the whole embedding pass (the worker ran 2 threads on a 4 vCPU task); as many threads as vCPUs
starves clamd, beat and the Celery ping the health check relies on. Like the sibling tests this
reads the Terraform as text, because the failure mode is a mismatch between files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


def _stack_exists(environment: str) -> bool:
    """The aws-exit destroy (E1, 2026-10-31) strips a stack to its provider blocks
    (product-docs/platform/runbooks/aws-exit-teardown.md); a stripped main.tf sets no
    worker size, so it drops out of the pinned pairs instead of failing as a broken config.
    """
    return 'module "stack"' in (
        REPO / "infra" / "live" / environment / "main.tf"
    ).read_text(encoding="utf-8")


ENVIRONMENTS = tuple(env for env in ("develop", "prod") if _stack_exists(env))


def _tf(environment: str) -> str:
    return (REPO / "infra" / "live" / environment / "main.tf").read_text(
        encoding="utf-8"
    )


def _threads(environment: str) -> int:
    match = re.search(
        r"(?m)^  worker_embedding_threads\s*=\s*(\d+)\s*$", _tf(environment)
    )
    assert match, f"{environment}/main.tf does not set worker_embedding_threads"
    return int(match.group(1))


def _worker_vcpus(environment: str) -> float:
    block = re.search(r"(?ms)^  worker = \{(.*?)^  \}", _tf(environment))
    assert block, f"{environment}/main.tf no longer declares `worker = {{ ... }}`"
    cpu = re.search(r"(?m)^\s*cpu\s*=\s*(\d+)", block.group(1))
    assert cpu, "no cpu in the worker block"
    return int(cpu.group(1)) / 1024


def test_environments_use_the_same_worker_thread_count() -> None:
    assert len({_threads(env) for env in ENVIRONMENTS}) == 1


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_threads_leave_a_vcpu_for_the_rest_of_the_task(environment: str) -> None:
    threads, vcpus = _threads(environment), _worker_vcpus(environment)
    assert threads >= 1
    assert threads <= vcpus - 1, (
        f"{threads} threads on a {vcpus:g} vCPU worker leaves nothing for clamd, beat and the "
        "Celery ping that the health check depends on"
    )


def test_the_module_applies_it_to_the_worker_only() -> None:
    stack = (REPO / "infra" / "modules" / "stack" / "main.tf").read_text(
        encoding="utf-8"
    )
    assert stack.count("EMBEDDING_ONNX_THREADS") == 1
    assert "var.worker_embedding_threads" in stack
