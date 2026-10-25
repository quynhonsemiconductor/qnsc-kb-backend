"""Develop, production and the compose files must run the same model at the same size.

They had drifted, and nothing failed loudly: production's Terraform sized the api at 1 vCPU
while the code's EMBEDDING_ONNX_THREADS default is 2, with a db.t4g.micro that develop had
already outgrown; the production compose file (since removed) still defaulted to the 384-wide MiniLM while
the image bakes the 1024-wide e5-large-instruct, so a stack started without overrides raised
"embedding backend returned 1024 dimensions, but EMBEDDING_DIMENSION is 384" on every embed;
its worker had no --concurrency, so each CPU on the host loaded its own ~1.5 GB session; and
.env.example said one embedding thread while the deployed tasks run two.

Production has never served traffic, so develop is the only environment its numbers were
measured on. These tests pin the pairs equal: change a number in both places or in neither.

What is deliberately NOT pinned equal, because it is policy and not size: the min_count
floors (production idles until go-live), Spot for the api, storage and max_count ceilings,
and the durability settings (backups, deletion protection, log retention). A test below
guards those against being "synced" away by accident.

Like test_embedding_config_matches_image.py these read the source files as text: the bug was
a mismatch BETWEEN files, so reading one of them twice would prove nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core.config import Settings

REPO = Path(__file__).parents[2]


def _stack_exists(environment: str) -> bool:
    """The aws-exit destroy (E1, 2026-10-31) strips a stack to its provider blocks
    (product-docs/platform/runbooks/aws-exit-teardown.md). A stripped main.tf declares no
    services, so it drops out of the pinned pairs instead of failing as a broken config;
    the surviving environment keeps its assertions."""
    return 'module "stack"' in (
        REPO / "infra" / "live" / environment / "main.tf"
    ).read_text(encoding="utf-8")


ENVIRONMENTS = tuple(env for env in ("develop", "prod") if _stack_exists(env))


def _tf(environment: str) -> str:
    return (REPO / "infra" / "live" / environment / "main.tf").read_text(
        encoding="utf-8"
    )


def _block(environment: str, name: str) -> str:
    """The body of a two-space-indented `name = { ... }` object in the stack call."""
    match = re.search(rf"(?ms)^  {name} = \{{(.*?)^  \}}", _tf(environment))
    assert match, f"{environment}/main.tf no longer declares `{name} = {{ ... }}`"
    return match.group(1)


def _value(block: str, key: str) -> str:
    # Anchored to the line start so a comment that merely mentions `cpu = ...` cannot match.
    match = re.search(rf'(?m)^\s*{key}\s*=\s*"?([^"\s/]+)"?', block)
    assert match, f"no `{key}` in block"
    return match.group(1)


def _top_level(environment: str, key: str) -> str | None:
    match = re.search(rf'(?m)^  {key}\s*=\s*"([^"]+)"', _tf(environment))
    return match.group(1) if match else None


def _setting_default(name: str):
    return Settings.model_fields[name].default


ENV_EXAMPLE = (REPO / ".env.example").read_text(encoding="utf-8")


# ── Terraform: develop and production ───────────────────────────────────────────


@pytest.mark.parametrize("service", ["api", "worker"])
@pytest.mark.parametrize("key", ["cpu", "memory"])
def test_both_environments_size_each_service_the_same(service, key):
    values = {env: _value(_block(env, service), key) for env in ENVIRONMENTS}
    assert len(set(values.values())) == 1, (
        f"{service}.{key} differs: {values}. Production has never run, so develop's "
        "measured numbers are the only ones there are; change both files or neither."
    )


def test_both_environments_use_the_same_database_class():
    classes = {
        env: _value(_block(env, "rds"), "instance_class") for env in ENVIRONMENTS
    }
    assert len(set(classes.values())) == 1, classes


@pytest.mark.parametrize(
    "key", ["embedding_model", "embedding_version", "embedding_runtime"]
)
def test_both_environments_state_the_same_embedding_setting(key):
    """embedding_runtime is stated in BOTH, not left to the module default in one of them:
    a default that moves would otherwise move one environment and not the other."""
    values = {env: _top_level(env, key) for env in ENVIRONMENTS}
    assert (
        None not in values.values()
    ), f"{key} is not stated in every environment: {values}"
    assert len(set(values.values())) == 1, values


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("service", ["api", "worker"])
def test_onnx_threads_never_exceed_the_vcpus_of_the_task(environment, service):
    """EMBEDDING_ONNX_THREADS is a ceiling tied to the task's cpu (config.py says to raise
    them together). More threads than cores spends the difference on scheduling — which is
    what production's old 1 vCPU api did against a default of 2."""
    vcpus = int(_value(_block(environment, service), "cpu")) / 1024
    assert _setting_default("EMBEDDING_ONNX_THREADS") <= vcpus


def test_syncing_size_did_not_touch_production_policy():
    """Pinning compute equal must not drag the durability and cost-posture settings along."""
    prod_rds = _block("prod", "rds")
    assert _value(prod_rds, "deletion_protection") == "true"
    assert (
        int(_value(prod_rds, "backup_retention_days")) >= 30
    ), "single-AZ is only safe with 30-day point-in-time recovery; see the comment in prod/main.tf"
    assert (
        _value(_block("prod", "api"), "use_spot") == "false"
    ), "a Spot interruption is a dropped request and a broken response mid-answer"
    # Production idles until go-live: floors stay at zero here.
    assert _value(_block("prod", "api"), "min_count") == "0"
    assert _value(_block("prod", "worker"), "min_count") == "0"


# ── .env.example ────────────────────────────────────────────────────────────────


def _env_example(name: str) -> str:
    match = re.search(rf'(?m)^{name}="?([^"\n]+)"?\s*$', ENV_EXAMPLE)
    assert match, f".env.example does not declare {name}"
    return match.group(1)


@pytest.mark.parametrize(
    "name",
    [
        "EMBEDDING_MODEL",
        "EMBEDDING_VERSION",
        "EMBEDDING_RUNTIME",
        "EMBEDDING_ONNX_POOLING",
        "EMBEDDING_ONNX_THREADS",
        "EMBEDDING_MAX_TOKENS",
    ],
)
def test_env_example_embeds_the_way_the_deployed_code_does(name):
    """A local run should embed with what the deployed tasks use. EMBEDDING_ONNX_THREADS was
    1 here for a 0.5 vCPU task while the code default, which production runs, is 2."""
    local_default = {"EMBEDDING_MODEL": "mock", "EMBEDDING_VERSION": "mock-v1"}
    if name in local_default:
        # The active local default is the deterministic mock, so its zero vectors and version never mix with real ones.
        # The real value stays documented as the opt-in, and that one must be what the deployed tasks run.
        assert _env_example(name) == local_default[name]
        assert (
            f'{name}="{_setting_default(name)}"' in ENV_EXAMPLE
        ), f"the documented real {name} in .env.example is not the deployed value"
        return
    assert _env_example(name) == str(_setting_default(name))
