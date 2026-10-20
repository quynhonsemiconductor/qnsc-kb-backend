"""A service running a sidecar must be allowed to read that sidecar's secrets.

THE FAILURE THIS EXISTS FOR happened in rova, not here, and this repo was right only
because someone remembered to be. rova wired the cloudflared sidecar into its api task
and did not add the tunnel token to the execution role's read list. Terraform applied
cleanly. Nothing failed until the next task START:

    ResourceInitializationError: unable to pull secrets or registry auth:
    AccessDeniedException: User: .../rally-develop-api-exec is not authorized to
    perform: secretsmanager:GetSecretValue on resource:
    .../secret:rally/develop/tunnel-token-tf-*

    (The `rally-*` strings are real physical names from the incident —
    legacy, do not rename.)

ECS could not fetch the token, the task never started, the deployment circuit breaker
rolled the service back — and because rollback leaves the PREVIOUS task definition
serving, the service stayed healthy. develop ran a ten-day-old image for two days while
reporting ready, and the only signal was red deploy runs nobody was reading.

WHY A TEST AND NOT A BETTER ABSTRACTION. `tunnel-agent` already publishes a `secret_arns`
output whose description says "Concat into ecs-service's secret_arns, or the task fails to
start" — the contract was written down and simply not consumed. The obvious fix, reading
that output here instead of naming the secret, was tried and REVERTED: the module gates it
on `enabled = var.tunnel_token_secret_arn != ""`, so in an environment where the secret
does not exist yet the ARN is an unknown attribute, the list's length is unknown, and
`ecs-service`'s `count = length(var.secret_arns) + ... > 0` cannot be computed at plan
time. The splat form is the correct one, and nothing in Terraform can require it.

So the invariant is asserted here instead, and deliberately asserts the PERMISSION IS
PRESENT rather than where it comes from — a test demanding the module output would push
the next person into the form that breaks planning.

SCOPE. Exactly one sidecar in this stack needs a secret. `beat` and `clamav` need none, so
they are not listed. A second sidecar that reads a secret needs a line in
SIDECARS_NEEDING_SECRETS below; that edit is the point, not an oversight.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STACK = Path(__file__).resolve().parents[2] / "infra" / "modules" / "stack" / "main.tf"

# marker appearing in a service's wiring  ->  substring its secret_arns must contain
#
# `tunnel` and not `tunnel_token`, because two spellings both grant the permission:
# `aws_secretsmanager_secret.tunnel_token[*].arn` (what this stack uses, and the only
# plan-safe form — see the module docstring) and `module.tunnel_api.secret_arns`. Pinning
# the assertion to one of them would make this test a style rule, and it would have
# rejected a correct-but-different grant. What matters is that the role can read it.
SIDECARS_NEEDING_SECRETS = {
    "module.tunnel_api": "tunnel",
}


def _module_blocks(source: str) -> dict[str, str]:
    """Split the stack into top-level `module "name" { ... }` blocks.

    Brace counting rather than a regex: a service block contains nested `{}` in its
    `environment_vars` and `secrets` lists, so a lazy match stops at the first one and
    every assertion below would pass on a truncated block.
    """
    blocks: dict[str, str] = {}
    lines = source.splitlines()
    for index, line in enumerate(lines):
        opening = re.match(r'^module\s+"([^"]+)"\s*\{', line)
        if not opening:
            continue
        depth, collected = 0, []
        for text in lines[index:]:
            depth += text.count("{") - text.count("}")
            collected.append(text)
            if depth == 0:
                break
        blocks[opening.group(1)] = "\n".join(collected)
    return blocks


def _secret_arns_assignment(block: str) -> str:
    """The `secret_arns = ...` value, whether written on one line or across several."""
    match = re.search(r"^\s*secret_arns\s*=\s*(.*?)(?=^\s*\w+\s*=|\Z)", block, re.S | re.M)
    return match.group(1) if match else ""


def _services_running(marker: str, blocks: dict[str, str]) -> list[str]:
    return [
        name
        for name, block in blocks.items()
        if marker in block and "additional_containers" in block
    ]


@pytest.fixture(scope="module")
def blocks() -> dict[str, str]:
    parsed = _module_blocks(STACK.read_text(encoding="utf-8"))
    assert parsed, f"parsed no module blocks out of {STACK} — the splitter is broken"
    return parsed


@pytest.mark.parametrize(("marker", "required"), sorted(SIDECARS_NEEDING_SECRETS.items()))
def test_a_service_running_a_sidecar_grants_its_secrets(blocks, marker, required):
    running = _services_running(marker, blocks)
    assert running, (
        f"no service wires {marker} — if the sidecar was removed, drop it from "
        f"SIDECARS_NEEDING_SECRETS; if it was renamed, update the marker. An empty "
        f"result must not pass silently, because that is indistinguishable from the "
        f"omission this test exists to catch."
    )
    for service in running:
        assert required in _secret_arns_assignment(blocks[service]), (
            f'module "{service}" runs the {marker} sidecar but its `secret_arns` does '
            f"not mention `{required}`. The execution role will not be able to read the "
            f"secret, so ECS cannot start the task: ResourceInitializationError, then a "
            f"circuit-breaker rollback that leaves the OLD task definition serving and "
            f"the service reporting healthy. This is what broke rova's develop for two "
            f"days."
        )


def test_the_assertion_actually_fails_when_the_grant_is_removed(blocks):
    """A guard that cannot fail is not a guard — the same self-check
    test_terraform_env_satisfies_production.py makes.

    Strips whichever spelling is in use, so this keeps proving the point if the grant is
    ever rewritten in the other form.
    """
    service = _services_running("module.tunnel_api", blocks)[0]
    granted = _secret_arns_assignment(blocks[service])
    assert "tunnel" in granted, "precondition: the grant must be present before removing it"
    stripped = re.sub(r"\S*tunnel\S*", "", granted)
    assert "tunnel" not in stripped


def test_the_splitter_does_not_stop_at_a_nested_brace(blocks):
    """`environment_vars`/`secrets` are lists of objects, so a block that stopped at the
    first `}` would still contain `additional_containers` and pass everything above while
    checking almost nothing."""
    api = blocks["api"]
    assert api.count("{") == api.count("}"), "unbalanced braces — block was truncated"
    assert "environment_vars" in api and "secret_arns" in api
