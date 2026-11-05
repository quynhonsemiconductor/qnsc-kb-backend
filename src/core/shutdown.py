"""PLATFORM-CONTRACT.md section 3: one ordered, bounded shutdown sequence.

On SIGTERM a kb process must, in this order: answer `/readyz` with 503 (the pod
leaves its Service), keep serving for the endpoint-removal delay so requests
already routed to it still land, and only then stop accepting connections and
wait for in-flight requests. The whole sequence is bounded by
`SHUTDOWN_TIMEOUT_MS`: past the deadline the process exits 1 itself instead of
waiting for the kubelet's SIGKILL. The endpoint-removal delay defaults to 5000
ms in a pod (`KUBERNETES_SERVICE_HOST` set) and 0 elsewhere.

`src/api/serve.py` owns the signal order (uvicorn's own handler stops accepting
connections immediately, which would skip steps 1 and 2); the lifespan shutdown
in `src/api/main.py` reads the deadline this module keeps and bounds its own
cleanup by it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_TIMEOUT_MS = 25000
DEFAULT_DELAY_IN_POD_MS = 5000
#: A drain that gets this close to the deadline stops cleanly at once: the last
#: half second is worth less than a cleanup that hangs the whole grace period.
MIN_CLEANUP_SECONDS = 0.5


class ShutdownConfigurationError(ValueError):
    """The SHUTDOWN_* settings cannot form a workable sequence; refuse to boot."""


@dataclass(frozen=True)
class ShutdownPlan:
    timeout_seconds: float
    endpoint_delay_seconds: float


def in_pod() -> bool:
    """True inside a Kubernetes pod: the env the container runtime itself sets."""
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def _validate_ms(value: int | None, name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ShutdownConfigurationError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ShutdownConfigurationError(f"{name} must be >= 0, got {value}")
    return value


def resolve_shutdown_plan(
    timeout_ms: int | None, delay_ms: int | None, pod: bool
) -> ShutdownPlan:
    """Defaults per the contract; a delay at or above the deadline refuses to boot.

    Malformed values never reach this function: pydantic rejects a non-integer
    environment value before Settings exists, which is the same startup error.
    """
    timeout = _validate_ms(timeout_ms, "SHUTDOWN_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)
    delay = _validate_ms(
        delay_ms, "SHUTDOWN_ENDPOINT_DELAY_MS", DEFAULT_DELAY_IN_POD_MS if pod else 0
    )
    if delay >= timeout:
        raise ShutdownConfigurationError(
            f"SHUTDOWN_ENDPOINT_DELAY_MS ({delay}) must stay below "
            f"SHUTDOWN_TIMEOUT_MS ({timeout}): the drain would never end"
        )
    return ShutdownPlan(
        timeout_seconds=timeout / 1000, endpoint_delay_seconds=delay / 1000
    )


class ShutdownState:
    """Draining flag and the hard deadline, shared by signal handler and lifespan.

    `begin` runs the moment SIGTERM lands -- before uvicorn stops accepting
    connections -- so `/readyz` answers 503 for the whole sequence. The lifespan
    shutdown reads `remaining` so its own cleanup cannot push the process past
    the deadline.
    """

    def __init__(self) -> None:
        self.draining = False
        self.deadline: float | None = None

    def begin(self, plan: ShutdownPlan, now: float) -> None:
        self.draining = True
        self.deadline = now + plan.timeout_seconds

    def remaining(self, now: float) -> float:
        if self.deadline is None:
            return 0.0
        return max(self.deadline - now, 0.0)


#: Patched by tests; a real drain past the deadline must not wait for SIGKILL.
hard_exit = os._exit
