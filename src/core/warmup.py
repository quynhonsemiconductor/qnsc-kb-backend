"""Whether the embedding model has finished loading, as the readiness probe needs to know it.

Why it exists. The API embeds the search QUERY in-process (ONNX, multilingual-e5-large-instruct),
and loading that model takes seconds on a fast machine and far longer on a cold, shared-CPU task.
The load used to run in the background with nothing recording it: `/health/ready` answered 200 the
moment the database and Redis did, so a deploy was declared healthy, and traffic was sent, to a task
whose first search still had to wait for the model.

WHAT A REQUEST DOES DURING THE LOAD, so nobody mistakes this for a correctness gate: nothing fails.
`Lazy.get()` holds a lock, so a search that arrives mid-load simply waits for it, and the wait is
the remaining load time. This module does not change that. It makes the task SAY it is not ready,
which is what a load balancer or Kubernetes Service acts on. The DEPLOY pipeline polls
`/health/deps` instead (database and Redis only), so a slow model load cannot fail a deploy.

Three terminal outcomes, and only one of them holds the probe:

* ``ready``    the model loaded; the duration is recorded.
* ``failed``   the load raised. Readiness is NOT held: keyword search still works, `local_onnx`
               raises a precise error on every vector search, and holding the probe forever would
               turn a degraded feature into an outage. The probe reports ``degraded`` instead.
* ``timeout``  the load did not finish within ``EMBEDDING_WARMUP_TIMEOUT_SECONDS``. Same reasoning:
               a stuck download must not hold a deploy hostage. The thread keeps loading, so the
               model may still arrive.

Liveness (`/health/live`, `/livez`) never reads this. It must not touch a dependency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

WARMING = "warming"
READY = "ready"
FAILED = "failed"
TIMEOUT = "timeout"
DISABLED = "disabled"  # mock embeddings, or never started: nothing to wait for


@dataclass
class EmbeddingWarmup:
    status: str = DISABLED
    seconds: float | None = None
    error: str | None = None
    _started: float = field(default=0.0, repr=False)

    def begin(self) -> None:
        self.status = WARMING
        self.seconds = None
        self.error = None
        self._started = time.monotonic()

    def finish(self, status: str, error: str | None = None) -> float:
        self.status = status
        self.error = error
        self.seconds = round(time.monotonic() - self._started, 2)
        return self.seconds

    @property
    def holds_readiness(self) -> bool:
        return self.status == WARMING

    @property
    def degraded(self) -> bool:
        return self.status in (FAILED, TIMEOUT)

    def describe(self) -> dict[str, object]:
        out: dict[str, object] = {"state": self.status}
        if self.seconds is not None:
            out["seconds"] = self.seconds
        return out


embedding_warmup = EmbeddingWarmup()
