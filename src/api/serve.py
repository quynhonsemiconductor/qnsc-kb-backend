"""uvicorn launcher implementing PLATFORM-CONTRACT.md §3's signal order.

uvicorn's own SIGTERM path stops accepting connections immediately, which skips
the contract's first two steps. `ContractServer` replaces the handler: readiness
answers 503 from the moment the signal lands (`shutdown_state` is shared with
src/api/main.py, where /readyz reads it), the process keeps serving for the
endpoint-removal delay so requests already routed to the pod still land, and
only then does uvicorn close the listeners and wait for in-flight requests.
The lifespan shutdown runs last, inside the same SHUTDOWN_TIMEOUT_MS budget,
and exits 1 if it overruns.

Run as `python -m src.api.serve` — the Dockerfile CMD. Not used by the worker:
ROLE=worker runs Celery, which drains its own queues.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from src.api import main as api
from src.lib.observability import ProbeAccessFilter


class ContractServer(uvicorn.Server):
    def handle_exit(self, sig, frame) -> None:
        state = api.shutdown_state
        if not state.draining:
            # First signal: leave the Service (readiness 503, §3 step 1), then
            # keep serving for the endpoint-removal delay (step 2). uvicorn's
            # default path — `should_exit = True` right away — would close the
            # listeners before that delay, refusing requests already routed here.
            state.begin(api.SHUTDOWN_PLAN, asyncio.get_event_loop().time())
            logging.getLogger("uvicorn.error").info(
                "Shutdown: draining",
                delay_ms=int(api.SHUTDOWN_PLAN.endpoint_delay_seconds * 1000),
                timeout_ms=int(api.SHUTDOWN_PLAN.timeout_seconds * 1000),
            )
            asyncio.get_event_loop().call_later(
                api.SHUTDOWN_PLAN.endpoint_delay_seconds, self._close_listeners
            )
            return
        # A second signal means the operator wants out now: default behaviour.
        super().handle_exit(sig, frame)

    def _close_listeners(self) -> None:
        # §3 step 3: stop accepting connections; uvicorn waits for in-flight
        # requests within `timeout_graceful_shutdown`, then runs the lifespan
        # shutdown, whose cleanup is bounded by the same deadline.
        self.should_exit = True


def main() -> None:
    plan = api.SHUTDOWN_PLAN
    # uvicorn bounds only the in-flight wait, so its share of the budget is
    # whatever the endpoint delay leaves over. The lifespan cleanup is bounded
    # by the same deadline from inside the process (ShutdownState.remaining).
    graceful = max(plan.timeout_seconds - plan.endpoint_delay_seconds, 1.0)
    # §2: probes must not appear in access logs; uvicorn's own logger needs the
    # same treatment the app's request middleware gets.
    logging.getLogger("uvicorn.access").addFilter(ProbeAccessFilter())
    config = uvicorn.Config(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=65,
        timeout_graceful_shutdown=int(graceful),
    )
    ContractServer(config).run()


if __name__ == "__main__":
    main()
