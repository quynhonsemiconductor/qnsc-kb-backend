"""Stable asyncio loop runner for synchronous worker entry points."""

from __future__ import annotations

import asyncio

_worker_event_loop: asyncio.AbstractEventLoop | None = None


def reset_worker_loop() -> None:
    global _worker_event_loop
    _worker_event_loop = None


def sync_run(coro):
    """Run a coroutine on one event loop per worker process.

    ``asyncio.run`` creates and closes a loop for every task. SQLAlchemy's
    asyncpg pool can retain connections bound to that closed loop, causing the
    next task to fail with ``RuntimeError: Event loop is closed``.
    """
    global _worker_event_loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        raise RuntimeError("Cannot run a worker coroutine inside an active event loop")
    if _worker_event_loop is None or _worker_event_loop.is_closed():
        _worker_event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_event_loop)
    return _worker_event_loop.run_until_complete(coro)
