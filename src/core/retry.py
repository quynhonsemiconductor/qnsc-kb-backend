from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def with_exponential_retry(
    operation: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 0.5,
    *,
    give_up_on: tuple[type[Exception], ...] = (),
) -> T:
    """Retry an operation, unless the failure says retrying cannot help.

    ``give_up_on`` names the failures where a second attempt is guaranteed to fail the
    same way — a rejected API key, for instance. Retrying those only delays the error
    the caller needs to see.
    """

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except give_up_on:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_error or RuntimeError("Retry operation failed")
