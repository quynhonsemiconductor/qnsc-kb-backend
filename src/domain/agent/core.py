"""Shared plumbing for the agent: a call budget, a step tracer, and tolerant JSON parsing."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger()

#: Receives one server-sent event body, e.g. ``{"type": "step", "step": "search", ...}``.
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


class Budget:
    """How many model calls of its own the agent may still make for one question.

    The final answer call is never counted: it always happens. The budget bounds the extras
    (planning, reformulating, charting, reading a figure), so a bad question cannot make the
    agent spend an unbounded amount of time or money looking for a better one.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max(0, max_calls)
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True

    @property
    def remaining(self) -> int:
        return self.max_calls - self.used


class Tracer:
    """Reports each step to the client as it happens and keeps the record for the answer.

    A sink that raises must never fail the answer, so every report is guarded. The recorded
    steps are stored with the message and shown again when the conversation is reopened.
    """

    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink = sink
        self.steps: list[dict[str, Any]] = []
        self._started: dict[str, float] = {}

    async def emit(
        self, step: str, status: str, detail: str | None = None, **data: Any
    ) -> None:
        now = time.perf_counter()
        if status == "running":
            self._started[step] = now
        event: dict[str, Any] = {"type": "step", "step": step, "status": status}
        if detail:
            event["detail"] = detail
        if data:
            event["data"] = data
        if status != "running":
            started = self._started.pop(step, None)
            record: dict[str, Any] = {"step": step, "status": status}
            if detail:
                record["detail"] = detail
            if data:
                record["data"] = data
            if started is not None:
                record["ms"] = int((now - started) * 1000)
            # A step reported twice (running, then done) is kept once, in its final state.
            self.steps = [item for item in self.steps if item["step"] != step]
            self.steps.append(record)
        if self._sink is None:
            return
        try:
            await self._sink(event)
        except Exception as exc:  # noqa: BLE001 - progress reporting is best effort
            logger.warning("Agent progress event not delivered", error=str(exc))


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """The first JSON object in a model reply, or None.

    Models wrap JSON in code fences or add a sentence before it; both are tolerated. Anything
    that is not an object (a list, a bare string) is rejected, because every caller reads
    named fields from it.
    """
    value = (text or "").strip()
    if not value:
        return None
    value = _FENCE.sub("", value).strip()
    start, end = value.find("{"), value.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
