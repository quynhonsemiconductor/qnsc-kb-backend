"""Cross-encoder reranking: an optional, heavier second opinion after hybrid_search.

`rag/reranker.py`'s deterministic keyword scorer is the reranker that always runs -- it
has no model to load, so it is on the request path for every search whether or not this
module is usable. This module is the "real" reranker its own docstring anticipates
("used when a cross-encoder is unavailable"): a joint query-passage model that scores
relevance far more precisely than lexical overlap can, at the cost of needing the
optional `ml` poetry group installed and a model warm enough to be worth the wait.

Same shape as `lib/embeddings/local_torch.py` on purpose -- both are optional
sentence-transformers backends behind a `Lazy` load, both raise a clear, actionable error
when the dependency is missing rather than degrading silently. `CROSS_ENCODER_RERANKER_ENABLED`
gates whether `search_service.py` calls this module at all; a fresh deploy with the flag on
but the dependency missing still fails loud, in logs, and the caller falls back to the
lexical scorer rather than 500ing the search.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence, TypeVar

import structlog

from src.core.config import settings
from src.lib.embeddings.base import Lazy

logger = structlog.get_logger()


class CrossEncoderUnavailable(RuntimeError):
    """The cross-encoder cannot run — a configuration fault, not a transient one."""


def _load() -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise CrossEncoderUnavailable(
            f"CROSS_ENCODER_RERANKER_ENABLED is set but sentence-transformers is not "
            f"installed, so {settings.CROSS_ENCODER_MODEL!r} cannot be loaded. Install "
            "the 'ml' poetry group (`poetry install --with ml`) or turn the flag off."
        ) from exc

    if settings.CROSS_ENCODER_THREADS > 0:
        import torch

        torch.set_num_threads(settings.CROSS_ENCODER_THREADS)
    logger.info("Loading cross-encoder model", model=settings.CROSS_ENCODER_MODEL)
    model = CrossEncoder(settings.CROSS_ENCODER_MODEL)
    logger.info("Cross-encoder model ready", model=settings.CROSS_ENCODER_MODEL)
    return model


_model = Lazy(_load, "cross-encoder")

# One scoring call at a time. A timed-out call cannot be cancelled (it is a thread), so without
# this a slow reranker would stack: each new search starting another multi-second job behind the
# one the previous search already gave up on, until the task is saturated.
_scoring = threading.Lock()


class Attempt:
    """One caller's turn at the reranker. `probe` marks the single call let through after the cooldown."""

    __slots__ = ("probe", "recorded")

    def __init__(self, probe: bool) -> None:
        self.probe = probe
        self.recorded = False


class CircuitBreaker:
    """Stop trying a component that keeps timing out, and try it again later.

    closed -> (threshold consecutive timeouts) -> open -> (cooldown) -> half-open: ONE caller is let
    through; its success closes the breaker, its timeout re-opens it for another cooldown.
    Only timeouts count towards opening. A refusal because a previous call is still running, or an
    exception from the model, say nothing about whether it is too slow.

    THE PROBE MUST ALWAYS BE ACCOUNTED FOR. If the one probe ends in any way other than success or
    timeout (CrossEncoderUnavailable, the "still scoring" refusal, cancellation of the request) and
    nothing clears the half-open state, every later caller is refused forever and the reranker is
    wedged off until the process restarts. So usage is `begin()` ... `finish()` in a `finally`, and
    an un-recorded probe re-opens the circuit for another cooldown. As a second guard a probe that
    has been out longer than the cooldown is considered lost and another is allowed.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._timeouts = 0
        self._opened_at: float | None = None
        self._probing = False
        self._probe_started = 0.0

    def begin(self) -> Attempt | None:
        """None means skip the reranker. Otherwise pass the returned attempt to finish()."""
        with self._lock:
            if self._opened_at is None:
                return Attempt(probe=False)
            now = self._clock()
            cooldown = settings.CROSS_ENCODER_BREAKER_COOLDOWN_SECONDS
            if self._probing:
                if now - self._probe_started < cooldown:
                    return None
                logger.warning(
                    "Cross-encoder probe was never accounted for; allowing another"
                )
            elif now - self._opened_at < cooldown:
                return None
            self._probing = True
            self._probe_started = now
            logger.info("Cross-encoder circuit half-open; letting one search through")
            return Attempt(probe=True)

    def allow(self) -> bool:
        """begin() without the bookkeeping, for callers that only ask. Prefer begin()/finish()."""
        return self.begin() is not None

    def record_timeout(self, attempt: Attempt | None = None) -> None:
        with self._lock:
            if attempt is not None:
                attempt.recorded = True
            self._timeouts += 1
            self._probing = False
            if (
                self._opened_at is not None
                or self._timeouts >= settings.CROSS_ENCODER_BREAKER_THRESHOLD
            ):
                first = self._opened_at is None
                self._opened_at = self._clock()
                logger.warning(
                    (
                        "Cross-encoder circuit open: skipping the reranker"
                        if first
                        else "Cross-encoder still timing out; circuit stays open"
                    ),
                    consecutive_timeouts=self._timeouts,
                    cooldown_seconds=settings.CROSS_ENCODER_BREAKER_COOLDOWN_SECONDS,
                )

    def record_success(self, attempt: Attempt | None = None) -> None:
        with self._lock:
            if attempt is not None:
                attempt.recorded = True
            if self._opened_at is not None:
                logger.info("Cross-encoder recovered; circuit closed")
            self._timeouts = 0
            self._opened_at = None
            self._probing = False

    def finish(self, attempt: Attempt | None) -> None:
        """Always call this, in a `finally`. A no-op unless this attempt was the probe and recorded nothing."""
        if attempt is None or not attempt.probe or attempt.recorded:
            return
        with self._lock:
            if not self._probing:
                return
            self._probing = False
            self._opened_at = self._clock()
            logger.warning(
                "Cross-encoder probe ended without a result; circuit re-opened for another cooldown",
                cooldown_seconds=settings.CROSS_ENCODER_BREAKER_COOLDOWN_SECONDS,
            )

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None


breaker = CircuitBreaker()

_BUDGET_SAMPLE = [
    (
        "what is the leave policy",
        "Employees accrue twelve days of annual leave per year, approved by their manager.",
    ),
    (
        "who approves expenses",
        "Expenses above the limit need approval from the department head before payment.",
    ),
]


def check_budget() -> float:
    """Load the model, time a few pairs, and warn if CROSS_ENCODER_CANDIDATE_POOL pairs cannot fit
    CROSS_ENCODER_TIMEOUT_SECONDS. Returns seconds per pair.

    Measured by running it, not assumed: the same model cost ~0.5 s per pair on 2 vCPU and ~0.47 s on
    4, so the right pool depends on the task and a constant in the code would be a guess. The sample
    passages are short, so this UNDER-estimates real chunks (parent text, titles); a warning here is
    reliable, silence is not a promise.
    """
    model = _model.get()
    # Under the same lock as score(): torch inference is not meant to run concurrently in one process, and
    # a search that arrives while this runs would otherwise either be timed alongside it (a bogus per-pair
    # cost) or contend with it. Waiting is fine here, this runs in a startup thread.
    with _scoring:
        model.predict(
            _BUDGET_SAMPLE
        )  # first call pays one-off allocation; not representative
        started = time.perf_counter()
        model.predict(_BUDGET_SAMPLE)
        per_pair = (time.perf_counter() - started) / len(_BUDGET_SAMPLE)
    pool = settings.CROSS_ENCODER_CANDIDATE_POOL
    expected = per_pair * pool
    if expected > settings.CROSS_ENCODER_TIMEOUT_SECONDS:
        logger.warning(
            "Cross-encoder cannot score its candidate pool inside its timeout on this task: "
            "every reranked search will time out and fall back. Lower CROSS_ENCODER_CANDIDATE_POOL, "
            "raise the timeout, or turn the reranker off.",
            seconds_per_pair=round(per_pair, 3),
            candidate_pool=pool,
            expected_seconds=round(expected, 2),
            timeout_seconds=settings.CROSS_ENCODER_TIMEOUT_SECONDS,
        )
    else:
        logger.info(
            "Cross-encoder budget ok",
            seconds_per_pair=round(per_pair, 3),
            candidate_pool=pool,
            expected_seconds=round(expected, 2),
        )
    return per_pair


def warm_up() -> None:
    """Load the model before traffic arrives. Called at API startup, failure swallowed
    by the caller the same way embeddings.warm_up() is -- a slow first request is
    recoverable, refusing to boot over an optional reranker is not."""
    if not settings.CROSS_ENCODER_RERANKER_ENABLED:
        return
    _model.get()


def score(query: str, passages: list[str]) -> list[float]:
    """Score each passage's relevance to `query`. Higher is more relevant.

    Raises `CrossEncoderUnavailable` on any failure to load or run the model — the
    caller (`search_service.py`) is expected to catch this and fall back to the
    deterministic lexical scorer, not to propagate a 500 for what is an optional
    enhancement.
    """
    if not passages:
        return []
    if not _scoring.acquire(blocking=False):
        raise CrossEncoderUnavailable(
            "cross-encoder is still scoring an earlier search"
        )
    try:
        model = _model.get()
        pairs = [(query, passage) for passage in passages]
        raw_scores = model.predict(pairs)
    except CrossEncoderUnavailable:
        raise
    except Exception as exc:
        raise CrossEncoderUnavailable(f"cross-encoder scoring failed: {exc}") from exc
    finally:
        _scoring.release()
    return [float(value) for value in raw_scores]


T = TypeVar("T")


def reorder_by_cross_encoder(
    ranked: Sequence[T], cross_scores: Sequence[float]
) -> list[T]:
    """Reorder `ranked` by `cross_scores`, highest first. Pure and DB/model-free.

    `ranked` items are opaque and NOT compared to each other -- only `cross_scores` values
    are, via `key=`, which is what makes it safe to pass ORM objects or arbitrary tuples
    here without them needing to support ordering themselves. `cross_scores[i]` must be
    the score for `ranked[i]`; the two are expected to already be the same length and in
    matching order, since both are built from the same candidate list one call site up.
    """
    if len(ranked) != len(cross_scores):
        raise ValueError(
            f"reorder_by_cross_encoder got {len(ranked)} items but "
            f"{len(cross_scores)} scores"
        )
    paired = sorted(zip(cross_scores, ranked), key=lambda item: item[0], reverse=True)
    return [item for _score, item in paired]
