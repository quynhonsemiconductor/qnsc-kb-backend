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

    logger.info("Loading cross-encoder model", model=settings.CROSS_ENCODER_MODEL)
    model = CrossEncoder(settings.CROSS_ENCODER_MODEL)
    logger.info("Cross-encoder model ready", model=settings.CROSS_ENCODER_MODEL)
    return model


_model = Lazy(_load, "cross-encoder")


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
    try:
        model = _model.get()
        pairs = [(query, passage) for passage in passages]
        raw_scores = model.predict(pairs)
    except CrossEncoderUnavailable:
        raise
    except Exception as exc:
        raise CrossEncoderUnavailable(f"cross-encoder scoring failed: {exc}") from exc
    return [float(value) for value in raw_scores]


T = TypeVar("T")


def reorder_by_cross_encoder(ranked: Sequence[T], cross_scores: Sequence[float]) -> list[T]:
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
