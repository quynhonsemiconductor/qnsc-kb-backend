"""The contract every embedding backend meets, and the invariants enforced once.

Two rules hold for every vector this package returns, whoever produced it:

  WIDTH — it must be exactly EMBEDDING_DIMENSION long. pgvector fixes the width in the
  column type, so a wrong width is not a soft failure: the insert raises, mid-ingest,
  once per chunk. Checking here turns that into one clear error at the source.

  UNIT LENGTH — it must be L2-normalised. VECTOR_DISTANCE_THRESHOLD,
  RAG_MIN_RELEVANCE_SCORE and RAG_MIN_CONTEXT_SCORE were all tuned against unit vectors,
  and cosine distance only agrees with dot product when they are. A backend that returns
  unnormalised vectors does not error, it silently shifts every score in the system.

They are applied HERE rather than in each backend because that is exactly how they came
apart before: SentenceTransformer normalised via its own `normalize_embeddings=True`
flag, the hosted API path normalised by hand, and the two drifted. Normalising an already
unit-length vector is a no-op, so backends may keep their own flag as an optimisation
without changing the result.
"""
from __future__ import annotations

import math
import threading
from typing import Callable, Protocol, TypeVar

from src.core.config import settings


class EmbeddingProvider(Protocol):
    """A source of raw vectors. Normalisation and width checks are NOT its job."""

    name: str

    def embed(
        self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Return one vector per input, in order. Raise on any failure."""
        ...

    def warm_up(self) -> None:
        """Load whatever is expensive, so the first request does not pay for it."""
        ...


class EmbeddingUnavailable(RuntimeError):
    """The configured backend cannot run — a configuration fault, not a transient one."""


def finalise(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    """Apply the two invariants to a backend's raw output."""
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"embedding backend returned {len(vectors)} vectors for {expected_count} inputs"
        )
    return [_unit(vector) for vector in vectors]


def _unit(vector: list[float]) -> list[float]:
    width = settings.EMBEDDING_DIMENSION
    if len(vector) != width:
        raise RuntimeError(
            f"embedding backend returned {len(vector)} dimensions, but EMBEDDING_DIMENSION "
            f"is {width} and the pgvector column is fixed at that width"
        )
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        # Never repaired into a zero vector: one matches nothing, so an indexing outage
        # would look healthy while quietly poisoning retrieval with meaningless
        # candidates. See test_embedding_failure_is_not_converted_to_zero_vector.
        raise RuntimeError("embedding backend returned a zero vector")
    return [value / norm for value in vector]


T = TypeVar("T")


class Lazy:
    """A thread-safe, load-once holder.

    Both in-process backends need exactly this and nothing more: the weights are
    expensive, several request threads may arrive at once, and a failed load must not be
    cached as success. Sharing one implementation keeps the double-checked locking in a
    single place instead of once per backend.
    """

    def __init__(self, load: Callable[[], T], description: str) -> None:
        self._load = load
        self._description = description
        self._value: T | None = None
        self._lock = threading.Lock()

    @property
    def description(self) -> str:
        return self._description

    def get(self) -> T:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._load()
        return self._value

    def loaded(self) -> bool:
        return self._value is not None
