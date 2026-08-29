"""Embeddings: one seam, several backends.

    get_bge_embedding / get_bge_embeddings   the only functions callers use
    warm_up                                  load the weights outside the request path
    resolve_provider                         which backend, and why

WHAT DECIDES THE BACKEND. Two orthogonal settings, deliberately separate:

    EMBEDDING_MODEL    WHICH vectors — identity of the space. Changing it invalidates
                       every stored chunk.
    EMBEDDING_RUNTIME  HOW they are computed — torch or ONNX. Same vectors either way,
                       once parity is proven.

They used to be one decision, inferred from the shape of the model name ("bge-",
"minilm", "sentence-transformers/"). That could not express "bge-m3 on ONNX", and the
sniffing silently sent an unrecognised local model to a hosted API. Model identity and
execution engine are different questions, so they are different settings.

THE INVARIANTS LIVE IN base.finalise, not in the backends: every vector is checked for
width and normalised to unit length in exactly one place. See base.py for why.

`EMBEDDING_MODEL = "mock"` short-circuits to a zero vector for tests that do not exercise
retrieval. Never set it anywhere that serves a person: a zero vector makes an indexing
outage look healthy.
"""
from __future__ import annotations

import structlog

from src.core.config import settings
from src.lib.embeddings.base import EmbeddingProvider, EmbeddingUnavailable, finalise
from src.lib.embeddings.hosted import HostedEmbeddingProvider
from src.lib.embeddings.local_onnx import OnnxEmbeddingProvider
from src.lib.embeddings.local_torch import TorchEmbeddingProvider

__all__ = [
    "EmbeddingUnavailable",
    "get_bge_embedding",
    "get_bge_embeddings",
    "resolve_provider",
    "warm_up",
]

logger = structlog.get_logger()

_LOCAL_MARKERS = ("bge-", "minilm", "sentence-transformers/", "e5-")


def is_mock() -> bool:
    return settings.EMBEDDING_MODEL == "mock"


def _runs_in_process() -> bool:
    """Whether EMBEDDING_MODEL names a model this process loads itself."""
    return any(marker in settings.EMBEDDING_MODEL.lower() for marker in _LOCAL_MARKERS)


def resolve_provider() -> EmbeddingProvider:
    """Pick the backend for the current configuration.

    Cheap and stateless — the expensive part is the weights, which each backend holds in
    its own module-level `Lazy`. Tests patch THIS function to simulate a backend failure;
    it is the seam, so nothing below it needs a test double.
    """
    if not _runs_in_process():
        return HostedEmbeddingProvider()
    if settings.EMBEDDING_RUNTIME == "onnx":
        return OnnxEmbeddingProvider()
    if settings.EMBEDDING_RUNTIME == "torch":
        return TorchEmbeddingProvider()
    raise EmbeddingUnavailable(
        f"EMBEDDING_RUNTIME={settings.EMBEDDING_RUNTIME!r} is not one of 'torch', 'onnx'"
    )


def warm_up() -> None:
    """Load the model before traffic arrives, so no request pays the cold start.

    Called at API startup. Failure is logged and swallowed by the caller: keyword search
    still works without embeddings, and refusing to boot would turn a degraded search
    into a total outage.
    """
    if is_mock():
        return
    resolve_provider().warm_up()


def _embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed through the configured provider.

    `task` matters only to hosted providers, which embed a QUESTION and a PASSAGE
    differently. Mixing them degrades retrieval quietly rather than visibly, so the
    distinction is carried from the two public entry points — singular is a search
    query, plural is a batch of chunks being indexed — rather than guessed further
    down. Local providers ignore it.
    """
    return finalise(resolve_provider().embed(texts, task=task), len(texts))


def get_bge_embedding(text: str) -> list[float]:
    """Embed one string. Raises on failure — never returns a zero vector.

    The name is kept because it is the seam every caller imports; the implementation has
    not been BGE-specific for some time.
    """
    if is_mock():
        return [0.0] * settings.EMBEDDING_DIMENSION
    try:
        return _embed([text], task="RETRIEVAL_QUERY")[0]
    except Exception as exc:
        logger.error(
            "Error generating embedding",
            error=str(exc),
            model=settings.EMBEDDING_MODEL,
            runtime=settings.EMBEDDING_RUNTIME,
        )
        # Never silently insert a zero vector: it makes an indexing outage look healthy
        # and contaminates retrieval with meaningless candidates.
        raise RuntimeError("Embedding generation failed") from exc


def get_bge_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch, so indexing amortises per-call overhead across chunks."""
    if not texts:
        return []
    if is_mock():
        return [[0.0] * settings.EMBEDDING_DIMENSION for _ in texts]
    try:
        return _embed(texts, task="RETRIEVAL_DOCUMENT")
    except Exception as exc:
        logger.error(
            "Error generating embedding batch",
            error=str(exc),
            batch_size=len(texts),
            model=settings.EMBEDDING_MODEL,
            runtime=settings.EMBEDDING_RUNTIME,
        )
        raise RuntimeError("Embedding batch generation failed") from exc
