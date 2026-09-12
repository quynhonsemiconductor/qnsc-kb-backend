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


# Models trained with an asymmetric instruction prefix. Base e5 is explicit about it:
# "Each input text should start with 'query: ' or 'passage: ', even for
# non-English texts", and omitting the prefix costs recall SILENTLY -- the
# vectors are still unit-norm and still retrieve something, just worse. bge-m3
# and MiniLM take no prefix, so the table is opt-in per model family.
_INSTRUCTION_PREFIXES = {
    "query": "query: ",
    "passage": "passage: ",
}

# The instruct-tuned e5 variants (multilingual-e5-large-instruct, e5-mistral-7b-instruct,
# gte-Qwen2-*-instruct, ...) use a DIFFERENT convention than base e5, not the same one at
# a bigger size: the query gets a full natural-language task instruction, and the PASSAGE
# gets no prefix at all -- not "passage: ", nothing. Model card: `f"Instruct: {task}\n
# Query: {query}"` for queries, raw text for passages. Applying the base-e5 prefix table
# to an instruct model instead would still produce valid, unit-norm vectors -- just ones
# the model was not tuned to place well, the same silent-degradation failure mode this
# whole prefix seam exists to avoid.
_RETRIEVAL_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"


def _needs_instruction_prefix() -> bool:
    model = settings.EMBEDDING_MODEL.lower()
    # bge-m3 dropped instructions entirely; only the e5 family needs them here.
    return "e5-" in model


def _is_instruct_tuned() -> bool:
    return "instruct" in settings.EMBEDDING_MODEL.lower()


def _decorate(texts: list[str], task: str) -> list[str]:
    """Prepend the model's instruction prefix, if it was trained with one.

    Applied HERE rather than in each backend so the torch and onnx runtimes
    cannot disagree about it -- a query embedded with a prefix and a passage
    embedded without one land in different regions of the space, which degrades
    retrieval with no error to notice.
    """
    if not _needs_instruction_prefix():
        return texts
    if _is_instruct_tuned():
        if task == "RETRIEVAL_QUERY":
            return [f"Instruct: {_RETRIEVAL_INSTRUCTION}\nQuery: {text}" for text in texts]
        return list(texts)
    key = "query" if task == "RETRIEVAL_QUERY" else "passage"
    prefix = _INSTRUCTION_PREFIXES[key]
    return [f"{prefix}{text}" for text in texts]


def _embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed through the configured provider.

    `task` tells a hosted provider to embed a QUESTION differently from a
    PASSAGE, and now also selects the instruction prefix for local models that
    were trained with one (e5). Mixing the two degrades retrieval quietly rather
    than visibly, so the distinction is carried from the two public entry points
    -- singular is a search query, plural is a batch of chunks being indexed --
    rather than guessed further down.
    """
    return finalise(
        resolve_provider().embed(_decorate(texts, task), task=task), len(texts)
    )


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
