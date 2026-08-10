"""Embedding providers.

Two implementations behind one pair of functions, selected by the SHAPE of
EMBEDDING_MODEL — `bge-*`, `minilm` and `sentence-transformers/*` are loaded in-process,
anything else is sent to an HTTP API:

  local   — SentenceTransformer in-process. The default. Requires torch +
            sentence-transformers, the `ml` dependency group, which the api and worker
            images therefore both install.
  hosted  — an HTTP embedding API (Gemini's batchEmbedContents shape). Kept working, and
            what the deployment used briefly, but not the default.

WHAT THE LOCAL DEFAULT COSTS. The API embeds the search QUERY on every search, so the
model lives in the API image as well as the worker's: ~2.3 GB of weights plus torch, in a
container that otherwise needs neither. That drives the image size, the task memory floor
before it can serve a request, the registry bill and the cold-start time — on a spot
instance, where a task can be replaced at any moment, all of it is paid again and again
to embed a few hundred characters. It buys keeping every document and query inside the
deployment, which is why it is the default here.

QUERIES AND DOCUMENTS MUST USE THE SAME MODEL. A query embedded by one model and a chunk
embedded by another are points in unrelated spaces, and the cosine distance between them
is meaningless — the search does not error, it just returns nonsense. So this is one
setting for the whole system, and changing it means re-embedding every chunk.

`EMBEDDING_MODEL = "mock"` short-circuits to a zero vector for tests that do not exercise
retrieval. Never set it in an environment that serves anyone: a zero vector makes an
indexing outage look healthy.
"""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from src.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


def _is_mock() -> bool:
    return settings.OPENAI_API_KEY == "mock" or settings.EMBEDDING_MODEL == "mock"


def _uses_local_model() -> bool:
    """Local models are the ones this process would have to load itself."""
    model = settings.EMBEDDING_MODEL.lower()
    return any(marker in model for marker in ("bge-", "minilm", "sentence-transformers/"))


# ── Local (optional) ─────────────────────────────────────────────────────────
class BGEModelSingleton:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info("Initializing SentenceTransformer model (loading weights)...", model=settings.EMBEDDING_MODEL)
                    try:
                        from sentence_transformers import SentenceTransformer

                        cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                        logger.info("SentenceTransformer model loaded successfully.")
                    except ImportError as exc:
                        # A deployed image does not carry torch. Reaching here means the
                        # configured model is a local one in an environment built for
                        # hosted embeddings — a configuration error, not a runtime fault.
                        raise RuntimeError(
                            f"EMBEDDING_MODEL={settings.EMBEDDING_MODEL!r} needs the optional 'ml' "
                            "dependency group (torch, sentence-transformers), which is not installed. "
                            "Use a hosted model, or install with `poetry install --with ml`."
                        ) from exc
                    except Exception as e:
                        logger.error("Failed to load SentenceTransformer model", error=str(e), model=settings.EMBEDDING_MODEL)
                        raise e
        return cls._model


def _local_embeddings(texts: list[str]) -> list[list[float]]:
    model = BGEModelSingleton.get_model()
    # Normalisation lets cosine distance and dot product agree.
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [vector.tolist() for vector in vectors]


# ── Hosted ───────────────────────────────────────────────────────────────────
def _hosted_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch through the Gemini embedding API.

    Batched in ONE request: the per-request overhead dominates for the short strings
    being embedded, and ingestion submits chunks in bulk.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set, and EMBEDDING_MODEL "
            f"({settings.EMBEDDING_MODEL!r}) is a hosted model."
        )

    model = settings.EMBEDDING_MODEL
    url = f"{settings.GEMINI_API_BASE_URL.rstrip('/')}/models/{model}:batchEmbedContents"
    payload = {
        "requests": [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": text}]},
                # Explicit, because the two are NOT interchangeable: the provider embeds a
                # question and a passage differently, and mixing them degrades retrieval
                # quietly rather than visibly.
                "taskType": "RETRIEVAL_QUERY",
                # gemini-embedding-001 returns 3072 dimensions by default, and pgvector's
                # HNSW index REFUSES to build above 2000 — a limit that would surface as a
                # failed migration, not a configuration error. Ask for the width the
                # column was actually created with.
                "outputDimensionality": settings.EMBEDDING_DIMENSION,
            }
            for text in texts
        ]
    }

    response = httpx.post(
        url,
        params={"key": settings.GEMINI_API_KEY},
        json=payload,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()

    vectors = [item["values"] for item in body.get("embeddings", [])]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    # Truncated vectors come back UNNORMALISED — measured L2 of 0.586 at 768 dimensions,
    # where the full-width 3072 output is unit length. The local model this replaced
    # always normalised (`normalize_embeddings=True`), and the tuned constants
    # VECTOR_DISTANCE_THRESHOLD and RAG_MIN_RELEVANCE_SCORE were chosen against unit
    # vectors, so normalising here keeps the invariant the rest of the pipeline assumes
    # rather than quietly shifting every score.
    return [_normalise(v) for v in vectors]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        # Not recoverable, and not something to paper over with a zero vector: an
        # all-zero embedding matches nothing and poisons retrieval.
        raise RuntimeError("embedding API returned a zero vector")
    return [x / norm for x in vector]


# ── Public surface ───────────────────────────────────────────────────────────
def _embed_batch(texts: list[str]) -> list[list[float]]:
    if _uses_local_model():
        return _local_embeddings(texts)
    return _hosted_embeddings(texts)


def get_bge_embedding(text: str) -> list[float]:
    """Embed one string. Raises on failure — never returns a zero vector.

    The name is kept because it is the seam every caller imports; the implementation is
    no longer necessarily BGE.
    """
    if _is_mock():
        return [0.0] * settings.EMBEDDING_DIMENSION

    try:
        return _embed_batch([text])[0]
    except Exception as e:
        logger.error("Error generating embedding", error=str(e), model=settings.EMBEDDING_MODEL)
        # Never silently insert a zero vector. It makes an indexing outage look
        # healthy and contaminates retrieval with meaningless candidates.
        raise RuntimeError("Embedding generation failed") from e


def get_bge_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch so indexing amortises per-call overhead across chunks."""
    if not texts:
        return []
    if _is_mock():
        return [[0.0] * settings.EMBEDDING_DIMENSION for _ in texts]
    try:
        return _embed_batch(texts)
    except Exception as exc:
        logger.error(
            "Error generating embedding batch",
            error=str(exc),
            batch_size=len(texts),
            model=settings.EMBEDDING_MODEL,
        )
        raise RuntimeError("Embedding batch generation failed") from exc
