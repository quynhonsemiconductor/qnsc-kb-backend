"""Embedding providers.

Two implementations behind one pair of functions, selected by the SHAPE of
EMBEDDING_MODEL — `bge-*`, `minilm` and Sentence Transformers model IDs are loaded in-process,
anything else is sent to an HTTP API:

  local   — ONNX Runtime in-process. The default. Requires onnxruntime + transformers,
            the `ml` dependency group, which the api and worker images therefore both
            install. Local models must publish an `onnx/model.onnx` graph and a standard
            Hugging Face tokenizer at the repository root.
  hosted  — an HTTP embedding API (Gemini's batchEmbedContents shape). Kept working, and
            what the deployment used briefly, but not the default.

WHAT THE LOCAL DEFAULT COSTS. The API embeds the search QUERY on every search, so the
model lives in the API image as well as the worker's: ~2.3 GB of ONNX weights, in a
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
from pathlib import Path
from typing import Any

import httpx
import structlog

from src.core.config import settings

logger = structlog.get_logger()


def _is_mock() -> bool:
    return settings.OPENAI_API_KEY == "mock" or settings.EMBEDDING_MODEL == "mock"


def _uses_local_model() -> bool:
    """Local models are the ones this process would have to load itself."""
    model = settings.EMBEDDING_MODEL.lower()
    return any(
        marker in model for marker in ("bge-", "minilm", "sentence-transformers/")
    )


# ── Local ONNX (optional) ────────────────────────────────────────────────────
class OnnxEmbeddingModel:
    """ONNX inference with the CLS pooling used by BGE-style embedding models."""

    # Keep the download small without putting the tokenizer in the graph directory.
    # Hugging Face model repositories conventionally keep tokenizer assets at their root.
    _MODEL_FILES = (
        "onnx/*",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
    )

    def __init__(self, model_name: str) -> None:
        try:
            from huggingface_hub import snapshot_download
            from onnxruntime import InferenceSession
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require the optional 'ml' dependency group "
                "(onnxruntime, transformers). Install it with `poetry install --with ml`."
            ) from exc

        snapshot_root = Path(
            snapshot_download(repo_id=model_name, allow_patterns=self._MODEL_FILES)
        )
        model_file = snapshot_root / "onnx" / "model.onnx"
        if not model_file.is_file():
            raise RuntimeError(
                f"{model_name!r} does not publish onnx/model.onnx. "
                "Use a model repository with pre-exported ONNX assets."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(snapshot_root)
        self._session = InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        self._input_names = {input_.name for input_ in self._session.get_inputs()}

    def encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        features = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="np"
        )
        # Hugging Face fast tokenizers commonly return int32 NumPy arrays on Windows,
        # while the BGE-M3 ONNX graph declares token inputs as int64.
        inputs = {
            name: np.asarray(features[name], dtype=np.int64)
            for name in self._input_names
            if name in features
        }
        missing_inputs = self._input_names - inputs.keys()
        if missing_inputs:
            raise RuntimeError(
                f"Tokenizer did not provide required ONNX inputs: {sorted(missing_inputs)}"
            )

        # BGE-M3's SentenceTransformer configuration specifies CLS pooling. The ONNX
        # graph returns token embeddings, so retain token 0 and normalize it to preserve
        # the cosine-distance invariant used by pgvector.
        token_embeddings = self._session.run(None, inputs)[0]
        vectors = token_embeddings[:, 0, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("ONNX model returned a zero embedding")
        return (vectors / norms).tolist()


class OnnxEmbeddingModelSingleton:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info(
                        "Initializing ONNX embedding model (loading weights)...",
                        model=settings.EMBEDDING_MODEL,
                    )
                    try:
                        cls._model = OnnxEmbeddingModel(settings.EMBEDDING_MODEL)
                        logger.info("ONNX embedding model loaded successfully.")
                    except Exception as e:
                        logger.error(
                            "Failed to load ONNX embedding model",
                            error=str(e),
                            model=settings.EMBEDDING_MODEL,
                        )
                        raise e
        return cls._model


def _local_embeddings(texts: list[str]) -> list[list[float]]:
    return OnnxEmbeddingModelSingleton.get_model().encode(texts)


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
    url = (
        f"{settings.GEMINI_API_BASE_URL.rstrip('/')}/models/{model}:batchEmbedContents"
    )
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
    # where the full-width 3072 output is unit length. The local ONNX model always
    # normalises its CLS embeddings, and the tuned constants
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
        logger.error(
            "Error generating embedding", error=str(e), model=settings.EMBEDDING_MODEL
        )
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
