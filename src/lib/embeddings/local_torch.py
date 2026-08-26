"""In-process embeddings via sentence-transformers — the reference implementation.

This is what the stored corpus was embedded with, so it defines the vector space every
other backend has to match. The runtime default is onnx and no shipped image installs
the `ml` group, so reaching this backend means a local checkout with
`poetry install --with ml` — which is exactly the situation the parity gate in
tests/unit/test_embedding_backends.py re-enters when a model change needs re-proving.
"""
from __future__ import annotations

from typing import Any

import structlog

from src.core.config import settings
from src.lib.embeddings.base import EmbeddingUnavailable, Lazy

logger = structlog.get_logger()


def _load() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        # Do not send the reader to `poetry install --with ml`: the `ml` group is
        # onnxruntime + transformers. Neither torch nor sentence-transformers is declared
        # anywhere in this project, so no install command makes this backend work — the
        # dependency has to be added first, or the ONNX backend used instead. Saying so
        # plainly beats an instruction that silently changes nothing.
        raise EmbeddingUnavailable(
            f"EMBEDDING_RUNTIME=torch needs sentence-transformers (and torch), which this "
            f"project does not declare in any dependency group, so {settings.EMBEDDING_MODEL!r} "
            "cannot be loaded. Either add sentence-transformers to the 'ml' group and "
            "rebuild, or use EMBEDDING_RUNTIME=onnx with an export in EMBEDDING_ONNX_DIR."
        ) from exc

    logger.info("Loading SentenceTransformer model", model=settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info("SentenceTransformer model ready", model=settings.EMBEDDING_MODEL)
    return model


_model = Lazy(_load, "sentence-transformers")


class TorchEmbeddingProvider:
    name = "torch"

    def warm_up(self) -> None:
        _model.get()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings stays on: it is how this backend has always behaved, and
        # the seam's own normalisation is then a no-op rather than a second opinion.
        vectors = _model.get().encode(
            texts, normalize_embeddings=True, batch_size=settings.EMBEDDING_BATCH_SIZE
        )
        return [vector.tolist() for vector in vectors]
