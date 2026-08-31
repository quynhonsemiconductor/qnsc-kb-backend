"""Cross-encoder reranking: re-score the retrieved candidates by joint relevance.

WHY THIS EXISTS. Retrieval returns a candidate set via hybrid dense+sparse fusion,
but the order that reaches the reader is decided by src/rag/reranker.py, a
deterministic LEXICAL scorer (term overlap). On MLQA / UIT-ViQuAD the question
rarely shares wording with the answer passage, so that lexical order ranks the
correct passage low -- iteration 4 measured fusion-weight changes as byte-identical
because this lexical stage discards the fused order entirely.

This package adds the CPU-only cross-encoder alternative: a multilingual model
that scores each (query, passage) pair jointly. It is opt-in behind
RERANKER_BACKEND, defaulting to the lexical scorer so nothing changes until a
deployment sets it. The seam mirrors src/lib/reader and src/lib/embeddings.
"""
from __future__ import annotations

from src.lib.reranker.base import CrossEncoderReranker, RerankerUnavailable, Scored

__all__ = ["CrossEncoderReranker", "RerankerUnavailable", "Scored", "resolve_reranker"]


def resolve_reranker() -> CrossEncoderReranker:
    """Return the configured cross-encoder reranker backend.

    Only the ONNX backend exists today; the indirection is here so a second
    runtime (or a hosted reranker) can be added without touching call sites.
    """
    from src.lib.reranker.local_onnx import OnnxCrossEncoderReranker

    return OnnxCrossEncoderReranker()
