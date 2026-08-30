"""Extractive answer reading: pick the answer span out of retrieved passages.

WHY THIS EXISTS. `AIService.ask` generates answers through a hosted LLM API
(src/domain/llm_client.py). That is the product's answer path and it is not
changed by anything here. But it cannot be measured on MLQA / UIT-ViQuAD under a
CPU-only constraint: without a provider configured, ai_service falls back to
truncating the top chunk at 200 characters, which scores nothing meaningful, and
with a provider configured the answer is produced by a remote GPU.

This package adds the CPU-only alternative the constraint requires: a small
multilingual extractive reader that runs in-process on ONNX Runtime, the same way
embeddings already do. Both benchmarks are span-extraction tasks -- the answer is
a substring of the passage -- so an extractive reader is the honest instrument
for them, not a workaround.

The seam mirrors src/lib/embeddings: `resolve_reader()` picks a backend, the
backend loads lazily, and callers depend on the protocol rather than the runtime.
"""
from __future__ import annotations

from src.lib.reader.base import ReaderUnavailable, Span

__all__ = ["ReaderUnavailable", "Span", "resolve_reader"]


def resolve_reader():
    """Return the configured reader backend.

    Only the ONNX backend exists today; the indirection is here so a second
    runtime (or a hosted reader) can be added without touching call sites.
    """
    from src.lib.reader.local_onnx import OnnxReader

    return OnnxReader()
