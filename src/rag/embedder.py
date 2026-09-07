"""Embedding boundary for the RAG pipeline."""
from __future__ import annotations

from src.domain.search_service import VectorSearchUnavailable, embed_query

__all__ = ["VectorSearchUnavailable", "embed_query"]
