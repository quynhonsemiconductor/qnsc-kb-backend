"""Shared types for cross-encoder reranker backends.

A cross-encoder scores a (query, passage) PAIR jointly, attending across both at
once, rather than embedding each independently and comparing vectors. That joint
attention is what lets it judge relevance the bi-encoder retriever cannot: a
question and the passage that answers it rarely share surface wording, so a
lexical or symmetric-embedding score ranks the right passage low. The cost is
that every candidate needs its own forward pass, which is why reranking runs only
on the small retrieved set, never the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RerankerUnavailable(RuntimeError):
    """The configured reranker cannot be loaded or run."""


@dataclass(frozen=True, slots=True)
class Scored:
    """One (passage index, relevance score) pair.

    `index` refers back to the caller's passage list, so the caller can reorder
    its own objects without the backend needing to know their type.
    `score` is the cross-encoder's raw relevance logit: higher is more relevant,
    and values are comparable within one query's candidate set.
    """

    index: int
    score: float


class CrossEncoderReranker(Protocol):
    """What every reranker backend must provide."""

    name: str

    def warm_up(self) -> None:
        """Load weights before first use, so no request pays the cold start."""

    def score(self, query: str, passages: list[str]) -> list[Scored]:
        """Score each passage against the query. Order of the input is preserved
        in the `index` field, NOT by sorting -- the caller decides the ordering."""
