"""Shared types for reader backends.

`Span` is deliberately more than a string: the score is what lets a caller
compare candidate answers ACROSS passages, and the character offsets are what let
it map an answer back to the chunk it came from for citation. A backend that
returned only text would force every caller to re-find the answer in the
passage, which is ambiguous whenever the same words appear twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ReaderUnavailable(RuntimeError):
    """The configured reader cannot be loaded or run."""


@dataclass(frozen=True, slots=True)
class Span:
    """One candidate answer.

    `text` is the extracted answer, empty for an abstention.
    `score` is the span's logit sum minus the null score, so it is comparable
    between passages read by the same model: higher means this passage supports
    the answer more strongly than it supports having no answer at all.
    `start`/`end` are character offsets into the passage that produced it.
    `null_score` is the model's own no-answer score for that passage, kept so a
    caller can apply its own abstention threshold without re-running inference.
    """

    text: str
    score: float
    start: int
    end: int
    null_score: float
    passage_index: int = 0

    @property
    def is_abstention(self) -> bool:
        return not self.text


class Reader(Protocol):
    """What every reader backend must provide."""

    name: str

    def warm_up(self) -> None:
        """Load weights before first use, so no request pays the cold start."""

    def read(self, question: str, passages: list[str]) -> Span:
        """Return the best span across `passages`, or an abstention."""
