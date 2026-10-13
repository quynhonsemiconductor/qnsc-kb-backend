"""The two guarantees `rag/cross_encoder.py` has to keep.

1. `reorder_by_cross_encoder` is pure and index-aligned: it must reorder by score without
   ever comparing the ranked items themselves (they can be ORM objects with no ordering of
   their own), and a length mismatch between items and scores is a caller bug that must
   raise, not silently zip to the shorter length and drop data.
2. `score()` on a missing dependency raises `CrossEncoderUnavailable`, not ImportError or
   some other exception a caller's `except CrossEncoderUnavailable` would not catch --
   that is the exact contract `domain/search_service.py` falls back on.
"""
from __future__ import annotations

import pytest

from src.rag.cross_encoder import CrossEncoderUnavailable, reorder_by_cross_encoder


class _Unorderable:
    """Stands in for a chunk ORM object: comparable only by identity, never by <."""

    def __init__(self, label: str):
        self.label = label

    def __lt__(self, other):  # pragma: no cover - must never be called
        raise TypeError("chunk objects are not orderable")

    def __repr__(self):
        return f"_Unorderable({self.label!r})"


def test_reorders_highest_score_first():
    items = [_Unorderable("a"), _Unorderable("b"), _Unorderable("c")]
    scores = [0.1, 0.9, 0.5]
    result = reorder_by_cross_encoder(items, scores)
    assert [item.label for item in result] == ["b", "c", "a"]


def test_never_compares_the_ranked_items_themselves():
    # Two items tied on score. If the sort ever fell back to comparing the items (as
    # Python's sort does for equal keys with no `key=`), this would raise from __lt__
    # instead of returning a stable order.
    items = [_Unorderable("first"), _Unorderable("second")]
    scores = [0.5, 0.5]
    result = reorder_by_cross_encoder(items, scores)
    assert [item.label for item in result] == ["first", "second"]


def test_preserves_the_paired_tuple_shape():
    """The real caller passes (chunk, lexical_score) tuples, not bare chunks."""
    ranked = [(_Unorderable("low"), 0.2), (_Unorderable("high"), 0.9)]
    cross_scores = [0.1, 0.8]
    result = reorder_by_cross_encoder(ranked, cross_scores)
    assert [chunk.label for chunk, _lexical_score in result] == ["high", "low"]
    # The lexical score travelling with "high" is untouched by reordering.
    assert result[0][1] == 0.9


def test_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        reorder_by_cross_encoder([_Unorderable("a"), _Unorderable("b")], [0.5])


def test_empty_input_returns_empty():
    assert reorder_by_cross_encoder([], []) == []


def test_missing_dependency_raises_cross_encoder_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Force a fresh load attempt regardless of any previous test's cached model.
    import src.rag.cross_encoder as cross_encoder_module

    monkeypatch.setattr(
        cross_encoder_module,
        "_model",
        cross_encoder_module.Lazy(cross_encoder_module._load, "cross-encoder"),
    )

    with pytest.raises(CrossEncoderUnavailable):
        cross_encoder_module.score("query", ["passage"])


# ── the reranker must not be able to make a search look stuck ────────────────────────────────
# Measured on 2 vCPU: ~0.5 s per passage, ~15 s for the default pool of 30. Two bounds keep that
# from reaching the user: a timeout at the call site, and one scoring call at a time so a
# timed-out call (a thread, which cannot be cancelled) cannot be stacked on by the next search.

import threading

from src.rag import cross_encoder


def test_a_second_scoring_call_is_refused_while_the_first_is_still_running(monkeypatch):
    started, release = threading.Event(), threading.Event()

    class _Slow:
        def predict(self, pairs):
            started.set()
            release.wait(5)
            return [0.5] * len(pairs)

    monkeypatch.setattr(cross_encoder._model, "get", lambda: _Slow())
    first = threading.Thread(target=lambda: cross_encoder.score("q", ["a", "b"]))
    first.start()
    try:
        assert started.wait(5)
        with pytest.raises(CrossEncoderUnavailable, match="still scoring"):
            cross_encoder.score("q", ["c"])
    finally:
        release.set()
        first.join(5)

    # and it is usable again once the first call has finished (the lock is released in `finally`)
    assert cross_encoder.score("q", ["a"]) == [0.5]


def test_the_lock_is_released_when_scoring_raises(monkeypatch):
    class _Broken:
        def predict(self, pairs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(cross_encoder._model, "get", lambda: _Broken())
    with pytest.raises(CrossEncoderUnavailable):
        cross_encoder.score("q", ["a"])

    assert cross_encoder._scoring.acquire(blocking=False), "a failed call left the reranker locked"
    cross_encoder._scoring.release()
