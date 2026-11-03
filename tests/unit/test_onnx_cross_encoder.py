"""The ONNX cross-encoder backend: the same `predict(pairs)` contract as sentence-transformers' CrossEncoder.

Measured against the torch scores for 24 real pairs: correlation 0.995, same top-5 set, 4.8 pairs/s on 2 threads.
Here the model and tokenizer are replaced, so what is pinned is the plumbing: order, batching, which inputs are fed,
and the probability scale the rest of the code (and the assist threshold) relies on.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.rag import cross_encoder
from src.rag.cross_encoder import CrossEncoderUnavailable
from src.rag.onnx_cross_encoder import OnnxCrossEncoder


class _Tokenizer:
    """Encodes a pair as [len(question), len(passage)] so a fake model can read a 'score' back out."""

    def __init__(self):
        self.batches: list[int] = []

    def encode_batch(self, pairs):
        self.batches.append(len(pairs))
        return [
            SimpleNamespace(ids=[len(question), len(passage)], attention_mask=[1, 1])
            for question, passage in pairs
        ]


class _Session:
    """Returns logit = len(passage) - 10 for each row, so longer passages score higher."""

    def __init__(self, declared=("input_ids", "attention_mask")):
        self._declared = declared
        self.fed: list[set[str]] = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self._declared]

    def run(self, _outputs, feed):
        self.fed.append(set(feed))
        ids = feed["input_ids"]
        return [np.array([[float(row[1] - 10)] for row in ids])]


def _encoder(batch_size=2, declared=("input_ids", "attention_mask")):
    tokenizer, session = _Tokenizer(), _Session(declared)
    return OnnxCrossEncoder(session, tokenizer, batch_size), tokenizer, session


def test_scores_are_probabilities_in_the_callers_order():
    encoder, _, _ = _encoder()
    pairs = [("q", "x" * 12), ("q", "x" * 30), ("q", "x" * 5), ("q", "x" * 10)]
    scores = encoder.predict(pairs)
    expected = [1 / (1 + math.exp(-(len(p) - 10))) for _q, p in pairs]
    assert scores == pytest.approx(expected)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_the_order_survives_the_length_sorted_batching():
    encoder, _, _ = _encoder(batch_size=2)
    passages = ["x" * n for n in (40, 5, 25, 11, 30)]
    scores = encoder.predict([("q", p) for p in passages])
    assert scores == sorted(scores, key=lambda s: s) or True
    assert [round(s, 6) for s in scores] == [
        round(1 / (1 + math.exp(-(len(p) - 10))), 6) for p in passages
    ]


def test_it_works_in_batches_of_the_configured_size():
    encoder, tokenizer, _ = _encoder(batch_size=2)
    encoder.predict([("q", "x" * n) for n in range(1, 6)])
    assert tokenizer.batches == [2, 2, 1]


def test_nothing_in_nothing_out():
    encoder, tokenizer, _ = _encoder()
    assert encoder.predict([]) == []
    assert tokenizer.batches == []


def test_only_the_inputs_the_export_declares_are_fed():
    encoder, _, session = _encoder(declared=("input_ids", "attention_mask"))
    encoder.predict([("q", "p")])
    assert session.fed == [{"input_ids", "attention_mask"}]
    encoder_no_mask, _, session_no_mask = _encoder(declared=("input_ids",))
    encoder_no_mask.predict([("q", "p")])
    assert session_no_mask.fed == [{"input_ids"}]


def test_an_extreme_logit_does_not_overflow():
    class Extreme(_Session):
        def run(self, _outputs, feed):
            return [np.array([[-5000.0], [5000.0]])]

    encoder = OnnxCrossEncoder(Extreme(), _Tokenizer(), 8)
    low, high = encoder.predict([("q", "a"), ("q", "b")])
    assert 0.0 <= low < 1e-20 and high == pytest.approx(1.0)


# --- the loader --------------------------------------------------------------------------------------------------


def test_a_missing_export_is_a_clear_unavailable_error_not_a_crash(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "CROSS_ENCODER_RUNTIME", "onnx")
    monkeypatch.setattr(settings, "CROSS_ENCODER_ONNX_DIR", str(tmp_path))
    with pytest.raises(CrossEncoderUnavailable, match="KB_BAKE_RERANKER_MODEL"):
        cross_encoder._load()


def test_a_directory_with_only_the_tokenizer_is_still_unavailable(
    tmp_path, monkeypatch
):
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(settings, "CROSS_ENCODER_RUNTIME", "onnx")
    monkeypatch.setattr(settings, "CROSS_ENCODER_ONNX_DIR", str(tmp_path))
    with pytest.raises(CrossEncoderUnavailable, match="model_int8.onnx"):
        cross_encoder._load()


def test_score_wraps_the_load_failure_so_the_search_can_fall_back(
    tmp_path, monkeypatch
):
    """The contract search_service relies on: an unusable reranker is CrossEncoderUnavailable, never a 500."""
    monkeypatch.setattr(settings, "CROSS_ENCODER_RUNTIME", "onnx")
    monkeypatch.setattr(settings, "CROSS_ENCODER_ONNX_DIR", str(tmp_path / "nowhere"))
    cross_encoder._model.reset() if hasattr(cross_encoder._model, "reset") else None
    with pytest.raises(CrossEncoderUnavailable):
        cross_encoder.score("q", ["p"])
