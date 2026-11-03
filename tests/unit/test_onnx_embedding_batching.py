"""ONNX embedding must batch by length, and must not reorder its output.

Indexing a 38-page PDF measured 29 s on the deployed worker. The backend handed the
tokenizer the whole document in one call, and `enable_padding()` pads every encoding to
the LONGEST member of the call — so a single chunk reaching the 128-token cap padded all
~87 of them to 128 and the model ran a full forward pass over the padding.

Sorting by length first means each batch pads to its own longest member instead of the
document's. It also bounds peak memory: the single-call version built one
`n_chunks x max_tokens` tensor, which for a 500-page document is thousands of rows in a
task that also holds clamd and PaddleOCR.

ORDER IS THE DANGEROUS PART. Callers zip the returned vectors against their chunk list,
so sorting internally without restoring positions would attach every embedding to the
wrong chunk — a corpus that looks indexed and retrieves nonsense, with nothing failing.
"""
from __future__ import annotations

import types

import pytest

from src.core.config import settings
from src.lib.embeddings import local_onnx


class _Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class _Tokenizer:
    """Pads to the longest member of each call, exactly as enable_padding() does."""

    def __init__(self):
        self.batch_widths: list[int] = []

    def encode_batch(self, texts):
        # One token per 4 characters, capped like the real truncation setting. The token
        # id identifies the TEXT, so a caller can tell which input produced a vector.
        lengths = [min(max(1, len(t) // 4), settings.EMBEDDING_MAX_TOKENS) for t in texts]
        width = max(lengths)
        self.batch_widths.append(width)
        return [
            _Encoding([ord(text[0])] * length + [0] * (width - length))
            for text, length in zip(texts, lengths)
        ]


class _Session:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in ("input_ids", "attention_mask")]

    def run(self, _outputs, feed):
        import numpy

        self.calls += 1
        rows, width = feed["input_ids"].shape
        # The pooled vector must depend ONLY on the row's own content, exactly as a real
        # model's does: padding is masked out by _pool, so batching a text with others
        # must not change its embedding. Encoding the batch size here instead would make
        # the order test fail for a reason that has nothing to do with ordering.
        hidden = numpy.zeros((rows, width, 1), dtype=numpy.float32)
        for row in range(rows):
            hidden[row, :, 0] = float(feed["input_ids"][row][0])
        return [hidden]


@pytest.fixture
def fake_model(monkeypatch):
    session, tokenizer = _Session(), _Tokenizer()
    monkeypatch.setattr(local_onnx._model, "get", lambda: (session, tokenizer))
    return session, tokenizer


def _texts(*lengths):
    """Distinct texts of the given character lengths."""
    return [chr(ord("a") + i) * length for i, length in enumerate(lengths)]


def test_every_input_gets_a_vector(fake_model):
    vectors = local_onnx.OnnxEmbeddingProvider().embed(_texts(100, 400, 40, 900))
    assert len(vectors) == 4


def test_order_is_preserved_despite_internal_sorting(fake_model):
    """The failure this would cause is silent: every chunk keeps its row, but the
    embedding attached to it belongs to a different chunk."""
    provider = local_onnx.OnnxEmbeddingProvider()
    texts = _texts(900, 40, 400, 100, 1200)

    batched = provider.embed(texts)
    one_at_a_time = [provider.embed([text])[0] for text in texts]

    assert batched == one_at_a_time


def test_batches_respect_EMBEDDING_BATCH_SIZE(monkeypatch, fake_model):
    """Previously the whole document went in a single call, unbounded."""
    session, _ = fake_model
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 4)

    local_onnx.OnnxEmbeddingProvider().embed(_texts(*([100] * 10)))

    assert session.calls == 3  # 4 + 4 + 2


def test_a_long_outlier_no_longer_pads_the_whole_document(monkeypatch, fake_model):
    """The regression: one chunk at the token cap used to widen every other chunk."""
    _, tokenizer = fake_model
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 4)

    # Seven short chunks and one very long one, interleaved.
    local_onnx.OnnxEmbeddingProvider().embed(_texts(40, 40, 40, 4000, 40, 40, 40, 40))

    widths = tokenizer.batch_widths
    assert len(widths) == 2
    # Only the batch holding the outlier is wide; the other stays at the short width.
    assert min(widths) <= 10, widths
    assert max(widths) > min(widths), widths


def test_no_texts_is_not_a_model_call(fake_model):
    session, _ = fake_model
    assert local_onnx.OnnxEmbeddingProvider().embed([]) == []
    assert session.calls == 0
