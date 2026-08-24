import numpy as np
from types import SimpleNamespace

from src.lib.embeddings.local_onnx import OnnxEmbeddingProvider


def _stub_model(monkeypatch):
    """A session and tokenizer that need no ONNX files on disk."""

    class Tokenizer:
        def encode_batch(self, texts):
            assert texts == ["first", "second"]
            return [
                SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0]),
                SimpleNamespace(ids=[3, 4], attention_mask=[1, 0], type_ids=[0, 0]),
            ]

    class Input:
        def __init__(self, name):
            self.name = name

    class Session:
        def get_inputs(self):
            return [Input("input_ids"), Input("attention_mask")]

        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"input_ids", "attention_mask"}
            assert all(value.dtype == np.int64 for value in inputs.values())
            return [
                np.array(
                    [
                        [[3.0, 4.0], [7.0, 8.0]],
                        [[5.0, 12.0], [100.0, 100.0]],
                    ]
                )
            ]

    class LazyModel:
        def get(self):
            return Session(), Tokenizer()

    import src.lib.embeddings.local_onnx as local_onnx

    monkeypatch.setattr(local_onnx, "_model", LazyModel())
    return local_onnx


def test_mean_pooling_respects_the_attention_mask(monkeypatch):
    """The default, and what paraphrase-multilingual-MiniLM-L12-v2 requires.

    The mask matters: the second input has one real token, so its padding position must
    not drag the vector toward itself. Getting this wrong yields a valid vector in the
    wrong space and degrades retrieval with no error anywhere.
    """
    local_onnx = _stub_model(monkeypatch)
    monkeypatch.setattr(local_onnx.settings, "EMBEDDING_ONNX_POOLING", "mean")

    assert local_onnx.OnnxEmbeddingProvider().embed(["first", "second"]) == [
        [5.0, 6.0],
        [5.0, 12.0],
    ]


def test_cls_pooling_takes_the_first_token(monkeypatch):
    """Still supported, and what the bge-* family needs."""
    local_onnx = _stub_model(monkeypatch)
    monkeypatch.setattr(local_onnx.settings, "EMBEDDING_ONNX_POOLING", "cls")

    assert local_onnx.OnnxEmbeddingProvider().embed(["first", "second"]) == [
        [3.0, 4.0],
        [5.0, 12.0],
    ]


def test_an_unknown_pooling_mode_fails_loudly(monkeypatch):
    import pytest

    local_onnx = _stub_model(monkeypatch)
    monkeypatch.setattr(local_onnx.settings, "EMBEDDING_ONNX_POOLING", "max")

    with pytest.raises(local_onnx.EmbeddingUnavailable, match="not one of"):
        local_onnx.OnnxEmbeddingProvider().embed(["first", "second"])
