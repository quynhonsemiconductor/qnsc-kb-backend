import numpy as np

from src.lib.embeddings import OnnxEmbeddingModel


def test_onnx_embedding_casts_token_inputs_and_normalizes_cls_vectors():
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": np.array([[1, 2], [3, 4]], dtype=np.int32),
                "attention_mask": np.array([[1, 1], [1, 1]], dtype=np.int32),
            }

    class Session:
        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"input_ids", "attention_mask"}
            assert all(value.dtype == np.int64 for value in inputs.values())
            return [
                np.array(
                    [
                        [[3.0, 4.0], [100.0, 100.0]],
                        [[5.0, 12.0], [100.0, 100.0]],
                    ]
                )
            ]

    model = object.__new__(OnnxEmbeddingModel)
    model._tokenizer = Tokenizer()
    model._session = Session()
    model._input_names = {"input_ids", "attention_mask"}

    assert model.encode(["first", "second"]) == [[0.6, 0.8], [5 / 13, 12 / 13]]
