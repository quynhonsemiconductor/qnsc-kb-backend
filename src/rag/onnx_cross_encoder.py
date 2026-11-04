"""bge-reranker-v2-m3 on ONNX Runtime: the cross-encoder without torch.

`rag/cross_encoder.py` scored with sentence-transformers, which needs torch and measured ~0.5 s per
(question, passage) pair on 2 vCPU, so a pool of 30 was ~15 s against a 3 s budget and the feature could never be
switched on in a shipped image (which has no torch). The same weights quantised to int8 and run by ONNX Runtime,
which the image already carries for embeddings, measured 4.8 pairs/s on 2 threads with scores that correlate 0.995
with the torch ones and the same top-5 set.

Only `predict(pairs)` is needed by the caller, and it returns probabilities in 0..1 exactly as a torch CrossEncoder
with one output does (sigmoid of the logit), so the two backends are interchangeable.
"""

from __future__ import annotations

from typing import Any, Sequence


class OnnxCrossEncoder:
    def __init__(self, session: Any, tokenizer: Any, batch_size: int = 8) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._batch_size = max(1, batch_size)
        # Feed only what this export declares; an undeclared input is a hard ONNX Runtime error.
        self._declared = {i.name for i in session.get_inputs()}

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Relevance of each (question, passage) pair, in the caller's order."""
        import numpy

        if not pairs:
            return []
        # Length-sorted batches: padding goes to the longest member of a batch, not of the whole call.
        order = sorted(
            range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1])
        )
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), self._batch_size):
            window = order[start : start + self._batch_size]
            encodings = self._tokenizer.encode_batch(
                [(pairs[i][0], pairs[i][1]) for i in window]
            )
            feed = {
                "input_ids": numpy.array([e.ids for e in encodings], dtype=numpy.int64),
                "attention_mask": numpy.array(
                    [e.attention_mask for e in encodings], dtype=numpy.int64
                ),
            }
            logits = self._session.run(
                None, {k: v for k, v in feed.items() if k in self._declared}
            )[0]
            logits = numpy.asarray(logits, dtype=numpy.float64).reshape(
                len(window), -1
            )[:, 0]
            probabilities = 1.0 / (1.0 + numpy.exp(-numpy.clip(logits, -60.0, 60.0)))
            for position, index in enumerate(window):
                scores[index] = float(probabilities[position])
        return scores
