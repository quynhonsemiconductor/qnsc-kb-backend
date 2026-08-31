"""Cross-encoder reranker on ONNX Runtime: bge-reranker-v2-m3.

WHY THIS MODEL. The retriever's ranking is the largest measured accuracy loss on
Vietnamese (iteration 0: MLQA-vi gold-retrieved 57.5%), and iteration 4 proved
the incumbent lexical reranker overrides whatever order fusion produces -- so the
ranking that reaches the reader is a bag-of-terms score, not a relevance score.
bge-reranker-v2-m3 (BAAI, Apache-2.0, XLM-R-large backbone) is a multilingual
cross-encoder: it improves English AND Vietnamese, needs no word segmentation
(unlike PhoRanker's VnCoreNLP dependency), and ships a prebuilt CPU ONNX export,
so it drops onto the same ONNX Runtime seam the embedder and reader already use.

HOW IT SCORES. One (query, passage) pair per row -> one relevance logit. Inputs
are `input_ids` + `attention_mask` (XLM-R has no token_type_ids). Higher logit =
more relevant; scores are comparable only within a single query's candidates, so
the caller sorts per query and never mixes scores across queries.

COST. This is a 568M-parameter model, ~2.3 GB fp32 -- far heavier than the
zero-parameter lexical scorer it replaces. That is the deliberate CPU tradeoff:
reranking runs on at most RAG_RERANK_LIMIT candidates per query, never the
corpus, so the cost is bounded and paid once per search. Latency is measured in
the iteration, not assumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.core.config import settings
from src.lib.reranker.base import RerankerUnavailable, Scored

logger = structlog.get_logger()


class _Lazy:
    """Load once, on first use, and remember the failure.

    Mirrors src/lib/reader/local_onnx._Lazy: a reranker that is unavailable must
    raise RerankerUnavailable so the caller attributes the fault correctly and can
    fall back to the lexical scorer rather than failing the search.
    """

    def __init__(self, loader, label: str) -> None:
        self._loader = loader
        self._label = label
        self._value: Any = None
        self._error: Exception | None = None

    def get(self) -> Any:
        if self._error is not None:
            raise RerankerUnavailable(f"{self._label} is unavailable: {self._error}")
        if self._value is None:
            try:
                self._value = self._loader()
            except Exception as exc:  # noqa: BLE001 - remembered and re-raised
                self._error = exc
                raise RerankerUnavailable(f"{self._label} failed to load: {exc}") from exc
        return self._value


def _load() -> tuple[Any, Any]:
    directory = Path(settings.RERANKER_ONNX_DIR or "")
    if not directory.is_dir():
        raise RerankerUnavailable(
            f"RERANKER_ONNX_DIR={str(directory)!r} is not a directory. Download a "
            "cross-encoder ONNX export into it (model .onnx + tokenizer.json)."
        )
    model_path = directory / (settings.RERANKER_ONNX_FILE or "model.onnx")
    # The tokenizer may sit at the root or under tokenizer_hf/ depending on the
    # export; accept either rather than pinning one repo's layout.
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        tokenizer_path = directory / "tokenizer_hf" / "tokenizer.json"
    for path in (model_path, tokenizer_path):
        if not path.is_file():
            raise RerankerUnavailable(f"{path} is missing from the reranker export")

    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RerankerUnavailable(
            "The reranker needs the optional 'onnx' dependency group (onnxruntime, "
            "tokenizers). Install with `poetry install --with onnx`."
        ) from exc

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = settings.RERANKER_ONNX_THREADS
    options.inter_op_num_threads = 1

    # CPU-only by default and by construction: the declared `onnxruntime` wheel has
    # no CUDA provider compiled in, so this cannot silently acquire a GPU. The
    # filtering mirrors the reader: an unavailable provider name is dropped rather
    # than passed to InferenceSession, which would be a hard error.
    available = set(onnxruntime.get_available_providers())
    requested = [
        name.strip()
        for name in (settings.RERANKER_ONNX_PROVIDERS or "").split(",")
        if name.strip()
    ]
    providers = [name for name in requested if name in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    session = onnxruntime.InferenceSession(str(model_path), options, providers=providers)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    logger.info(
        "ONNX reranker ready",
        path=str(model_path),
        providers=session.get_providers(),
        inputs=[item.name for item in session.get_inputs()],
    )
    return session, tokenizer


_model = _Lazy(_load, "ONNX reranker")


class OnnxCrossEncoderReranker:
    name = "onnx-cross-encoder"

    def warm_up(self) -> None:
        _model.get()

    def score(self, query: str, passages: list[str]) -> list[Scored]:
        """Relevance logit for each (query, passage) pair, in one batched pass set.

        Every pair is truncated to RERANKER_MAX_TOKENS with `only_second`, so the
        query is always kept whole and only the passage tail is dropped -- the
        opposite would discard the question. Pairs are sorted by real length
        before batching so a batch pads to its own longest member, not to the
        longest pair in the whole set (the same reason the reader and embedder
        sort).
        """
        import numpy

        session, tokenizer = _model.get()
        declared = {item.name for item in session.get_inputs()}

        if not passages:
            return []

        tokenizer.no_padding()
        tokenizer.no_truncation()
        tokenizer.enable_truncation(
            max_length=settings.RERANKER_MAX_TOKENS, strategy="only_second"
        )

        encodings = [tokenizer.encode(query, passage) for passage in passages]
        order = sorted(range(len(encodings)), key=lambda i: sum(encodings[i].attention_mask))
        batch_size = max(1, settings.RERANKER_BATCH_SIZE)
        scores: list[float] = [0.0] * len(passages)

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            width = max(len(encodings[i].ids) for i in chunk)
            for i in chunk:
                encodings[i].pad(width)
            feed = {
                "input_ids": numpy.array(
                    [encodings[i].ids for i in chunk], dtype=numpy.int64
                ),
                "attention_mask": numpy.array(
                    [encodings[i].attention_mask for i in chunk], dtype=numpy.int64
                ),
            }
            if "token_type_ids" in declared:
                feed["token_type_ids"] = numpy.array(
                    [encodings[i].type_ids for i in chunk], dtype=numpy.int64
                )
            outputs = session.run(
                None, {name: value for name, value in feed.items() if name in declared}
            )
            logits = outputs[0]
            for row, i in enumerate(chunk):
                # Output is (batch, 1); take the single relevance logit per row.
                scores[i] = float(numpy.ravel(logits[row])[0])

        return [Scored(index=i, score=scores[i]) for i in range(len(passages))]
