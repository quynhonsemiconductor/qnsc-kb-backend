"""Extractive reader on ONNX Runtime: mDeBERTa-v3-base fine-tuned on SQuAD 2.0.

WHY THIS MODEL. It is the strongest multilingual span reader that fits a CPU
budget: 278M parameters, prebuilt ONNX (fp32 1.15 GB, int8 317 MB), MIT licensed,
SQuAD 2.0 dev F1 84.0 with NoAns F1 82.1. The NoAns number is the reason it was
chosen over `deepset/xlm-roberta-base-squad2` (SQuAD2 F1 77.2): UIT-ViQuAD 2.0
scores a correct abstention as a full point, so abstention quality is not a side
concern, it is a third of the Vietnamese benchmark.

HOW SQuAD 2.0 SPAN DECODING WORKS, and why the details below are not optional:

* The model emits `start_logits` and `end_logits` over TOKENS. Position 0 is the
  [CLS] token, and a SQuAD2-trained model is taught to put probability mass there
  when the passage does not contain the answer. So `null_score = start[0] + end[0]`
  is the model's own "no answer here" score, and every candidate span is judged
  against it. Ignoring position 0 turns an abstaining model into one that always
  answers, which is the single biggest scoring mistake available here.

* A passage longer than the window must be split into overlapping strides, or the
  answer that sits past the cut is unreachable. Each stride is decoded
  independently and the best span across strides wins.

* Only tokens belonging to the PASSAGE may be selected. Without that mask the
  argmax can land inside the question, which yields an answer that is a fragment
  of the question itself -- fluent, plausible, and always wrong.

* The answer is returned by slicing the ORIGINAL passage with the character
  offsets the tokenizer reports, never by joining decoded tokens. Detokenizing
  loses the original spacing and, in Vietnamese, mangles words that the
  sentencepiece vocabulary split mid-syllable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.core.config import settings
from src.lib.reader.base import ReaderUnavailable, Span

logger = structlog.get_logger()

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"


class _Lazy:
    """Load once, on first use, and remember the failure if it fails.

    Mirrors src/lib/embeddings/base.Lazy rather than importing it: a reader that
    is unavailable must raise ReaderUnavailable, not EmbeddingUnavailable, or the
    caller's error handling attributes the fault to the wrong subsystem.
    """

    def __init__(self, loader, label: str) -> None:
        self._loader = loader
        self._label = label
        self._value: Any = None
        self._error: Exception | None = None

    def get(self) -> Any:
        if self._error is not None:
            raise ReaderUnavailable(f"{self._label} is unavailable: {self._error}")
        if self._value is None:
            try:
                self._value = self._loader()
            except Exception as exc:  # noqa: BLE001 - remembered and re-raised
                self._error = exc
                raise ReaderUnavailable(f"{self._label} failed to load: {exc}") from exc
        return self._value


def _load() -> tuple[Any, Any]:
    directory = Path(settings.READER_ONNX_DIR or "")
    if not directory.is_dir():
        raise ReaderUnavailable(
            f"READER_ONNX_DIR={str(directory)!r} is not a directory. Export or download "
            "an ONNX question-answering model into it (model.onnx + tokenizer.json)."
        )
    model_path = directory / (settings.READER_ONNX_FILE or MODEL_FILE)
    tokenizer_path = directory / TOKENIZER_FILE
    for path in (model_path, tokenizer_path):
        if not path.is_file():
            raise ReaderUnavailable(f"{path} is missing from the reader export")

    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ReaderUnavailable(
            "The reader needs the optional 'onnx' dependency group (onnxruntime, "
            "tokenizers). Install with `poetry install --with onnx`."
        ) from exc

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = settings.READER_ONNX_THREADS
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Truncation and padding are applied per call: the stride logic below needs
    # exact control over both, and a tokenizer-level setting would silently
    # override it.
    logger.info(
        "ONNX reader ready",
        path=str(model_path),
        inputs=[item.name for item in session.get_inputs()],
    )
    return session, tokenizer


_model = _Lazy(_load, "ONNX reader")


class OnnxReader:
    name = "onnx"

    def warm_up(self) -> None:
        _model.get()

    def read(self, question: str, passages: list[str]) -> Span:
        """Best span across every passage, or an abstention.

        EVERY WINDOW OF EVERY PASSAGE IS RUN IN ONE BATCH, not one forward pass
        at a time. A question against 8 retrieved parents is ~15-40 stride
        windows; issued individually that measured 6.1 s per question on 4 CPU
        threads, because each `session.run` pays its own dispatch and leaves the
        thread pool idle between calls. Batched, the same windows go through as
        one `n x max_tokens` tensor.

        Windows are sorted by real (unpadded) length before batching so a batch
        pads to its own longest member rather than to the longest window in the
        whole question -- the same reason src/lib/embeddings/local_onnx.py sorts.

        Scores stay null-relative, so a passage that supports no answer cannot
        outrank one that does merely by being longer or more confident overall.
        """
        import numpy

        session, tokenizer = _model.get()
        declared = {item.name for item in session.get_inputs()}

        max_length = settings.READER_MAX_TOKENS
        stride = settings.READER_DOC_STRIDE
        tokenizer.no_padding()
        tokenizer.no_truncation()
        tokenizer.enable_truncation(
            max_length=max_length, stride=stride, strategy="only_second"
        )
        tokenizer.enable_padding(length=max_length)

        # (window, passage text, passage index) for every stride of every passage.
        units: list[tuple[Any, str, int]] = []
        for index, passage in enumerate(passages):
            if not passage.strip():
                continue
            encoding = tokenizer.encode(question, passage)
            # `overflowing` holds the later strides of a passage too long for one
            # window. Reading only `encoding` would silently drop the tail.
            for window in [encoding, *encoding.overflowing]:
                units.append((window, passage, index))

        if not units:
            return Span(text="", score=0.0, start=0, end=0, null_score=0.0)

        order = sorted(range(len(units)), key=lambda position: sum(units[position][0].attention_mask))
        batch_size = max(1, settings.READER_BATCH_SIZE)
        best: Span | None = None

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            feed = {
                "input_ids": numpy.array(
                    [units[position][0].ids for position in chunk], dtype=numpy.int64
                ),
                "attention_mask": numpy.array(
                    [units[position][0].attention_mask for position in chunk],
                    dtype=numpy.int64,
                ),
                "token_type_ids": numpy.array(
                    [units[position][0].type_ids for position in chunk], dtype=numpy.int64
                ),
            }
            outputs = session.run(
                None, {name: value for name, value in feed.items() if name in declared}
            )
            start_logits, end_logits = outputs[0], outputs[1]
            for row, position in enumerate(chunk):
                window, passage, passage_index = units[position]
                candidate = self._decode(
                    window, start_logits[row], end_logits[row], passage, passage_index
                )
                if candidate is not None and (best is None or candidate.score > best.score):
                    best = candidate

        if best is None:
            return Span(text="", score=0.0, start=0, end=0, null_score=0.0)
        return best

    def _decode(
        self,
        window: Any,
        start_logits: Any,
        end_logits: Any,
        passage: str,
        passage_index: int,
    ) -> Span | None:
        """Pick the best passage-internal span and compare it to the null score."""
        import numpy

        null_score = float(start_logits[0] + end_logits[0])
        # Sequence id 1 is the passage (0 is the question, None is special/padding).
        sequence_ids = window.sequence_ids
        offsets = window.offsets
        valid = [
            position
            for position, sequence in enumerate(sequence_ids)
            if sequence == 1 and offsets[position][1] > offsets[position][0]
        ]
        if not valid:
            return None

        top = settings.READER_NBEST
        candidates = [position for position in valid]
        starts = sorted(candidates, key=lambda position: -start_logits[position])[:top]
        ends = sorted(candidates, key=lambda position: -end_logits[position])[:top]

        best_score = -numpy.inf
        best_bounds: tuple[int, int] | None = None
        max_answer_tokens = settings.READER_MAX_ANSWER_TOKENS
        for start in starts:
            for end in ends:
                if end < start or (end - start + 1) > max_answer_tokens:
                    continue
                score = float(start_logits[start] + end_logits[end])
                if score > best_score:
                    best_score = score
                    best_bounds = (start, end)
        if best_bounds is None:
            return None

        start_char = offsets[best_bounds[0]][0]
        end_char = offsets[best_bounds[1]][1]
        text = passage[start_char:end_char].strip()
        if not text:
            return None
        return Span(
            text=text,
            # Null-relative, so scores from different passages are comparable and
            # the abstention decision is a simple sign test at the caller.
            score=best_score - null_score,
            start=int(start_char),
            end=int(end_char),
            null_score=null_score,
            passage_index=passage_index,
        )
