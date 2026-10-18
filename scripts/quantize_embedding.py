"""Build the int8 embedding export the api image serves search queries from.

Run by the `embedding-int8` stage of the Dockerfile; it is a build tool, never shipped.

    python scripts/quantize_embedding.py --download intfloat/multilingual-e5-large-instruct \
        --fp32-dir /opt/embedding-fp32 --out-dir /opt/embedding-int8

WHY THE API ONLY. The api embeds one short question per search; the worker embeds every
chunk that is stored. Quantisation moves a vector (cosine ~0.995 against fp32 here), so
a worker that wrote int8 vectors would need the whole corpus re-embedded to stay
consistent. An int8 QUERY against the fp32 vectors already stored shifts each
query-to-chunk score by ~0.002 on average and 0.009 at most (measured, see
docs/embedding-runtime-onnx.md), needs no re-embed, and is the part that runs on every
search. So the api gets this export and the worker keeps fp32.

THE RECIPE IS NOT INTERCHANGEABLE. Every choice below was measured on
multilingual-e5-large-instruct, the shipped recipe on the onnxruntime in poetry.lock and the
rejected ones on an adjacent release (docs/embedding-runtime-onnx.md has the table):

  MatMul only            1331 MB. The word-embedding table (~1 GB of the 2.2 GB) is a
                         Gather, not a MatMul, so it stays fp32. Resident memory is the
                         same as with it quantised (rows no query touches are never paged
                         in: 701 MB vs 700 MB), but the image is ~770 MB bigger.
  + Gather               562 MB. The table is quantised too.
  per-tensor             mean cosine 0.993 (reduce_range off) / 0.984 (on)
  per-channel + reduce_range
                         mean cosine 0.995, same speed. reduce_range keeps 7-bit activations
                         so the u8s8 kernels cannot saturate on CPUs without VNNI, which
                         Fargate's x86 fleet cannot be relied on to have.
  int4 (MatMulNBits)     447 MB but 4x SLOWER than int8 on CPU, mean cosine 0.98: rejected.

The earlier "int8 rejected" measurement in docs/embedding-runtime-onnx.md was bge-m3, not
this model, and was not re-run when the model changed.

TWO GATES FAIL THE BUILD rather than ship a degraded export, because a worse query vector
raises nothing at runtime — retrieval just gets quietly worse:
  * size: the output must be under MAX_SIZE_RATIO of the fp32 export, which catches a
    recipe that silently skipped the embedding table.
  * parity: cosine against the fp32 export on fixed probe texts must stay above MIN_COSINE.
    The probes are run through BOTH exports with the same pooling, so this compares the
    two exports, not either one against the application.

onnxruntime APPENDS to an existing external-data file instead of replacing it, so
re-running into a used directory silently doubles the output. The output directory must
therefore be empty or absent.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"

# Measured minimum over 47 probe texts was 0.9914 (passages) / 0.9936 (queries). The gate
# sits below that with room for build-machine CPU differences, and well above the 0.984 a
# per-tensor recipe produces, so it still separates the recipes.
MIN_COSINE = 0.985
# Measured 0.251. The embedding-table-left-in-fp32 failure measures 0.60.
MAX_SIZE_RATIO = 0.35
MAX_TOKENS = 512

_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"
_LONG_PASSAGE = (
    "Quy trinh xin nghi phep cua nhan vien yeu cau nguoi lam don gui cho truong phong "
    "truoc it nhat ba ngay lam viec va truong phong phai xac nhan don bang van ban "
    "truoc khi de nghi nghi phep duoc phe duyet boi phong nhan su. "
) * 40  # well over MAX_TOKENS, so truncation is part of what is compared
_PROBES = (
    f"Instruct: {_INSTRUCTION}\nQuery: Xin nghỉ phép phải báo trước bao lâu?",
    f"Instruct: {_INSTRUCTION}\nQuery: annual leave policy",
    f"Instruct: {_INSTRUCTION}\nQuery: Vào phòng sạch cần mặc gì?",
    "Quy trình xin nghỉ phép: nhân viên gửi đơn cho trưởng phòng trước ít nhất ba ngày làm việc.",
    "Chính sách công tác phí: nhân viên được thanh toán vé máy bay hạng phổ thông và khách sạn.",
    "The annual leave policy requires a request to be filed three working days in advance.",
    _LONG_PASSAGE,
)


class QuantisationError(RuntimeError):
    """A gate failed, or the inputs cannot produce a trustworthy export."""


def directory_bytes(directory: Path) -> int:
    """Weights plus graph; the tokenizer is identical in both exports and not what shrinks."""
    return sum(
        path.stat().st_size
        for path in directory.iterdir()
        if path.is_file() and path.name != TOKENIZER_FILE
    )


def require_empty(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise QuantisationError(
            f"{directory} is not empty. onnxruntime appends to an existing external-data "
            "file rather than replacing it, so quantising into a used directory silently "
            "doubles the output. Point --out-dir at an empty or missing directory."
        )


def check_size(fp32_bytes: int, quantised_bytes: int) -> float:
    ratio = quantised_bytes / fp32_bytes
    if ratio > MAX_SIZE_RATIO:
        raise QuantisationError(
            f"quantised export is {ratio:.2f} of the fp32 size (limit {MAX_SIZE_RATIO}). "
            "The usual cause is the embedding table (a Gather, not a MatMul) staying fp32."
        )
    return ratio


def check_parity(cosines: list[float]) -> float:
    worst = min(cosines)
    if worst < MIN_COSINE:
        raise QuantisationError(
            f"cosine against the fp32 export fell to {worst:.4f} (limit {MIN_COSINE}); "
            "a query embedded this far off the stored vectors degrades retrieval silently."
        )
    return worst


def download_fp32(model_id: str, target: Path) -> None:
    """The export the model publishes: graph + tokenizer, plus the weights sibling file
    only when the repo has one (e5-large's fp32 graph exceeds protobuf's 2 GB limit)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    target.mkdir(parents=True, exist_ok=True)
    names = ["onnx/model.onnx", "onnx/tokenizer.json"]
    if "onnx/model.onnx_data" in list_repo_files(model_id):
        names.append("onnx/model.onnx_data")
    for name in names:
        shutil.copyfile(hf_hub_download(model_id, name), target / Path(name).name)


def quantise(fp32_dir: Path, out_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    require_empty(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(fp32_dir / MODEL_FILE),
        str(out_dir / MODEL_FILE),
        weight_type=QuantType.QInt8,
        # Gather is the embedding table; see the module docstring.
        op_types_to_quantize=["MatMul", "Gather"],
        per_channel=True,
        reduce_range=True,
        use_external_data_format=True,
    )
    shutil.copyfile(fp32_dir / TOKENIZER_FILE, out_dir / TOKENIZER_FILE)


def _load(directory: Path):
    import onnxruntime
    from tokenizers import Tokenizer

    session = onnxruntime.InferenceSession(
        str(directory / MODEL_FILE), providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_FILE))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding()
    return session, tokenizer


def _embed(model, texts: tuple[str, ...]):
    import numpy

    session, tokenizer = model
    declared = {i.name for i in session.get_inputs()}
    encodings = tokenizer.encode_batch(list(texts))
    feed = {
        "input_ids": numpy.array([e.ids for e in encodings], dtype=numpy.int64),
        "attention_mask": numpy.array([e.attention_mask for e in encodings], dtype=numpy.int64),
        "token_type_ids": numpy.array([e.type_ids for e in encodings], dtype=numpy.int64),
    }
    hidden = session.run(None, {k: v for k, v in feed.items() if k in declared})[0]
    mask = feed["attention_mask"][..., None].astype(hidden.dtype)
    pooled = (hidden * mask).sum(axis=1) / numpy.clip(mask.sum(axis=1), 1e-9, None)
    return pooled / numpy.linalg.norm(pooled, axis=1, keepdims=True)


def measure_cosines(fp32_dir: Path, out_dir: Path) -> list[float]:
    reference = _embed(_load(fp32_dir), _PROBES)
    candidate = _embed(_load(out_dir), _PROBES)
    return [float(v) for v in (reference * candidate).sum(axis=1)]


def build(fp32_dir: Path, out_dir: Path) -> None:
    quantise(fp32_dir, out_dir)
    ratio = check_size(directory_bytes(fp32_dir), directory_bytes(out_dir))
    cosines = measure_cosines(fp32_dir, out_dir)
    worst = check_parity(cosines)
    print(
        f"int8 export ready: {directory_bytes(out_dir) / 1e6:.0f} MB "
        f"({ratio:.2f} of fp32), worst cosine {worst:.4f} over {len(cosines)} probes",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fp32-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--download",
        metavar="MODEL_ID",
        help="fetch the published fp32 export of this model into --fp32-dir first",
    )
    args = parser.parse_args(argv)
    try:
        # Checked before the download: a used output directory is a mistake worth
        # reporting in seconds, not after 2.2 GB.
        require_empty(args.out_dir)
        if args.download:
            download_fp32(args.download, args.fp32_dir)
        build(args.fp32_dir, args.out_dir)
    except QuantisationError as exc:
        print(f"quantize_embedding: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
