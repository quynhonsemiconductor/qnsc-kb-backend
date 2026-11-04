"""Parity gate: the shared TEI service vs the in-process export, on a fixed corpus.

Switching a deployment from EMBEDDING_RUNTIME=onnx to "tei" moves where the forward pass
runs, so it must not move the vectors: every stored chunk was produced by the in-process
export, and a query embedded into a slightly different space degrades retrieval without
erroring. The Service runs the same model and revision at fp32 with mean pooling — the
same three decisions that fix the vector space — so this test is what turns "should be
identical" into a number.

Skipped unless BOTH sides are reachable: an export in EMBEDDING_ONNX_DIR (with onnxruntime
and tokenizers installed) and a live TEI endpoint in TEI_EMBEDDINGS_URL. That is the image
on a machine that can reach the cluster, not a laptop. Run it before flipping a deployment,
and record the number in the PR that flips it.
"""

from __future__ import annotations

import os

import pytest

from src.core.config import settings
from src.lib import embeddings

# The same shape of input the corpus holds: short questions (decorated with the instruct
# prefix on the way in), Vietnamese, and one input long enough that the 512-token
# truncation actually bites — parity that holds only below the truncation point proves
# nothing about real chunks, which are ~1500 characters of Vietnamese.
_CORPUS = [
    "When must employees submit travel requests?",
    "annual leave policy",
    "Nhân viên phải làm gì trước khi đặt vé máy bay?",
    "Quy trình xin nghỉ phép của nhân viên yêu cầu người làm đơn gửi cho trưởng phòng",
    "ISO 9001:2015 clause 8.5.1 — control of production and service provision",
    "Nhiệt độ bảo quản tối đa của wafer là 30°C; độ ẩm tương đối dưới 60%.",
    (
        "Quy trình xin nghỉ phép của nhân viên yêu cầu người làm đơn gửi cho trưởng phòng "
        "trước ít nhất ba ngày làm việc và trưởng phòng phải xác nhận đơn bằng văn bản. "
    )
    * 20,
    "",
]


def _both_sides_reachable() -> bool:
    if not os.path.isdir(settings.EMBEDDING_ONNX_DIR):
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    if not settings.TEI_EMBEDDINGS_URL:
        return False
    import httpx

    try:
        response = httpx.get(
            f"{settings.TEI_EMBEDDINGS_URL.rstrip('/')}/health", timeout=5.0
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@pytest.mark.skipif(
    not _both_sides_reachable(),
    reason="needs a baked ONNX export AND a reachable TEI service; runs before a deployment flip",
)
@pytest.mark.parametrize("text", _CORPUS)
def test_tei_matches_the_in_process_export(monkeypatch, text):
    monkeypatch.setattr(settings, "EMBEDDING_RUNTIME", "onnx")
    reference = embeddings.get_bge_embedding(text)

    monkeypatch.setattr(settings, "EMBEDDING_RUNTIME", "tei")
    candidate = embeddings.get_bge_embedding(text)

    similarity = sum(a * b for a, b in zip(reference, candidate))
    assert similarity >= 0.999, (
        f"cosine similarity {similarity:.6f} for {text[:40]!r} — below this the two "
        "backends are different vector spaces, so switching EMBEDDING_RUNTIME requires "
        "re-embedding every chunk"
    )
