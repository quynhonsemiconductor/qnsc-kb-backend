"""The hosted embedding backend must batch, retry, and label the task correctly.

Dormant, not harmless. The deployment runs a local model, so none of this fires today —
but `src/lib/embeddings.py` already carried a duplicate of this code that Python never
imported, because the package `src/lib/embeddings/` shadows it. Reading the shadowed copy
is how these were nearly "fixed" in a file that does not run. The live one had the same
three faults:

BATCHING — every chunk of a document went into a single batchEmbedContents request. 87 of
them for a 38-page PDF, thousands for a large one, against an API that bounds both entry
count and total tokens. The whole document fails at once. EMBEDDING_BATCH_SIZE existed
and only one backend honoured it.

RETRIES — `raise_for_status()` turned a 429 or a transient 503 into a failed index for
the entire article. Rate limits are the expected condition during bulk ingestion.

TASK TYPE — hardcoded RETRIEVAL_QUERY for everything, including the chunks being indexed,
directly contradicting the comment beside it warning that mixing questions and passages
"degrades retrieval quietly rather than visibly".
"""
from __future__ import annotations

import httpx
import pytest

from src.core.config import settings
from src.lib.embeddings import hosted


class _Response:
    def __init__(self, status: int, count: int = 0):
        self.status_code = status
        self._count = count

    def json(self):
        return {"embeddings": [{"values": [0.1, 0.2]} for _ in range(self._count)]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(settings, "EMBEDDING_HTTP_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(hosted.time, "sleep", lambda _s: None)


def _capture(monkeypatch, statuses=None):
    """Record each POST; `statuses` drives failures before the eventual success."""
    calls = []
    queue = list(statuses or [])

    def _post(url, params=None, json=None, timeout=None):
        calls.append(json)
        status = queue.pop(0) if queue else 200
        return _Response(status, count=len(json["requests"]))

    monkeypatch.setattr(hosted.httpx, "post", _post)
    return calls


def test_a_document_is_split_into_bounded_batches(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 4)
    calls = _capture(monkeypatch)

    vectors = hosted.HostedEmbeddingProvider().embed([f"chunk {i}" for i in range(10)])

    assert len(calls) == 3  # 4 + 4 + 2
    assert [len(c["requests"]) for c in calls] == [4, 4, 2]
    assert len(vectors) == 10


def test_a_rate_limit_is_retried(monkeypatch):
    calls = _capture(monkeypatch, statuses=[429, 200])

    vectors = hosted.HostedEmbeddingProvider().embed(["one"])

    assert len(calls) == 2
    assert len(vectors) == 1


def test_a_transient_server_error_is_retried(monkeypatch):
    calls = _capture(monkeypatch, statuses=[503, 200])
    hosted.HostedEmbeddingProvider().embed(["one"])
    assert len(calls) == 2


def test_a_bad_request_is_not_retried(monkeypatch):
    """Repeating a malformed request only spends the quota again."""
    calls = _capture(monkeypatch, statuses=[400, 200])

    with pytest.raises(httpx.HTTPStatusError):
        hosted.HostedEmbeddingProvider().embed(["one"])

    assert len(calls) == 1


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_HTTP_MAX_ATTEMPTS", 3)
    calls = _capture(monkeypatch, statuses=[429, 429, 429, 429, 429])

    with pytest.raises(httpx.HTTPStatusError):
        hosted.HostedEmbeddingProvider().embed(["one"])

    assert len(calls) == 3


def test_indexing_and_searching_use_different_task_types(monkeypatch):
    """The fault the code's own comment warned about."""
    calls = _capture(monkeypatch)
    provider = hosted.HostedEmbeddingProvider()

    provider.embed(["a passage"], task="RETRIEVAL_DOCUMENT")
    provider.embed(["a question"], task="RETRIEVAL_QUERY")

    assert calls[0]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert calls[1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_a_short_response_is_rejected_rather_than_misaligned(monkeypatch):
    """Fewer vectors than inputs would silently shift every chunk's embedding."""

    def _post(url, params=None, json=None, timeout=None):
        return _Response(200, count=len(json["requests"]) - 1)

    monkeypatch.setattr(hosted.httpx, "post", _post)

    with pytest.raises(Exception):
        hosted.HostedEmbeddingProvider().embed(["a", "b", "c"])


def test_the_shadowed_duplicate_is_gone():
    """`src/lib/embeddings.py` was never imported — the package shadows it — so the two
    copies drifted and a reader could fix the wrong one."""
    from pathlib import Path

    repo = Path(__file__).parents[2]
    assert not (repo / "src" / "lib" / "embeddings.py").exists()
