"""The TEI client: the URL it will connect to, how it batches, and how it degrades.

The three things these tests pin, in order of how quietly they fail in production:

  THE ENDPOINT. `TEI_EMBEDDINGS_URL` is operator configuration, so it is validated against
  exactly one shape — an in-cluster Service name. A setting that can point anywhere is a
  setting that gets pointed somewhere it should not be.

  THE BATCH. The Service refuses more than 32 inputs and more than 16384 tokens in one
  request. Order is part of the contract: callers zip vectors back onto their chunks, so a
  reordered or short answer mislabels every embedding after the gap.

  THE FAILURE PATH. A transient error is retried; a request that is wrong in shape is not;
  a batch the server calls too large is SPLIT, because repeating the same shape against the
  same limit fails forever. A single input too large is a configuration fault, not a
  transient one, and must surface as such.
"""

from __future__ import annotations

import httpx
import pytest

from src.core.config import settings
from src.lib import embeddings
from src.lib.embeddings import tei
from src.lib.embeddings.base import EmbeddingUnavailable


class _Response:
    def __init__(self, status: int = 200, count: int = 0):
        self.status_code = status
        self._count = count

    def json(self):
        return [[0.5] * settings.EMBEDDING_DIMENSION for _ in range(self._count)]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(
        settings, "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
    )
    monkeypatch.setattr(settings, "EMBEDDING_RUNTIME", "tei")
    monkeypatch.setattr(settings, "TEI_EMBEDDINGS_URL", "http://tei-e5.ai.svc")
    monkeypatch.setattr(settings, "EMBEDDING_HTTP_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(tei.time, "sleep", lambda _seconds: None)


class _JsonResponse:
    """A server that answers with vectors the test builds, not a constant."""

    def __init__(self, body):
        self.status_code = 200
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


def _serve(monkeypatch, statuses=None):
    """Record every POST body; `statuses` drives failures before the eventual success."""
    calls: list[list[str]] = []
    queue = list(statuses or [])

    def _post(url, json=None, timeout=None):
        calls.append(json["inputs"])
        status = queue.pop(0) if queue else 200
        return _Response(status, count=len(json["inputs"]))

    monkeypatch.setattr(tei.httpx, "post", _post)
    return calls


# ── The endpoint guard ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://tei-e5.ai.svc", "http://tei-e5.ai.svc"),
        ("http://tei-e5.ai.svc/", "http://tei-e5.ai.svc"),
        ("  http://tei-e5.ai.svc:80  ", "http://tei-e5.ai.svc:80"),
    ],
)
def test_an_in_cluster_service_url_is_accepted(value, expected):
    assert tei.validate_in_cluster_endpoint(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "https://tei-e5.ai.svc",  # the Service speaks plain HTTP on the cluster network
        "http://huggingface.co",  # a public host is not this setting's job
        "http://10.43.121.193",  # the ClusterIP: an address, not the Service contract
        "http://tei-e5.ai.svc.cluster.local",  # one spelling per Service
        "http://tei-e5.other.svc:8080/embed",  # a path would be appended to /embed
        "http://user:pass@tei-e5.ai.svc",
        "http://tei-e5.ai.svc?x=1",
        "http://localhost:8000",
        "tei-e5.ai.svc",
    ],
)
def test_anything_but_an_in_cluster_service_url_is_refused(value):
    with pytest.raises(EmbeddingUnavailable):
        tei.validate_in_cluster_endpoint(value)


def test_a_missing_url_is_a_named_configuration_fault(monkeypatch):
    monkeypatch.setattr(settings, "TEI_EMBEDDINGS_URL", "")

    with pytest.raises(EmbeddingUnavailable, match="TEI_EMBEDDINGS_URL"):
        embeddings.resolve_provider()


def test_the_runtime_selects_tei_before_the_hosted_branch(monkeypatch):
    """The model name stays a local-e5 name under TEI — it names what the Service serves."""
    assert embeddings.resolve_provider().name == "tei"


# ── Batching ─────────────────────────────────────────────────────────────────
def test_batches_are_capped_at_the_services_client_limit(monkeypatch):
    monkeypatch.setattr(
        settings, "EMBEDDING_BATCH_SIZE", 64
    )  # a copied value must not win
    calls = _serve(monkeypatch)
    texts = [f"chunk {i}" for i in range(70)]

    vectors = embeddings.get_bge_embeddings(texts)

    assert [len(call) for call in calls] == [32, 32, 6]
    assert len(vectors) == 70
    # Order is the contract: the caller zips these onto its chunk list, so the windows
    # must concatenate back to the input in the input's order.
    assert [text for call in calls for text in call] == texts


def test_the_instruction_prefix_is_what_reaches_the_service(monkeypatch):
    """TEI is started without --default-prompt, so the client decorates — exactly once."""
    calls = _serve(monkeypatch)

    embeddings.get_bge_embedding("annual leave policy")

    (sent,) = calls
    assert sent == [
        "Instruct: Given a search query, retrieve relevant passages that answer the "
        "query\nQuery: annual leave policy"
    ]


def test_a_passage_is_sent_undecorated(monkeypatch):
    """The instruct-tuned e5 convention: passages take NO prefix."""
    calls = _serve(monkeypatch)

    embeddings.get_bge_embeddings(["a passage"])

    assert calls == [["a passage"]]


def test_no_texts_makes_no_request(monkeypatch):
    calls = _serve(monkeypatch)

    assert embeddings.get_bge_embeddings([]) == []
    assert calls == []


def test_warm_up_does_not_call_the_service(monkeypatch):
    calls = _serve(monkeypatch)

    embeddings.warm_up()

    assert calls == []


# ── Retries and the oversize batch ───────────────────────────────────────────
def test_a_transient_error_is_retried(monkeypatch):
    calls = _serve(monkeypatch, statuses=[503, 503])

    vectors = embeddings.get_bge_embeddings(["one"])

    assert len(calls) == 3
    assert len(vectors) == 1


def test_a_rejected_request_is_not_retried(monkeypatch):
    calls = _serve(monkeypatch, statuses=[400])

    with pytest.raises(RuntimeError, match="Embedding batch generation failed"):
        embeddings.get_bge_embeddings(["one"])

    assert len(calls) == 1


def test_vectors_come_back_in_the_callers_order(monkeypatch):
    """Each fake vector is one-hot on its input's index, so a reorder is visible."""

    def _post(url, json=None, timeout=None):
        body = []
        for text in json["inputs"]:
            index = int(text.rsplit(" ", 1)[-1])
            body.append([1.0 if position == index else 0.0 for position in range(1024)])
        return _JsonResponse(body)

    monkeypatch.setattr(tei.httpx, "post", _post)

    vectors = embeddings.get_bge_embeddings([f"chunk {i}" for i in range(40)])

    assert [vector.index(max(vector)) for vector in vectors] == list(range(40))


def test_an_oversize_batch_is_split_rather_than_retried_unchanged(monkeypatch):
    """413 is answered by halving: the same 32 inputs would be refused again forever."""
    sizes: list[int] = []

    def _post(url, json=None, timeout=None):
        sizes.append(len(json["inputs"]))
        if len(json["inputs"]) > 8:
            return _Response(413)
        return _Response(200, count=len(json["inputs"]))

    monkeypatch.setattr(tei.httpx, "post", _post)

    vectors = embeddings.get_bge_embeddings([f"chunk {i}" for i in range(32)])

    assert len(vectors) == 32
    # 32 → (16, 16); each 16 → (8, 8). The split stops at the size the server takes.
    assert sizes == [32, 16, 8, 8, 16, 8, 8]


def test_a_single_oversize_input_is_a_configuration_fault(monkeypatch):
    _serve(monkeypatch, statuses=[413])

    with pytest.raises(
        RuntimeError, match="Embedding batch generation failed"
    ) as raised:
        embeddings.get_bge_embeddings(["one"])

    assert isinstance(raised.value.__cause__, EmbeddingUnavailable)


def test_a_short_answer_is_refused_rather_than_misaligned(monkeypatch):
    def _post(url, json=None, timeout=None):
        return _Response(200, count=len(json["inputs"]) - 1)

    monkeypatch.setattr(tei.httpx, "post", _post)

    with pytest.raises(RuntimeError, match="Embedding batch generation failed"):
        embeddings.get_bge_embeddings(["one", "two"])


def test_the_seam_still_normalises_tei_vectors(monkeypatch):
    """TEI's output is raw; base.finalise is the one place that makes it unit length."""
    _serve(monkeypatch)
    provider = embeddings.resolve_provider()

    vector = embeddings.get_bge_embedding("anything")

    assert pytest.approx(sum(value * value for value in vector) ** 0.5, rel=1e-9) == 1.0
    assert provider.name == "tei"
