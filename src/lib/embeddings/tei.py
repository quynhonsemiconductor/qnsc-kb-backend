"""Embeddings from the shared in-cluster TEI service (ONPREM-PLAN §9.1).

WHY THIS EXISTS. The in-process export costs ~2.3 GB of weights in BOTH the api and the
worker, and the api carries the whole model to embed one short query per search. The
on-prem platform runs one GPU node, so the model is served once, for every product, by
text-embeddings-inference in the `ai` namespace (`platform/ai/tei-e5.yaml` in the gitops
repo). Moving KB onto it takes the weights out of both images; the vector space does not
move, because the Service runs the SAME model, revision and dtype the corpus was embedded
with (fp32, mean pooling), and this client sends the exact string it wants embedded.

THE INSTRUCTION PREFIX STAYS CLIENT-SIDE. TEI is started without `--default-prompt`, so
it never decorates the input: `src.lib.embeddings._decorate` builds the
`Instruct: …\\nQuery: …` string here, exactly as the in-process backends need it. A server
prompt would apply it twice, or to passages that must not have it.

THE URL IS OPERATOR CONFIGURATION, NOT USER INPUT, and it is validated as such:
`validate_in_cluster_endpoint` accepts an in-cluster Service name and nothing else. This is
a DELIBERATE departure from `src.domain.llm_endpoint_security`, which exists for endpoints
an administrator types into the UI and therefore permits only public HTTPS — the exact
opposite of what an in-cluster Service looks like. Resolving `tei-e5.ai.svc` to a pod IP
and then refusing it as "private" would break the only destination this is for.

BATCHING AND BACKPRESSURE. The Service is started with `--max-client-batch-size=32` and
`--max-batch-tokens=16384`; a larger request is rejected (413), not silently split. So the
window is capped at 32 here, and a 413 is answered by halving the window and retrying:
32 chunks that each reach the 512-token cap are 16384 tokens, exactly the server's limit,
and the margin shrinks once the tokenizer counts its special tokens. Retrying a batch that
is too big against the same limit would fail forever, so the split is the fix, not a retry
with the same shape.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import structlog

from src.core.config import settings
from src.lib.embeddings.base import EmbeddingUnavailable

logger = structlog.get_logger()

#: `--max-client-batch-size=32` in platform/ai/tei-e5.yaml. Raise both together.
TEI_MAX_BATCH = 32

# One DNS-1123 label. Service names are lower-case; a name with an upper-case letter or an
# underscore would not resolve in the cluster, so refusing it here names the mistake at
# boot rather than as a connection error on the first search.
_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")


def validate_in_cluster_endpoint(value: str) -> str:
    """Normalise TEI_EMBEDDINGS_URL, allowing exactly `http://<service>.<namespace>.svc`.

    An optional port is accepted. Everything else is refused — an https:// public host, a
    bare IP, a path, credentials, a query — because the only destination this setting has
    is a Service in the same cluster, and a setting that can name an arbitrary host is one
    that can be repointed at the internet or at a node's own ports. There is no DNS
    resolution here: the name is handed to the cluster's DNS, which is what makes it an
    in-cluster name in the first place.

    Kubernetes already resolves the short form in-cluster, so `tei-e5.ai.svc` is the whole
    contract; a fully qualified `…svc.cluster.local` is refused to keep one spelling for
    one Service (and because the short form is what the manifests and values files use).
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        raise EmbeddingUnavailable(
            "EMBEDDING_RUNTIME=tei requires TEI_EMBEDDINGS_URL, e.g. http://tei-e5.ai.svc"
        )

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise EmbeddingUnavailable(
            f"TEI_EMBEDDINGS_URL={value!r} has an invalid port"
        ) from exc

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "http"
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path
    ):
        raise EmbeddingUnavailable(
            f"TEI_EMBEDDINGS_URL={value!r} must be a credential-free http://<service>"
            ".<namespace>.svc URL without a path or query"
        )

    labels = hostname.split(".")
    if (
        len(labels) != 3
        or labels[2] != "svc"
        or not all(_LABEL.fullmatch(x) for x in labels[:2])
    ):
        raise EmbeddingUnavailable(
            f"TEI_EMBEDDINGS_URL={value!r} must name an in-cluster Service "
            "(<service>.<namespace>.svc); a public host, an IP address or a fully "
            "qualified .svc.cluster.local name is not accepted"
        )

    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunsplit(("http", netloc, "", "", ""))


def _endpoint() -> str:
    return validate_in_cluster_endpoint(settings.TEI_EMBEDDINGS_URL)


class TeiEmbeddingProvider:
    name = "tei"

    def __init__(self) -> None:
        # Validated on construction, so `warm_up()` at API startup reports a misconfigured
        # URL with its own name instead of the first search failing and falling back to
        # keyword-only, where nobody would look for it.
        self._url = _endpoint()

    def warm_up(self) -> None:
        """Nothing to load — the model lives in the other pod.

        Deliberately no health check here either: readiness gates on this call, and tying a
        KB pod's readiness to the GPU node would take search down whenever the shared
        service is restarted. `_embed` fails loudly per request, which is what the caller's
        keyword fallback is for.
        """

    def embed(
        self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        if not texts:
            return []

        # `task` is a hosted-provider concern: TEI takes the string it is given, and the
        # instruction prefix is already applied by the seam (see this module's docstring).
        batch_size = max(1, min(settings.EMBEDDING_BATCH_SIZE, TEI_MAX_BATCH))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_window(texts[start : start + batch_size]))
        return vectors

    def _embed_window(self, window: list[str]) -> list[list[float]]:
        """POST one window to /embed, splitting it when the server says it is too big."""
        attempts = max(1, settings.EMBEDDING_HTTP_MAX_ATTEMPTS)
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    f"{self._url}/embed",
                    json={"inputs": window},
                    timeout=settings.TEI_EMBEDDINGS_TIMEOUT_SECONDS,
                )
                if response.status_code == 413 and len(window) > 1:
                    # TEI's documented remedy for an oversized batch is a smaller one:
                    # half and recurse, so the split is proportional to how far over the
                    # token budget this window actually was. The recursion is done AFTER
                    # this loop, not here — see below.
                    break
                response.raise_for_status()
                return self._parse(response.json(), len(window))
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if status == 413:
                    # A single input that still exceeds the budget: retrying cannot help.
                    raise EmbeddingUnavailable(
                        "TEI rejected a single input as larger than its "
                        "--max-batch-tokens budget; lower EMBEDDING_MAX_TOKENS on the "
                        "client or raise the limit on the server"
                    ) from exc
                if not retryable or attempt == attempts:
                    raise
                last = exc
                delay = settings.EMBEDDING_HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "TEI embedding call failed; retrying",
                    attempt=attempt,
                    of=attempts,
                    status=status,
                    retry_in_seconds=round(delay, 2),
                )
                time.sleep(delay)
        else:
            raise last if last else EmbeddingUnavailable("TEI embedding call failed")

        # Split outside the retry `try` on purpose. A half's failure is the half's: if the
        # recursive calls ran inside the `try` above, this frame's `except` would catch a
        # half that exhausted its own retries, discard the split, and re-POST the whole
        # un-split window. That window 413s again, splits again, and fails again — turning
        # a bounded `attempts` policy into one that multiplies by depth (up to
        # attempts**depth requests, each layer also re-paying its exponential backoff).
        middle = len(window) // 2
        logger.info(
            "TEI rejected a batch as too large; splitting it",
            batch_size=len(window),
            first_half=middle,
        )
        return self._embed_window(window[:middle]) + self._embed_window(window[middle:])

    @staticmethod
    def _parse(body: Any, expected: int) -> list[list[float]]:
        # `/embed` answers a list of inputs with a list of vectors. A bare float list would
        # be the single-string shape; treating it as one vector keeps the count check below
        # honest instead of zipping characters.
        if not isinstance(body, list) or not body:
            raise EmbeddingUnavailable(
                f"TEI /embed returned {type(body).__name__}, expected a list of vectors"
            )
        if isinstance(body[0], (int, float)):
            body = [body]
        if len(body) != expected:
            raise EmbeddingUnavailable(
                f"TEI /embed returned {len(body)} vectors for {expected} inputs"
            )
        return [[float(value) for value in vector] for vector in body]
