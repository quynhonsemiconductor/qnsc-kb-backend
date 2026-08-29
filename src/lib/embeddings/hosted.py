"""Embeddings from an HTTP API (Gemini's batchEmbedContents shape).

Not the default: document and query text leaves the deployment, which is the property the
in-process backends exist to keep. Kept working because it is the fallback when a
deployment cannot afford to carry a model at all, and because it is what the hosted
option costs in code — a few dozen lines — if that trade is ever re-made.
"""
from __future__ import annotations

import time

import httpx
import structlog

from src.core.config import settings
from src.lib.embeddings.base import EmbeddingUnavailable

logger = structlog.get_logger()


class HostedEmbeddingProvider:
    name = "hosted"

    def warm_up(self) -> None:
        """Nothing to load — the model lives on the other side of the wire."""

    def embed(
        self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed through the batch API, in bounded batches, with retries.

        Three things were wrong here, all dormant only because the deployment runs a
        local model. Any switch to a hosted one would have met them immediately.

        BATCHING. Every chunk of a document went into a single batchEmbedContents
        request — 87 of them for a 38-page PDF, thousands for a large one — against an
        API that bounds both the number of entries and their total tokens. The whole
        document would have failed at once. EMBEDDING_BATCH_SIZE existed and was
        honoured by exactly one backend.

        RETRIES. `raise_for_status()` turned a 429 or a transient 503 into a failed
        index for the entire article, with no second attempt. Rate limits are the
        expected condition for a hosted embedding API during bulk ingestion, not an
        exceptional one.

        TASK TYPE. This was hardcoded to RETRIEVAL_QUERY for everything, including the
        chunks being indexed — directly contradicting the comment beside it, which warns
        that a question and a passage are embedded differently and that mixing them
        degrades retrieval quietly. It now comes from the caller.
        """
        if not settings.GEMINI_API_KEY:
            raise EmbeddingUnavailable(
                "GEMINI_API_KEY is not set, and EMBEDDING_MODEL "
                f"({settings.EMBEDDING_MODEL!r}) is a hosted model."
            )
        if not texts:
            return []

        model = settings.EMBEDDING_MODEL
        url = f"{settings.GEMINI_API_BASE_URL.rstrip('/')}/models/{model}:batchEmbedContents"
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            window = texts[start : start + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task,
                        # gemini-embedding-001 returns 3072 dimensions by default and
                        # pgvector's HNSW index refuses to build above 2000. Ask for the
                        # width the column was actually created with.
                        "outputDimensionality": settings.EMBEDDING_DIMENSION,
                    }
                    for text in window
                ]
            }
            body = self._post_with_retries(url, payload)
            batch = [item["values"] for item in body.get("embeddings", [])]
            if len(batch) != len(window):
                raise EmbeddingUnavailable(
                    f"embedding API returned {len(batch)} vectors for {len(window)} inputs"
                )
            # Truncated vectors come back UNNORMALISED — measured L2 of 0.586 at 768
            # dimensions, where the full-width output is unit length. The seam
            # normalises everything, so that is handled there rather than here.
            vectors.extend(batch)

        return vectors

    @staticmethod
    def _post_with_retries(url: str, payload: dict) -> dict:
        """POST, retrying rate limits and transient server errors with backoff.

        Only 429 and 5xx are retried. A 400 means the request itself is wrong and
        repeating it just spends the quota again.
        """
        attempts = max(1, settings.EMBEDDING_HTTP_MAX_ATTEMPTS)
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    url,
                    params={"key": settings.GEMINI_API_KEY},
                    json=payload,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == attempts:
                    raise
                last = exc
                delay = settings.EMBEDDING_HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding API call failed; retrying",
                    attempt=attempt,
                    of=attempts,
                    status=status,
                    retry_in_seconds=round(delay, 2),
                )
                time.sleep(delay)
        raise last if last else EmbeddingUnavailable("embedding API call failed")
