"""The worker must embed one article at a time, because it is sized for one session.

Celery defaults `--concurrency` to `os.cpu_count()`. On Fargate that reports the HOST's
CPU count, not the task's, so the prefork child count is neither 1 nor reliably related to
what the task was actually given — develop came up at 2 on a 2048-CPU-unit task.

Each prefork child loads its own embedding session. The e5-large-instruct ONNX session is
~1.5 GB resident, and the task budget in `infra/live/*/main.tf` allows exactly one: clamav
and beat hold hard container limits of 2048 and 256 MB out of 4096, leaving the worker
~1.8 GB, against which the sizing comment also reserves a per-file PaddleOCR spike.

Two children embedding concurrently exceed that. On develop, six articles picked up
between 15:25:50 and 15:26:34 all died with

    billiard.exceptions.WorkerLostError: Worker exited prematurely: signal 9 (SIGKILL)

and the damage outlived the kill: SIGKILL raises nothing in Python, so the `except` in
`src/domain/indexing.py:index_article` never ran and every row stayed at
`index_status="processing"` — which the UI renders as still indexing, forever, with
nothing working on it. The reconciliation sweep in `src/api/main.py` re-queues on
`chunk_count == 0` regardless of status, so those rows are recoverable, but only when
something triggers a sweep.

This was latent under e5-small (384 dims, a much smaller session) and only began failing
when develop moved to e5-large-instruct.

These are string assertions against the Dockerfile rather than a running container, for
the same reason as `test_embedding_config_matches_image.py`: the invariant lives in the
relationship between the image's CMD and the memory arithmetic in the infra files, so
inspecting a container built from that CMD would only restate one side of it.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

REPO = Path(__file__).parents[2]


def _worker_cmd() -> list[str]:
    """The CMD of the `worker` stage, as a token list."""
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    stage = re.search(
        r"(?ms)^FROM\s+\S+\s+AS\s+worker\s*$(.*?)(?=^FROM\s|\Z)",
        text,
    )
    assert stage, "Dockerfile no longer declares a `worker` stage"

    body = stage.group(1)
    # Line continuations first: the CMD is wrapped across several lines.
    body = body.replace("\\\n", " ")
    cmd = re.search(r"(?m)^CMD\s+(\[.*\])\s*$", body)
    assert cmd, "the worker stage no longer declares a JSON-array CMD"

    tokens = re.findall(r'"([^"]*)"', cmd.group(1))
    assert tokens, f"could not parse tokens out of worker CMD: {cmd.group(1)!r}"
    return tokens


def test_the_worker_pins_concurrency_to_one() -> None:
    tokens = _worker_cmd()

    assert "worker" in tokens, f"the worker CMD no longer runs a Celery worker: {tokens}"

    concurrency = [token for token in tokens if token.startswith("--concurrency")]
    assert concurrency, (
        "the worker CMD does not pass --concurrency, so Celery falls back to "
        "os.cpu_count() — on Fargate that is the host's CPU count, and every extra "
        "prefork child loads its own ~1.5 GB embedding session into a task sized for one"
    )
    assert concurrency == ["--concurrency=1"], (
        f"expected exactly --concurrency=1, got {concurrency}. Raising this requires "
        "re-sizing worker memory in infra/live/*/main.tf for N concurrent embedding "
        "sessions first — the current budget allows one"
    )


def test_beat_is_not_given_a_concurrency_flag() -> None:
    """Beat is a scheduler, not a pool; the flag would be meaningless there.

    Beat runs from this same image with its own command, supplied by the task definition
    rather than the Dockerfile. This test pins the division: `--concurrency` belongs to the
    worker CMD only, so a future edit that moves it to a shared ENTRYPOINT has to think
    about beat.
    """
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    beat_stage = re.search(
        r"(?ms)^FROM\s+\S+\s+AS\s+beat\s*$(.*?)(?=^FROM\s|\Z)",
        text,
    )
    if beat_stage is None:
        # Beat shares the worker image and is configured by the task definition. Nothing
        # to pin here, and asserting a stage exists would be inventing a requirement.
        return

    assert "--concurrency" not in beat_stage.group(1), (
        "beat is a singleton scheduler with no worker pool; --concurrency there is "
        "either a no-op or a sign the stages have been conflated"
    )


def test_the_worker_still_consumes_every_expected_queue() -> None:
    """Guards the edit itself: adding a flag must not drop a queue.

    The flag was inserted between `--loglevel` and `-Q`, which is exactly the kind of
    edit that silently loses a token in a line-continued CMD.
    """
    tokens = _worker_cmd()

    assert "-Q" in tokens, f"the worker CMD no longer passes -Q: {tokens}"
    queues = tokens[tokens.index("-Q") + 1]
    assert set(queues.split(",")) == {
        "celery",
        "ingestion",
        "connectors",
        "permissions",
    }, f"the worker's queue list changed: {queues!r}"


def test_the_cmd_is_still_a_valid_token_list() -> None:
    """A JSON-array CMD is exec form: tokens must not carry embedded shell syntax."""
    tokens = _worker_cmd()

    for token in tokens:
        assert token == token.strip(), f"token carries stray whitespace: {token!r}"
        assert shlex.split(token) == [token] or token.startswith("-"), (
            f"token would be re-split by a shell, which exec form does not run: {token!r}"
        )
