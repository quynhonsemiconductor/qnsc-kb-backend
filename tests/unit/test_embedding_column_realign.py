"""The pgvector column width must track EMBEDDING_DIMENSION, and drift must be loud.

#77 moved EMBEDDING_MODEL to MiniLM (384) and claimed migration 20260810_51 would
re-align the column. It cannot: that revision is already in alembic_version everywhere
this is deployed, and Alembic never re-runs an applied revision — which is precisely why
20260810_51 had to exist instead of editing 20260810_36. The same trap, one revision on.

So the column stayed vector(1024) from the bge-m3 era while the model emitted 384, and
every article failed to index with

    asyncpg.exceptions.DataError: expected 1024 dimensions, not 384

showing in the UI only as "Search index: failed", and to the user as an assistant that
answers nothing about any published article.

Two things are pinned here: that a realign revision exists at the head of the chain for
the current dimension, and that a mismatch is reported at startup instead of being
discovered one failed article at a time.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from src.api import main as api_main
from src.core.config import settings

REPO = Path(__file__).parents[2]
VERSIONS = REPO / "migrations" / "versions"


def _realign_revisions() -> list[Path]:
    return sorted(VERSIONS.glob("*realign_embedding_dimension*.py"))


def test_a_realign_revision_exists_for_the_current_dimension():
    """Changing EMBEDDING_MODEL without a NEW realign revision leaves the old width in
    place, because Alembic will not re-run the previous one."""
    newest = _realign_revisions()[-1].read_text(encoding="utf-8")
    assert str(settings.EMBEDDING_DIMENSION) in newest, (
        f"no realign revision mentions the current EMBEDDING_DIMENSION "
        f"({settings.EMBEDDING_DIMENSION}); a model change needs its own revision"
    )


def test_every_realign_revision_reads_the_setting_rather_than_a_literal():
    """The width is derived, so a hardcoded ALTER would drift from the config again."""
    for path in _realign_revisions():
        text = path.read_text(encoding="utf-8")
        assert "settings.EMBEDDING_DIMENSION" in text, path.name


def test_the_realign_refuses_to_discard_existing_vectors():
    """Widths are not convertible, so silently dropping them must never be the default."""
    newest = _realign_revisions()[-1].read_text(encoding="utf-8")
    assert "embedding IS NOT NULL" in newest
    assert "raise RuntimeError" in newest


def test_the_realign_requeues_published_articles():
    """Nothing retries a failed index on its own, so a resize that leaves every article
    'failed' fixes the column and none of the symptoms."""
    newest = _realign_revisions()[-1].read_text(encoding="utf-8")
    assert "index_status = 'pending'" in newest


# ── the startup guard ─────────────────────────────────────────────────────────


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _Connection:
    def __init__(self, value):
        self._value = value

    async def execute(self, _statement):
        return _Result(self._value)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _Engine:
    def __init__(self, value):
        self._value = value

    def connect(self):
        return _Connection(self._value)


def _run_guard(monkeypatch, column_width, capsys):
    """structlog writes through PrintLoggerFactory, so the line lands on stdout rather
    than in caplog's stdlib records."""
    import src.api.deps as deps

    monkeypatch.setattr(deps, "engine", _Engine(column_width))
    asyncio.run(api_main.verify_embedding_column_width())
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_a_mismatched_column_is_reported_at_startup(monkeypatch, capsys):
    # Deliberately NOT a hardcoded literal (e.g. 1024): whatever that literal is, a future
    # EMBEDDING_MODEL swap can make it equal settings.EMBEDDING_DIMENSION again, at which
    # point this stops testing a mismatch at all -- exactly what happened here once
    # already when the default moved from e5-small (384) to e5-large-instruct (1024).
    mismatched_width = settings.EMBEDDING_DIMENSION + 1
    text = _run_guard(monkeypatch, mismatched_width, capsys)
    assert "Embedding column width does not match" in text
    # Both numbers must be present or the line does not save anyone a debugging round.
    assert str(mismatched_width) in text
    assert str(settings.EMBEDDING_DIMENSION) in text


def test_a_matching_column_is_silent(monkeypatch, capsys):
    text = _run_guard(monkeypatch, settings.EMBEDDING_DIMENSION, capsys)
    assert "Embedding column width does not match" not in text


def test_the_guard_never_blocks_boot(monkeypatch):
    """Diagnostics failing must not take the API down with them."""

    class _Broken:
        def connect(self):
            raise RuntimeError("database asleep")

    import src.api.deps as deps

    monkeypatch.setattr(deps, "engine", _Broken())
    asyncio.run(api_main.verify_embedding_column_width())  # must not raise
