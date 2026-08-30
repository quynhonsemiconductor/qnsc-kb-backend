"""A failed structure layer must say which failure it was.

Every upload logged the same line:

    {"reason": "markitdown_unavailable_or_failed",
     "event": "Source Markdown conversion used page extractor"}

That string covers two unrelated situations — the package is not importable, or this
particular document defeated it — and both paths returned None or "" in silence. So a
dependency that had never loaded in any image looked exactly like one awkward PDF, and
the structure layer being dead on 100% of uploads went unnoticed. `markitdown` is
declared in the main dependency group and the api installs `--only main,ml`, so it is
supposed to be there.

There is also a real failure mode hiding in the loader: only TypeError was caught around
the constructor, so any other construction error escaped _markitdown() entirely — past
_convert_with_markitdown, which does not guard that call, and out through
extract_source_markdown, which the upload endpoint does not guard either. An optional
enhancement layer could return a 500 for the whole upload.
"""
from __future__ import annotations

import builtins

import pytest

from src.domain import source_extraction


@pytest.fixture(autouse=True)
def _clear_cache():
    source_extraction._markitdown.cache_clear()
    yield
    source_extraction._markitdown.cache_clear()


def test_an_unimportable_package_is_reported_once(monkeypatch, capsys):
    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if name == "markitdown":
            raise ImportError("No module named 'markitdown'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)

    assert source_extraction._markitdown() is None
    # readouterr() DRAINS the buffer, so it must be called once.
    captured = capsys.readouterr()
    assert "not importable" in captured.out + captured.err


def test_a_construction_failure_does_not_escape(monkeypatch, capsys):
    """The latent 500: only TypeError was caught, so anything else reached the upload."""

    class _Boom:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("optional backend missing")

    module = type("m", (), {"MarkItDown": _Boom})
    monkeypatch.setitem(__import__("sys").modules, "markitdown", module)

    assert source_extraction._markitdown() is None  # must not raise


def test_the_legacy_constructor_signature_still_works(monkeypatch):
    """MarkItDown 0.0.x has no plugin argument; that path must still return an instance
    rather than being swallowed by the new broad handler."""
    built = {}

    class _Legacy:
        def __init__(self, *args, **kwargs):
            if "enable_plugins" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_plugins'")
            built["ok"] = True

    module = type("m", (), {"MarkItDown": _Legacy})
    monkeypatch.setitem(__import__("sys").modules, "markitdown", module)

    assert source_extraction._markitdown() is not None
    assert built["ok"]


def test_a_conversion_failure_names_the_file_and_the_error(monkeypatch, capsys):
    class _Converter:
        def convert_stream(self, *_args, **_kwargs):
            raise ValueError("unsupported stream")

    monkeypatch.setattr(source_extraction, "_markitdown", lambda: _Converter())

    assert source_extraction._convert_with_markitdown("Lecture.pdf", b"%PDF-1.4") == ""

    output = capsys.readouterr().out
    assert "Lecture.pdf" in output
    assert "ValueError" in output


def test_a_disabled_layer_is_not_reported_as_a_failure(monkeypatch, capsys):
    """MARKITDOWN_ENABLED=false is a choice, not a fault, and must not log noise."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "MARKITDOWN_ENABLED", False)

    assert source_extraction._convert_with_markitdown("Lecture.pdf", b"x") == ""
    assert "failed" not in capsys.readouterr().out
