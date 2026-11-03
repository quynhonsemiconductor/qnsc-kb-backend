from src.domain import source_extraction


def test_markitdown_is_bypassed_for_lossless_text_formats(monkeypatch):
    called = False

    def fail_if_called():
        nonlocal called
        called = True
        raise AssertionError("MarkItDown should not handle Markdown input")

    monkeypatch.setattr(source_extraction, "_markitdown", fail_if_called)
    for filename in ("procedure.md", "notes.txt", "rows.csv"):
        assert source_extraction._convert_with_markitdown(filename, b"content") == ""
    assert called is False


def test_office_documents_with_a_native_extractor_never_reach_markitdown(monkeypatch):
    """MarkItDown needs optional extras for .docx and .xlsx that are not installed: it raised, was logged as a
    failure, and the native extractor read the file anyway, on every upload. Not asked, not raising.
    """

    def fail_if_called():
        raise AssertionError(
            "MarkItDown must not be asked for a format with a native extractor"
        )

    monkeypatch.setattr(source_extraction, "_markitdown", fail_if_called)
    for filename in ("spec.docx", "pins.xlsx", "macro.xlsm", "SPEC.DOCX"):
        assert source_extraction._convert_with_markitdown(filename, b"PK") == ""


def test_a_pdf_still_goes_through_markitdown(monkeypatch):
    asked = []

    class _Converter:
        def convert_stream(self, *_args, **kwargs):
            asked.append(kwargs.get("file_extension"))
            raise ValueError("stop here")

    monkeypatch.setattr(source_extraction, "_markitdown", lambda: _Converter())
    source_extraction._convert_with_markitdown("a.pdf", b"%PDF-1.4")
    assert asked == [".pdf"]
