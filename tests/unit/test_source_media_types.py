"""What a browser is allowed to be handed for an original upload.

The multipart Content-Type is chosen by whoever uploaded the file, so the response type
is decided here instead. Anything that could be parsed as markup would execute in the
application's origin the moment a reviewer opened the source.
"""
from __future__ import annotations

from src.domain.source_storage import safe_source_media_type, source_should_display_inline


def test_markup_never_gets_a_type_a_browser_will_execute():
    for filename in ("payload.html", "payload.htm", "payload.svg", "payload.xml", "a.js"):
        assert safe_source_media_type(filename) == "application/octet-stream"
        assert not source_should_display_inline(filename)


def test_an_extensionless_or_unknown_upload_is_a_download():
    assert safe_source_media_type("archive") == "application/octet-stream"
    assert safe_source_media_type(None) == "application/octet-stream"
    assert safe_source_media_type("report.docx") == "application/octet-stream"


def test_text_sources_are_previewable_as_plain_text():
    """Most of this corpus is .md and .txt. Serving those as octet-stream made every
    one of them an undisplayable download in the source viewer.

    text/plain is never parsed as markup, and the response also sets nosniff, so a .md
    file full of HTML is shown as the characters it contains.
    """
    for filename in ("runbook.md", "notes.txt", "rows.csv", "worker.log"):
        assert safe_source_media_type(filename) == "text/plain; charset=utf-8"
        assert source_should_display_inline(filename)


def test_documents_and_images_keep_their_own_types():
    assert safe_source_media_type("Lecture-3.pdf") == "application/pdf"
    assert safe_source_media_type("diagram.PNG") == "image/png"
    assert safe_source_media_type("photo.jpeg") == "image/jpeg"
