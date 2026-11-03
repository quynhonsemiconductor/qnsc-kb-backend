import asyncio
import io
import zipfile
from tempfile import SpooledTemporaryFile

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.core.config import settings
from src.api.routers.articles import _read_upload_limited
from src.domain.source_extraction import SourceExtractionError, _scan_with_clamd, _validate_source_bytes


def test_office_signature_is_validated():
    with pytest.raises(SourceExtractionError, match="invalid file signature"):
        _validate_source_bytes("policy.docx", b"not-a-zip")


def test_archive_file_count_is_limited(monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("one.txt", "one")
        archive.writestr("two.txt", "two")
    monkeypatch.setattr(settings, "MAX_SOURCE_ARCHIVE_FILES", 1)
    with pytest.raises(SourceExtractionError, match="too many files"):
        _validate_source_bytes("policy.docx", payload.getvalue())


def test_clamd_scan_rejects_unavailable_scanner(monkeypatch):
    monkeypatch.setattr(settings, "MALWARE_SCANNER_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "MALWARE_SCANNER_PORT", 1)
    with pytest.raises(SourceExtractionError, match="unavailable"):
        _scan_with_clamd(b"safe")


def test_upload_reader_enforces_limit_while_streaming(monkeypatch):
    monkeypatch.setattr(settings, "MAX_SOURCE_UPLOAD_BYTES", 3)
    stream = SpooledTemporaryFile()
    stream.write(b"four")
    stream.seek(0)
    upload = UploadFile(filename="too-large.txt", file=stream, size=None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_read_upload_limited(upload))
    assert exc.value.status_code == 413


def test_upload_reader_accepts_content_at_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_SOURCE_UPLOAD_BYTES", 4)
    stream = SpooledTemporaryFile()
    stream.write(b"four")
    stream.seek(0)
    upload = UploadFile(filename="allowed.txt", file=stream, size=None)

    assert asyncio.run(_read_upload_limited(upload)) == b"four"
