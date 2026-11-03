"""Provenance belongs beside the source, not inside the sentence.

The context block hands the model a last-reviewed date and an owner email per document
and the model wrote them into the prose, so a three-claim paragraph read:

    … [last-reviewed: 2026-08-08T16:34:46.563958; owner: admin@example.com] [4]
    … [last-reviewed: 2026-08-08T16:34:46.563958; owner: admin@example.com] [2]

The citation payload already carries `last_reviewed` and `owner_email` for the source
card, so none of that ever needed to be in the answer text.
"""
from __future__ import annotations

from src.rag.answer_sections import strip_source_metadata
from src.rag.citations import extract_citation_ids


def test_the_blob_goes_and_the_citation_stays():
    answer = (
        "HDL là ngôn ngữ mô tả phần cứng "
        "[last-reviewed: 2026-08-08T16:34:46.563958; owner: admin.manager@qnsc.vn] [C4]"
    )

    cleaned = strip_source_metadata(answer)

    assert cleaned == "HDL là ngôn ngữ mô tả phần cứng [C4]"
    assert extract_citation_ids(cleaned) == ["C4"]


def test_repeated_blobs_across_a_paragraph_are_all_removed():
    answer = (
        "Điều một [last-reviewed: 2026-08-08T16:34:46; owner: a@qnsc.vn] [C4]. "
        "Điều hai [Last Reviewed: 2026-08-08T17:00:36; Owner: a@qnsc.vn] [C2]. "
        "Điều ba [ngày xem xét: 2026-08-08; email chủ sở hữu: a@qnsc.vn] [C5]."
    )

    cleaned = strip_source_metadata(answer)

    assert "owner" not in cleaned.lower()
    assert "reviewed" not in cleaned.lower()
    assert "qnsc.vn" not in cleaned
    assert extract_citation_ids(cleaned) == ["C2", "C4", "C5"]


def test_an_owner_email_is_never_mistaken_for_a_citation():
    """`owner: c4@example.com` contains a `C4` at a word boundary. Stripping happens
    before extraction precisely so a contact address cannot become a source."""
    answer = "Quy định áp dụng [last-reviewed: 2026-01-01; owner: c4@example.com] [C1]"

    assert extract_citation_ids(strip_source_metadata(answer)) == ["C1"]


def test_markdown_hard_line_breaks_survive():
    """Two TRAILING spaces are a line break in Markdown. Collapsing whitespace without
    care joins the model's list items into a single paragraph."""
    answer = "- Điều một [C1]  \n- Điều hai [C2]  \n"

    assert strip_source_metadata(answer) == "- Điều một [C1]  \n- Điều hai [C2]"


def test_ordinary_bracketed_text_is_left_alone():
    """Only provenance blobs are removed; the answer's own asides are content."""
    answer = "Verilog [xem thêm ví dụ] và VHDL [C3] đều là HDL."

    assert strip_source_metadata(answer) == answer
