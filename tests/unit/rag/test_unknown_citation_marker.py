"""An invented citation must cost the marker, not the whole answer.

MEASURED IN PRODUCTION. Asking "CTS la gi" against kb-dev produced a correct, well-cited
Vietnamese answer citing [C1] and [C2] — both retrieved. When the model additionally
invented a marker for a passage that was never retrieved, the guard failed closed and
replaced the entire reply with:

    "Không thể tạo câu trả lời có căn cứ từ các nguồn được cấp quyền trong Cơ sở tri thức."

which reads as "the knowledge base has nothing" — while the knowledge base had the answer,
the model had written it, and two of its three citations were perfectly valid. The user
watched the answer stream in and then get thrown away.

The guard's real purpose is that nothing in the answer may be attributed to a source that
was not consulted. Stripping the dangling marker preserves that exactly, without punishing
the reader for the model's arithmetic.
"""
from __future__ import annotations

from src.rag.citations import extract_citation_ids, strip_unknown_markers

RETRIEVED = {"C1", "C2"}

#: The shape observed in production: two good citations, one invented.
PRODUCTION_ANSWER = (
    "**CTS (Clock-tree synthesis)** là bước xây dựng cây đồng hồ trong quá trình "
    "thiết kế chip [C1][C2].\n\n"
    "- Các công cụ CTS thương mại: Fusion Compiler CTS, Innovus CCOpt [C1].\n"
    "- Các công cụ mã mở: OpenROAD CTS, TritonCTS 2.0 [C3].\n"
)


def test_the_valid_citations_survive_an_invented_one():
    """The regression: [C1] and [C2] used to be discarded along with [C3]."""
    stripped = strip_unknown_markers(PRODUCTION_ANSWER, RETRIEVED)

    assert extract_citation_ids(stripped) == ["C1", "C2"]


def test_the_answer_text_itself_survives():
    stripped = strip_unknown_markers(PRODUCTION_ANSWER, RETRIEVED)

    assert "Clock-tree synthesis" in stripped
    assert "Innovus CCOpt" in stripped
    # The sentence that carried the bad marker is kept; only the marker goes.
    assert "OpenROAD CTS" in stripped


def test_no_marker_naming_an_unretrieved_passage_remains():
    """The property the guard exists to protect, still enforced."""
    stripped = strip_unknown_markers(PRODUCTION_ANSWER, RETRIEVED)

    assert "C3" not in stripped
    assert all(marker in RETRIEVED for marker in extract_citation_ids(stripped))


def test_a_grouped_marker_keeps_only_the_retrieved_half():
    stripped = strip_unknown_markers("Balanced skew [C1, C3] applies.", RETRIEVED)

    assert extract_citation_ids(stripped) == ["C1"]
    assert "C3" not in stripped


def test_an_answer_citing_nothing_real_is_left_uncited():
    """Every marker invented: the caller must then refuse, so none may remain."""
    stripped = strip_unknown_markers("Entirely invented [C7][C9].", RETRIEVED)

    assert extract_citation_ids(stripped) == []


def test_years_and_footnotes_are_not_treated_as_citations():
    """Promoting a bracketed year to a citation is what fails an answer closed."""
    text = "Published [2026] and revised [1999] with source [C1]."
    stripped = strip_unknown_markers(text, RETRIEVED)

    assert "[2026]" in stripped
    assert "[1999]" in stripped
    assert extract_citation_ids(stripped) == ["C1"]


def test_markdown_links_are_untouched():
    text = "See [the OpenROAD docs](https://openroad.readthedocs.io) and [C2]."
    stripped = strip_unknown_markers(text, RETRIEVED)

    assert "[the OpenROAD docs](https://openroad.readthedocs.io)" in stripped
    assert extract_citation_ids(stripped) == ["C2"]


def test_removing_a_marker_does_not_leave_a_gap_before_punctuation():
    """Cosmetic, but a doubled space before a full stop is visible to the reader."""
    stripped = strip_unknown_markers("Routing is repaired [C1] [C3].", RETRIEVED)

    assert stripped == "Routing is repaired [C1]."


def test_an_answer_with_only_valid_markers_is_returned_unchanged():
    text = "Clock skew is balanced [C1][C2]."

    assert strip_unknown_markers(text, RETRIEVED) == text


def test_empty_input_is_handled():
    assert strip_unknown_markers("", RETRIEVED) == ""
    assert strip_unknown_markers(None, RETRIEVED) is None


def test_the_service_strips_rather_than_discarding():
    """Pin the call site: reverting to a wholesale replacement reintroduces the bug."""
    import inspect

    from src.domain import ai_service

    source = inspect.getsource(ai_service.AIService.ask)

    assert "strip_unknown_markers(grounded_answer" in source
    # The refusal string must no longer be assigned inside the unknown-marker branch.
    branch = source.split("if unknown_markers:")[1].split("citation_guard_failed =")[0]
    assert "Không thể tạo câu trả lời" not in branch
