from src.rag.answer_sections import (
    EXTENDED_SENTINEL,
    GROUNDED_SENTINEL,
    render_answer_sections,
    split_answer_sections,
    strip_citation_markers,
)
from src.rag.citations import extract_citation_ids


def test_split_grounded_only():
    grounded, extended = split_answer_sections(f"  {GROUNDED_SENTINEL}  \nFact [C1].")
    assert grounded == "Fact [C1]."
    assert extended == ""


def test_split_grounded_and_extended():
    grounded, extended = split_answer_sections(
        f"{GROUNDED_SENTINEL}\nFact [C1].\n{EXTENDED_SENTINEL}\nGeneral explanation."
    )
    assert grounded == "Fact [C1]."
    assert extended == "General explanation."


def test_missing_grounded_defaults_entire_output_to_grounded():
    grounded, extended = split_answer_sections(f"Unmarked answer.\n{EXTENDED_SENTINEL}\nMore text.")
    assert grounded == "Unmarked answer.\nMore text."
    assert extended == ""


def test_sentinel_like_text_inside_code_fence_is_not_a_boundary():
    raw = f"{GROUNDED_SENTINEL}\n```text\n{EXTENDED_SENTINEL}\n```\nFact [C1]."
    grounded, extended = split_answer_sections(raw)
    assert EXTENDED_SENTINEL in grounded
    assert extended == ""


def test_extended_citations_are_removed_and_not_extracted():
    extended = strip_citation_markers("General statement [C1][C2].")
    assert extended == "General statement ."
    assert extract_citation_ids(extended) == []


def test_rendering_has_explicit_sections_only_when_enabled():
    rendered = render_answer_sections("Grounded [C1].", "General context.", enabled=True)
    assert "Answer from the Knowledge Base" in rendered
    assert "Additional context" in rendered
    assert render_answer_sections("Grounded [C1].", "General context.", enabled=False) == "Grounded [C1]."
