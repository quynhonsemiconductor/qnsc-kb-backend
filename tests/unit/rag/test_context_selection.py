from src.domain.ai_service import _conflict_answer, _detect_explicit_conflicts, _select_context, classify_fact_type
from src.rag.citations import extract_citation_ids


def test_classify_fact_type_covers_every_label_the_detector_regex_produces():
    from src.domain.ai_service import _EXPLICIT_FACT_RE

    # The taxonomy is a lookup over the SAME closed label set the regex's first capture
    # group can produce -- this pins that every label the regex names has a bucket,
    # rather than silently falling back to "other" for one nobody noticed was missing.
    labelled_group = _EXPLICIT_FACT_RE.pattern.split("(", 2)[-1].split(")", 1)[0]
    labels = labelled_group.split("|")
    for label in labels:
        assert classify_fact_type(label) != "other", label


def test_classify_fact_type_falls_back_to_other_for_an_unknown_label():
    assert classify_fact_type("something new") == "other"


def test_context_selection_deduplicates_children_and_assigns_stable_source_ids(monkeypatch):
    monkeypatch.setattr("src.domain.ai_service.settings.RAG_MIN_CONTEXT_SCORE", 0.2)
    results = [
        {
            "chunk_id": "child-2",
            "parent_chunk_id": "parent-1",
            "article_id": "article-1",
            "title": "Policy",
            "parent_text": "The approved policy applies to all teams.",
            "score": 0.9,
        },
        {
            "chunk_id": "child-1",
            "parent_chunk_id": "parent-1",
            "article_id": "article-1",
            "title": "Policy",
            "parent_text": "The approved policy applies to all teams.",
            "score": 0.8,
        },
        {
            "chunk_id": "child-3",
            "parent_chunk_id": "parent-2",
            "article_id": "article-2",
            "title": "Runbook",
            "parent_text": "The runbook describes the escalation path.",
            "score": 0.7,
        },
    ]

    selected = _select_context(results)

    assert [item["source_id"] for item in selected] == ["C1", "C2"]
    assert selected[0]["child_chunk_ids"] == ["child-2", "child-1"]
    assert selected[0]["parent_chunk_id"] == "parent-1"


def test_context_selection_applies_score_and_budget_gates(monkeypatch):
    monkeypatch.setattr("src.domain.ai_service.settings.RAG_MIN_CONTEXT_SCORE", 0.8)
    monkeypatch.setattr("src.domain.ai_service.settings.RAG_CONTEXT_MAX_TOKENS", 5)
    results = [
        {
            "chunk_id": "child-1",
            "parent_chunk_id": "parent-1",
            "article_id": "article-1",
            "parent_text": "This passage is deliberately long enough to exceed the tiny test budget.",
            "score": 0.79,
        },
        {
            "chunk_id": "child-2",
            "parent_chunk_id": "parent-2",
            "article_id": "article-2",
            "parent_text": "Short passage.",
            "score": 0.9,
        },
    ]

    selected = _select_context(results)

    assert len(selected) == 1
    assert selected[0]["parent_chunk_id"] == "parent-2"


def test_citations_normalize_backend_ids_and_legacy_numeric_markers():
    assert extract_citation_ids("Answer [C2] and [Source ID: C1], plus [3].") == ["C1", "C2", "C3"]


def test_explicit_conflicts_are_presented_without_adjudication():
    results = [
        {
            "chunk_id": "chunk-1",
            "parent_chunk_id": "parent-1",
            "article_id": "article-1",
            "title": "Old policy",
            "section_ref": "Approval",
            "context_text": "Approval deadline: 5 business days.",
            "chunk_text": "Approval deadline: 5 business days.",
            "source_id": "C1",
            "child_chunk_ids": ["chunk-1"],
        },
        {
            "chunk_id": "chunk-2",
            "parent_chunk_id": "parent-2",
            "article_id": "article-2",
            "title": "Updated policy",
            "section_ref": "Approval",
            "context_text": "Approval deadline: 10 business days.",
            "chunk_text": "Approval deadline: 10 business days.",
            "source_id": "C2",
            "child_chunk_ids": ["chunk-2"],
        },
    ]

    conflicts = _detect_explicit_conflicts(results)
    answer, citations = _conflict_answer(conflicts)

    assert conflicts and conflicts[0]["fact"] == "approval deadline"
    assert "cannot determine which statement is current" in answer
    assert "5 business days" in answer and "10 business days" in answer
    assert "[C1]" in answer and "[C2]" in answer
    assert {item["source_id"] for item in citations} == {"C1", "C2"}
