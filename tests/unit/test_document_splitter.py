from src.domain.document_splitter import split_document_candidates, splitter_metrics


def test_splitter_preserves_order_and_source_positions():
    text = "# Introduction\nA short introduction.\n\n## Procedure\n" + (
        "Follow the approved procedure. " * 100
    )
    candidates = split_document_candidates("Runbook", text)

    assert candidates
    assert [item["position"] for item in candidates] == list(
        range(1, len(candidates) + 1)
    )
    assert candidates[0]["heading"] == "Introduction"
    assert all(item["source_start"] < item["source_end"] for item in candidates)
    assert all(
        candidates[index]["source_start"] >= candidates[index - 1]["source_start"]
        for index in range(1, len(candidates))
    )
    assert "Procedure" in "\n".join(item["body_md"] for item in candidates)


def test_splitter_pilot_metrics_are_reproducible():
    report = splitter_metrics(
        [("one.md", "# One\nContent"), ("two.md", "# Two\nMore content")]
    )
    assert report["document_count"] == 2
    assert report["article_count"] >= 2
    assert report["manual_correction_count"] == 0
    assert report["manual_correction_share"] == 0.0


def test_splitter_keeps_ai_markdown_sections_in_one_review_document():
    text = """# Operations handbook

## Release process
Deploy only after the approved change window starts.

## Incident response
Record the incident number before contacting the on-call engineer.
"""

    candidates = split_document_candidates(
        "Operations handbook", text, prefer_markdown_sections=True
    )

    assert [item["title"] for item in candidates] == ["Operations handbook"]
    assert candidates[0]["body_md"].startswith(
        "# Operations handbook\n\n## Release process"
    )
    assert "## Incident response" in candidates[0]["body_md"]


def test_splitter_only_splits_exceptionally_long_review_documents():
    text = "# Handbook\n\n" + ("Approved procedure paragraph.\n\n" * 2_000)

    candidates = split_document_candidates("Handbook", text, prefer_markdown_sections=True)

    assert len(candidates) == 2
    assert all(len(item["body_md"]) <= 48_000 for item in candidates)
    assert "".join(item["body_md"] for item in candidates).replace("\n", "")
