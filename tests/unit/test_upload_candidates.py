from src.api.routers.articles import _add_split_candidates
from src.models.governance import PendingDraft


def test_split_candidates_follow_an_unflushed_draft_relationship():
    """Draft candidates must not need a UUID before the draft is flushed."""
    draft = PendingDraft(
        title="Runbook",
        company_domain="acme.test",
        source_ref="upload://runbook.md",
        source_hash="a" * 64,
        status="draft",
    )
    added = []

    class FakeDb:
        def add(self, item):
            added.append(item)

    _add_split_candidates(FakeDb(), draft, "# First\n" + ("Important step. " * 200))

    assert added
    assert all(candidate.draft is draft for candidate in added)
