from types import SimpleNamespace

from src.workers.tasks import (
    _draft_accepts_restructure_result,
    _should_suggest_draft_tags,
)


def test_worker_only_finalizes_the_claimed_pending_processing_draft():
    assert _draft_accepts_restructure_result(
        SimpleNamespace(status="pending", restructure_status="processing")
    )

    for draft in (
        None,
        SimpleNamespace(status="approved", restructure_status="processing"),
        SimpleNamespace(status="rejected", restructure_status="processing"),
        SimpleNamespace(status="pending", restructure_status="queued"),
        SimpleNamespace(status="pending", restructure_status="lossless_ready"),
    ):
        assert not _draft_accepts_restructure_result(draft)


def test_worker_skips_optional_tag_ai_for_oversized_lossless_results():
    draft = SimpleNamespace(tags=[])

    assert not _should_suggest_draft_tags(draft, auxiliary_ai_allowed=False)
    assert _should_suggest_draft_tags(draft, auxiliary_ai_allowed=True)
    assert not _should_suggest_draft_tags(
        SimpleNamespace(tags=["existing"]), auxiliary_ai_allowed=True
    )
