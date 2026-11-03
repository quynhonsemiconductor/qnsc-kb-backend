import asyncio
import uuid

import pytest
from fastapi import HTTPException

from src.domain.governance import GovernanceService
from src.models.governance import DraftCandidate, PendingDraft
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User


def make_reviewer(email: str = "reviewer@acme.test") -> User:
    user = User(
        id=uuid.uuid4(),
        email=email,
        name=email,
        company_domain="acme.test",
        role="Reviewer",
        active=True,
    )
    permission = Permission(key="article.review", name="Review")
    role = Role(name="Reviewer", company_domain="acme.test", active=True)
    role.permissions.append(RolePermission(permission=permission, scope="company"))
    user.roles.append(role)
    return user


class BatchDb:
    def __init__(self, repository):
        self.repository = repository
        self.added = []

    def add(self, item):
        if getattr(item, "id", None) is None:
            item.id = uuid.uuid4()
        self.added.append(item)
        if isinstance(item, PendingDraft):
            self.repository.drafts[item.id] = item
        elif isinstance(item, DraftCandidate):
            self.repository.drafts[item.draft_id].candidates.append(item)

    async def flush(self):
        return None

    async def commit(self):
        return None


class BatchRepository:
    def __init__(self, draft: PendingDraft):
        self.drafts = {draft.id: draft}
        self.db = BatchDb(self)
        self.audits = []

    async def get_draft(self, draft_id):
        return self.drafts.get(draft_id)

    async def update_draft(self, draft):
        self.drafts[draft.id] = draft
        return draft

    async def log_audit(self, audit):
        self.audits.append(audit)
        return audit


def make_draft(*candidates: DraftCandidate) -> PendingDraft:
    draft = PendingDraft(
        id=uuid.uuid4(),
        title="Operations handbook",
        company_domain="acme.test",
        dept="Engineering",
        source_ref="upload://handbook.md",
        source_hash="a" * 64,
        created_by=uuid.uuid4(),
        status="pending",
    )
    draft.candidates = list(candidates)
    return draft


def make_candidate(draft_id: uuid.UUID, position: int, title: str, body: str, start: int, end: int) -> DraftCandidate:
    return DraftCandidate(
        id=uuid.uuid4(),
        draft_id=draft_id,
        position=position,
        title=title,
        body_md=body,
        source_start=start,
        source_end=end,
        heading=title,
        status="candidate",
    )


def test_batch_review_can_rename_and_discard_candidates():
    draft = make_draft()
    first = make_candidate(draft.id, 1, "Old title", "First body", 0, 10)
    second = make_candidate(draft.id, 2, "Second", "Second body", 10, 21)
    draft.candidates = [first, second]
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())
    user = make_reviewer()

    asyncio.run(service.review_candidate(user, draft.id, "rename", first.id, title="Renamed title"))
    asyncio.run(service.review_candidate(user, draft.id, "discard", second.id, note="Out of scope"))

    assert first.title == "Renamed title"
    assert second.status == "discarded"
    assert second.review_note == "Out of scope"
    assert [audit.action for audit in repo.audits] == ["candidate_rename", "candidate_discard"]


def test_batch_review_merge_uses_source_order_and_marks_other_discarded():
    draft = make_draft()
    first = make_candidate(draft.id, 1, "First", "A", 0, 1)
    second = make_candidate(draft.id, 2, "Second", "B", 1, 2)
    draft.candidates = [first, second]
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())

    asyncio.run(service.review_candidate(make_reviewer(), draft.id, "merge", second.id, first.id, title="Combined"))

    assert first.body_md == "A\n\nB"
    assert first.title == "Combined"
    assert first.source_end == 2
    assert second.status == "discarded"


def test_batch_review_split_preserves_source_ranges_and_shifts_following_positions():
    draft = make_draft()
    first = make_candidate(draft.id, 1, "First", "abcdef", 0, 6)
    following = make_candidate(draft.id, 2, "Following", "later", 6, 11)
    draft.candidates = [first, following]
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())

    asyncio.run(service.review_candidate(make_reviewer(), draft.id, "split", first.id, split_at=3))

    by_position = {item.position: item for item in draft.candidates}
    assert by_position[1].body_md == "abc"
    assert by_position[1].source_start == 0
    assert by_position[1].source_end == 3
    assert by_position[2].body_md == "def"
    assert by_position[2].source_start == 3
    assert by_position[2].source_end == 6
    assert by_position[3] is following


def test_batch_review_rejects_user_without_review_permission():
    draft = make_draft(make_candidate(uuid.uuid4(), 1, "First", "Body", 0, 4))
    draft.candidates[0].draft_id = draft.id
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())
    unauthorized = User(
        id=uuid.uuid4(),
        email="staff@acme.test",
        name="Staff",
        company_domain="acme.test",
        role="Staff",
        active=True,
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.review_candidate(unauthorized, draft.id, "rename", draft.candidates[0].id, title="Nope"))

    assert exc.value.status_code == 403


def test_batch_commit_creates_pending_children_with_source_positions():
    draft = make_draft()
    first = make_candidate(draft.id, 1, "First", "A", 0, 1)
    second = make_candidate(draft.id, 2, "Second", "B", 1, 2)
    draft.candidates = [first, second]
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())

    children = asyncio.run(service.commit_candidates(make_reviewer(), draft.id))

    assert draft.status == "rejected"
    assert [item.status for item in draft.candidates] == ["committed", "committed"]
    assert len(children) == 2
    assert [item.status for item in children] == ["pending", "pending"]
    assert [item.content_metadata["source_position"] for item in children] == [
        {"start": 0, "end": 1, "heading": "First"},
        {"start": 1, "end": 2, "heading": "Second"},
    ]
    assert repo.audits[-1].action == "batch_commit"


def test_committed_children_are_marked_as_split_products_so_retry_cannot_re_split():
    """The infinite batch-review loop, pinned at its source.

    `commit_candidates` stamps every child `restructure_status="lossless_ready"`, and the
    reviewer lands straight back on those children. Retrying AI format on one used to
    re-run `route_document_candidates_llm`, which splits by DEPARTMENT rather than by
    size, so a child came back with >1 candidates, demanded another batch review, and
    produced a further generation of children — fanning out on every pass instead of
    converging.

    `run_restructure_pending_draft` now skips candidate regeneration when this key is
    present, so the contract worth pinning is that commit actually sets it. Losing it
    would silently restore the loop.
    """
    draft = make_draft()
    first = make_candidate(draft.id, 1, "First", "A", 0, 1)
    second = make_candidate(draft.id, 2, "Second", "B", 1, 2)
    draft.candidates = [first, second]
    repo = BatchRepository(draft)
    service = GovernanceService(repo, object())

    children = asyncio.run(service.commit_candidates(make_reviewer(), draft.id))

    assert [item.content_metadata["submission_kind"] for item in children] == [
        "split_candidate",
        "split_candidate",
    ]
    # The status that makes the retry control eligible in the first place; if this ever
    # stops being a settled state the frontend gate has to change with it.
    assert [item.restructure_status for item in children] == [
        "lossless_ready",
        "lossless_ready",
    ]
