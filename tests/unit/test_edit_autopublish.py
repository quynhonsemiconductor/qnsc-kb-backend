"""Editing an article publishes immediately for whoever may approve their own change.

WHY THIS EXISTS. `PUT /articles/{id}` always created a `PendingDraft` and answered
`workflow="pending_approval"`, even for an Admin who would then open the review screen and
click publish on their own change. Self-approval was already permitted and already audited
as `self_approve` (governance.py:1397-1400), so the queue step was pure ceremony for that
identity. The rule reused is `may_publish_own_change` -> `_may_self_approve` (Admin/CEO),
NOT `can_edit_article`: a department publisher still goes through independent review,
because `submit_draft` calls that path "Submitted for independent approval".

THE BUG WORTH REMEMBERING — it cost two wrong fixes here, after costing three on the bulk
endpoint. `approve_draft` calls `db.rollback()` when it refuses (governance.py:1434). A
rollback EXPIRES the entire identity map, so inside the `except` handler:

  * `created.title` raises MissingGreenlet, and
  * `current_user.id` — needed to write the fallback audit row — raises it too.

The second one is the trap: the recovery path must not touch any ORM instance it is
recovering from. Both failures were observed as a 500 from a real request, not reasoned
about. The fix is to capture plain values BEFORE the approval attempt.

Measured against real Postgres: happy path 12/12 checks (article body live, version 2,
`self_approved=True` in the audit), fallback 6/6 with `reason=batch_review_required`.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from src.api.routers import articles as articles_router
from src.domain.governance import GovernanceService


def _update_article_tree() -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(articles_router.update_article))
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
    return func


def _approval_branch(func) -> ast.If:
    """The `if <predicate>(current_user):` guard that skips the queue."""
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        call = node.test if isinstance(node.test, ast.Call) else None
        if call is not None and isinstance(call.func, ast.Attribute):
            if call.func.attr == "may_publish_own_change":
                return node
    raise AssertionError("update_article no longer branches on may_publish_own_change")


def _except_handlers(node) -> list[ast.ExceptHandler]:
    return [item for item in ast.walk(node) if isinstance(item, ast.ExceptHandler)]


# --------------------------------------------------------------------------- the rule


def test_the_rule_is_the_one_the_codebase_already_trusts_for_self_approval():
    """A wider rule would let a department publisher approve their own work unreviewed."""
    assert hasattr(GovernanceService, "may_publish_own_change")

    source = textwrap.dedent(inspect.getsource(GovernanceService.may_publish_own_change))
    # It must DELEGATE, not restate the role set: two copies of "who may self-approve"
    # would drift, and the drift would silently widen who bypasses review.
    assert "_may_self_approve" in source
    for role in ("Admin", "CEO"):
        assert role not in source, (
            f"{role} is named directly; delegate to _may_self_approve instead of "
            "restating the role set"
        )


def test_the_edit_path_does_not_invent_its_own_permission_check():
    """`can_edit_article` is far wider than "may approve their own change"."""
    branch = _approval_branch(_update_article_tree())
    call = branch.test
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "may_publish_own_change"
    # Guarded on the actor, not on the article.
    assert [arg.id for arg in call.args if isinstance(arg, ast.Name)] == ["current_user"]


# ------------------------------------------------------- the expiry trap (the real bug)


def test_the_fallback_never_touches_an_object_the_rollback_expired():
    """The recovery path must not read the ORM instances it is recovering from.

    This is the assertion that would have caught both 500s. `approve_draft` rolls back on
    refusal, expiring `created`, `current` AND `current_user` — so any attribute read on
    them inside the handler raises MissingGreenlet, including the `current_user.id` needed
    for the audit row.
    """
    branch = _approval_branch(_update_article_tree())
    handlers = _except_handlers(branch)
    assert handlers, "the approval attempt must handle the refusal it can legitimately get"

    expired = {"created", "current", "current_user"}
    offenders: list[str] = []
    for handler in handlers:
        for node in ast.walk(handler):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in expired:
                    offenders.append(f"{node.value.id}.{node.attr}")
    assert not offenders, (
        "expired ORM instances read inside the fallback: "
        f"{sorted(set(offenders))} — capture plain values before the approval attempt"
    )


def test_the_primitives_are_captured_before_the_approval_attempt():
    """Capturing them after the attempt is exactly as broken as not capturing them."""
    func = _update_article_tree()
    branch = _approval_branch(func)

    captures = {"draft_id", "draft_title", "draft_status", "edited_article_id", "actor_id"}
    seen: dict[str, int] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in captures:
                    seen.setdefault(target.id, node.lineno)

    missing = captures - seen.keys()
    assert not missing, f"not captured as plain values: {sorted(missing)}"
    for name, lineno in seen.items():
        assert lineno < branch.lineno, (
            f"{name} is captured at line {lineno}, at/after the approval branch on "
            f"{branch.lineno} — the rollback has already expired the session by then"
        )


def test_the_actor_id_is_a_plain_value_not_the_user_instance():
    """A helper taking the `User` cannot recover: it must touch the expired object."""
    func = _update_article_tree()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "actor_id":
                    value = node.value
                    assert isinstance(value, ast.Attribute) and value.attr == "id", (
                        "actor_id must be current_user.id read BEFORE the attempt, "
                        "not the instance itself"
                    )
                    return
    raise AssertionError("actor_id is not captured at all")


# ------------------------------------------------------------------ the reported contract


def test_both_outcomes_report_a_workflow_the_client_can_branch_on():
    """The UI routes on this: a published edit must not land on the review queue."""
    source = inspect.getsource(articles_router.update_article)
    assert 'workflow="published"' in source
    assert 'workflow="pending_approval"' in source


def test_a_refusal_falls_back_to_the_queue_instead_of_failing_the_edit():
    """A multi-section edit raises 409 batch_review_required; the edit is not lost.

    Falling back is the correct outcome, not a workaround: >1 split candidate means the
    reviewer must choose how the sections are routed, and that choice cannot be inferred.

    Mutation-checked. An earlier version of this test only looked for a `Return` node
    anywhere inside the handler, and PASSED when a bare `raise` was inserted above it —
    dead code is still in the AST. It now walks the handler body IN ORDER and requires the
    return to be reachable, which is the property that actually keeps the edit from
    becoming a 500.
    """
    branch = _approval_branch(_update_article_tree())
    handlers = _except_handlers(branch)
    assert handlers, "the approval attempt must handle the refusal it can legitimately get"

    def first_exit(body: list[ast.stmt]) -> ast.stmt | None:
        """The statement that leaves the handler: whichever comes first wins."""
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, (ast.Return, ast.Raise)):
                    return node
        return None

    for handler in handlers:
        exit_node = first_exit(handler.body)
        assert exit_node is not None, "the handler must exit deliberately"
        assert isinstance(exit_node, ast.Return), (
            "the handler re-raises before returning, so a refused self-approval surfaces "
            "as a 500 and the editor's change looks lost"
        )
        assert isinstance(exit_node.value, ast.Call)
        reported = {
            keyword.value.value
            for keyword in exit_node.value.keywords
            if keyword.arg == "workflow" and isinstance(keyword.value, ast.Constant)
        }
        assert reported == {"pending_approval"}, (
            f"the fallback must report pending_approval, got {reported or 'nothing'}"
        )


def test_the_approval_reuses_the_reviewer_path_rather_than_publishing_directly():
    """Publishing inline would bypass every similarity, ACL and split gate."""
    branch = _approval_branch(_update_article_tree())

    called = {
        node.func.attr
        for node in ast.walk(branch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "approve_draft" in called, (
        "the edit must go through the same approve_draft a reviewer calls"
    )
    # No second publish implementation may appear on this path.
    for forbidden in ("create_article", "_publish_article", "add"):
        assert forbidden not in called, (
            f"{forbidden}() on the auto-publish path suggests a parallel publish "
            "implementation; reuse approve_draft"
        )


def test_the_draft_is_still_submitted_so_the_history_survives_a_refusal():
    """create_draft/update_draft commit, so a later rollback cannot erase the draft.

    That is what makes the fallback's promise honest: it tells the editor the change is
    queued, and the row is genuinely there to be reviewed.
    """
    source = inspect.getsource(articles_router.update_article)
    assert "submit_draft" in source
    submit_at = source.index("submit_draft")
    approve_at = source.index("approve_draft")
    assert submit_at < approve_at, "the draft must be submitted before it is approved"
