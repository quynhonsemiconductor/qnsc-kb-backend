"""Bulk draft decisions: partial success, and the expiry bug that made it fail three times.

WHY THIS ENDPOINT EXISTS. Every other draft endpoint is `/{id}`-scoped, so a 122-deep queue
cost 122 separate reviews at ~4 interactions each — roughly 500 clicks, while the connector
refilled the queue every 600 seconds. Measured against the live stack: 270 synced documents
produced 122 drafts (45%) and 7 articles (6% of the queue).

THE BUG WORTH REMEMBERING. `GovernanceService.approve_draft` calls `db.rollback()` when it
refuses a draft (governance.py:1424). A rollback EXPIRES the entire identity map —
`expire_on_commit=False` governs commits, not rollbacks — so after the first refusal:

  * the next draft's permission check reads `user.roles` and raises MissingGreenlet
    (`rbac.py:224 in has_permission`), and
  * reading `actor.id` to RELOAD the user raises the same error.

That second point defeated two fixes. A helper taking the `User` instance cannot work,
because it must touch the expired object to recover from the expiry. Measured three times:
2 of 5 drafts approved, then 24 MissingGreenlets. The fix is to capture the actor id as a
plain UUID BEFORE the loop.

These tests pin that behaviour, because the failure only appears on the draft AFTER a
refusal — a batch of all-good or all-bad drafts passes either way.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import uuid

import pytest

from src.api.routers import governance as gov


def _source() -> str:
    return inspect.getsource(gov.bulk_decide_drafts)


# ------------------------------------------------------- request contract and bounds


def test_the_batch_is_bounded():
    """An unbounded batch is a timeout and a 122-row transaction waiting to happen."""
    field = gov.BulkDecideRequest.model_fields["draft_ids"]
    constraints = str(field.metadata)

    assert "100" in constraints, constraints
    assert "1" in constraints, "an empty batch is a client bug, not a no-op"


def test_only_approve_and_reject_are_accepted():
    with pytest.raises(Exception):
        gov.BulkDecideRequest(draft_ids=[uuid.uuid4()], decision="delete")

    for decision in ("approve", "reject"):
        assert gov.BulkDecideRequest(draft_ids=[uuid.uuid4()], decision=decision)


def test_the_review_note_is_bounded_like_the_single_draft_endpoint():
    """Same ceiling as RejectRequest, or bulk becomes a way past a validated limit."""
    bulk = gov.BulkDecideRequest.model_fields["review_note"]
    single = gov.RejectRequest.model_fields["review_note"]

    assert "2000" in str(bulk.metadata)
    assert "2000" in str(single.metadata)


# ------------------------------------------------- the expiry bug, pinned three ways


def test_the_actor_id_is_captured_outside_the_loop_not_inside_it():
    """The recovery path must not depend on the object it recovers from.

    Reading `.id` off an expired instance raises MissingGreenlet, so capturing it inside the
    loop is exactly as broken as not capturing it at all — the second iteration reads it
    from an already-expired `current_user`.

    Asserted with `ast`, not source positions. A positional check ("capture appears before
    the first decision") passes when the assignment is moved INSIDE the loop, because it
    still appears earlier in the text. Mutation-checked: that weaker version passed against
    the reintroduced bug.
    """
    tree = ast.parse(textwrap.dedent(_source()))
    func = tree.body[0]

    def assigns_actor_id(node) -> bool:
        return isinstance(node, ast.AnnAssign | ast.Assign) and any(
            getattr(target, "id", None) == "actor_id"
            for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        )

    at_body_level = any(assigns_actor_id(node) for node in func.body)
    inside_a_loop = any(
        assigns_actor_id(inner)
        for node in ast.walk(func)
        if isinstance(node, ast.For)
        for inner in ast.walk(node)
    )

    assert at_body_level, "actor_id must be captured in the function body"
    assert not inside_a_loop, (
        "capturing actor_id inside the loop re-reads it from an expired current_user"
    )


def test_the_reload_helper_takes_a_primitive_id_not_an_orm_instance():
    """A helper taking `User` cannot recover: it must touch the expired object to do so."""
    signature = inspect.signature(gov._reload_actor)
    params = list(signature.parameters.values())

    assert params[1].name == "actor_id"
    assert params[1].annotation is uuid.UUID, params[1].annotation


def test_the_actor_is_reloaded_after_every_failure_path():
    """Both handlers, not just one: a 409 expires the map exactly as a 500 does."""
    source = _source()
    handlers = source.count("_reload_actor(db, actor_id)")

    assert handlers == 2, f"expected a reload in both except branches, found {handlers}"


def test_the_audit_row_uses_the_primitive_id_too():
    """It runs after the loop, where current_user may still be expired."""
    source = _source()
    audit = source[source.index("AuditLog(") :]

    assert "user_id=actor_id" in audit
    assert "user_id=current_user.id" not in audit


def test_a_vanished_actor_stops_the_batch_rather_than_looping():
    """Deactivated mid-batch: continuing would attempt decisions with an unusable actor."""
    source = _source()

    assert source.count("if reloaded is None:") == 2
    assert "break" in source


# --------------------------------------------------------- partial-success behaviour


def test_no_savepoint_wraps_the_self_committing_service():
    """`approve_draft` commits internally; a savepoint around it closes underneath it.

    Comments are stripped before asserting: the function DOCUMENTS why `begin_nested()` is
    absent, so a plain substring search matches the explanation and fails against correct
    code. Measured — that is exactly how this test failed first.
    """
    code = "\n".join(
        line for line in _source().splitlines() if not line.strip().startswith("#")
    )

    assert "begin_nested" not in code, (
        "approve_draft commits at governance.py:1421 — wrapping it in a savepoint made the "
        "post-commit reload fail with 'Can't operate on closed transaction'"
    )


def test_ids_are_de_duplicated_while_preserving_order():
    """A repeated id must not be decided twice, and the report should read as selected."""
    source = _source()

    assert "dict.fromkeys" in source, "set() would scramble the reviewer's order"


def test_a_reject_without_a_note_fails_once_for_the_batch():
    """Not once per id: 100 copies of the same validation error is not a usable response."""
    source = _source()
    guard = source[: source.index("for draft_id")]

    assert "review_note_required" in guard


def test_blocked_drafts_keep_their_structured_code():
    """The UI branches on batch_review_required / update_confirmation_required / ACL."""
    source = _source()

    assert 'detail.get("code")' in source
    assert "isinstance(detail, dict)" in source, "a string detail must not crash the report"


def test_both_outcomes_are_reported_separately():
    """A count alone cannot tell a reviewer WHICH drafts still need them."""
    source = _source()

    for key in ("decided_count", "blocked_count", '"decided"', '"blocked"'):
        assert key in source, key


def _walk_routes(routes, prefix: str = ""):
    """Yield ``(full_path, methods)`` for every endpoint, however it was mounted.

    Inlined rather than imported from tests/integration: there are no `__init__.py` files
    under tests/, so `tests.integration...` is not an importable package and the import
    fails with ModuleNotFoundError. Measured.

    The walk itself is necessary because FastAPI 0.141 / Starlette 1.6 stopped flattening
    `include_router()` into `app.routes`. Each include is a wrapper carrying no `path`,
    holding sub-routes on `original_router.routes` with paths relative to the mount and the
    prefix on `include_context.prefix`. Reading `route.path` off the top level therefore
    returns only the few routes declared directly on the app — this test passed locally and
    failed in CI against the newer FastAPI, asserting against an inventory that did not
    contain the application at all.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            context = getattr(route, "include_context", None)
            yield from _walk_routes(
                included.routes, prefix + (getattr(context, "prefix", "") or "")
            )
            continue
        path = getattr(route, "path", None)
        if path:
            yield prefix + path, set(getattr(route, "methods", set()) or set())


def test_the_endpoint_is_registered_before_the_id_scoped_routes_can_shadow_it():
    """`bulk-decide` is a literal segment where `{id}` also matches."""
    from src.api.main import app

    paths = {path for path, _methods in _walk_routes(app.routes)}

    assert paths, "the route inventory must not be empty, or this checks nothing"
    assert "/api/v1/governance/pending-drafts/bulk-decide" in paths
    # No bare /pending-drafts/{id} route exists, so the literal cannot be captured by one.
    assert "/api/v1/governance/pending-drafts/{id}" not in paths


def test_the_audit_outcome_uses_the_established_vocabulary():
    """Only success/failure/applied exist; a fourth value breaks the audit-log filter."""
    source = _source()
    audit = source[source.index("AuditLog(") :]

    assert 'outcome="success"' in audit
    assert "partial" not in audit
