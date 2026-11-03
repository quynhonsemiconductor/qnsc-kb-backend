"""A User handed to authorization code must arrive with its relationships loaded.

Connector syncs aborted with

    Aborted after 25 consecutive item failures; last error: greenlet_spawn has not been
    called; can't call await_only() here. Was IO attempted in an unexpected place?

`_persist_connector_draft` built its actor with `db.get(User, draft.created_by)`, which
loads columns and no relationships. It then called `submit_draft`, which reaches
`GovernanceRepository.get_draft_for_user`, which calls the SYNCHRONOUS
`AuthorizationService.has_permission` and reads `user.roles` directly.

Reading an unloaded relationship is a lazy load. On an async session that means IO from
plain attribute access -- no await for the greenlet to suspend on -- so SQLAlchemy
raises. It failed for every document that produced a draft, so the walk hit its 25
consecutive item failures and gave up on the whole scope.

Nothing caught it because every HTTP path loads users through `UserRepository.get_by_id`,
which eager-loads exactly what authorization reads. Only the connector built its own
actor, and connector authorization had never succeeded, so this code had never run.

The first test is the general contract, and is the one worth keeping: whatever
authorization reads off a user, the repository that loads users must eager-load. It
fails if either side drifts.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).parents[2]
SRC = REPO / "src"

#: Modules whose functions receive a User and read authorization state off it.
AUTHORIZATION_READERS = (
    SRC / "domain" / "rbac.py",
    SRC / "domain" / "permissions.py",
    SRC / "repositories" / "governance.py",
)
#: Names those modules use for the user being authorized.
USER_PARAMS = {"user", "viewer", "actor", "approver"}


def _user_relationships() -> set[str]:
    tree = ast.parse((SRC / "models" / "user.py").read_text(encoding="utf-8"))
    names = set()
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "User"]:
        for s in cls.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.value, ast.Call):
                f = s.value.func
                if getattr(f, "id", None) == "relationship" or getattr(f, "attr", None) == "relationship":
                    names.add(s.target.id)
    assert names, "found no relationships on User; the parser has drifted"
    return names


def _relationships_authorization_reads() -> set[str]:
    """Relationships read off a user parameter, including via getattr(user, "x")."""
    rels = _user_relationships()
    found = set()
    for path in AUTHORIZATION_READERS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and n.attr in rels
                    and isinstance(n.value, ast.Name) and n.value.id in USER_PARAMS):
                found.add(n.attr)
            if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "getattr"
                    and len(n.args) >= 2
                    and isinstance(n.args[0], ast.Name) and n.args[0].id in USER_PARAMS
                    and isinstance(n.args[1], ast.Constant) and n.args[1].value in rels):
                found.add(n.args[1].value)
    return found


def _eager_loaded_by_get_by_id() -> set[str]:
    """`User.<attr>` named inside the .options(...) chain of UserRepository.get_by_id."""
    tree = ast.parse((SRC / "repositories" / "user.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_by_id"
    )
    loaded = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "User"):
            loaded.add(n.attr)
    return loaded


def test_authorization_actually_reads_relationships_off_the_user():
    """Guards the sweep: if this were empty the contract test would pass vacuously."""
    assert "roles" in _relationships_authorization_reads()


@pytest.mark.parametrize("attr", sorted(_relationships_authorization_reads()))
def test_the_user_repository_eager_loads_what_authorization_reads(attr):
    """Reading any of these off a lazily-loaded User is a greenlet error on an async
    session -- attribute access cannot suspend, so SQLAlchemy cannot do the IO."""
    assert attr in _eager_loaded_by_get_by_id(), (
        f"AuthorizationService reads user.{attr}, but UserRepository.get_by_id does not "
        f"eager-load it"
    )


def test_the_connector_does_not_build_its_own_actor():
    """The regression itself. `db.get(User, ...)` returns a User with no relationships;
    handing that to submit_draft is what aborted every sync."""
    tree = ast.parse((SRC / "domain" / "cloud_sync.py").read_text(encoding="utf-8"))
    bare = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get"
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == "User"
    ]
    assert not bare, f"cloud_sync.py:{bare} loads a User without its relationships"
