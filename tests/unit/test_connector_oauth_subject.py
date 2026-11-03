"""We must only call attributes that the JWT library we actually install provides.

Clicking "Authorize source" opened the provider tab, came straight back to
/admin/connectors?oauth=error, and left the connector unauthorized -- with nothing shown
to say why.

`oauth_callback` read an account label out of the provider's id_token with
`jwt.get_unverified_claims`. That function belongs to **python-jose**. This project
installs **PyJWT** (pyproject.toml declares pyjwt; nothing else provides a `jwt` module),
which has no such attribute, so the call raised AttributeError. The handler beside it
caught `jwt.PyJWTError`, which does not cover AttributeError, so the exception escaped
the callback, hit the blanket `except Exception` in oauth_callback_entry, and redirected
to the error page.

Microsoft's requested scope includes `openid`, so an id_token is ALWAYS returned and that
branch always ran. SharePoint authorization failed on every attempt, looking exactly like
the provider had declined.

The check below is deliberately general rather than a test of that one line. The bug was
not bad logic, it was calling a function that does not exist -- something no amount of
mocking finds, and something an import cannot catch because the attribute is only
resolved when the line runs. Walking the AST for `jwt.<attr>` catches the entire class,
across every module, including the paths that are hard to exercise in tests.

`jwt.get_unverified_header` is a real PyJWT function and stays legal here; only
`get_unverified_claims` is jose-only. That distinction is exactly why this is checked
against the installed module rather than against a hand-written denylist.
"""
from __future__ import annotations

import ast
from pathlib import Path

import jwt
import pytest

SRC = Path(__file__).parents[2] / "src"


def _modules_using_jwt() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "import jwt" in p.read_text(encoding="utf-8")]


def _jwt_attributes(path: Path) -> set[str]:
    """Every `jwt.<attr>` the module actually references.

    Parsed rather than grepped so docstrings, comments and strings that merely name a
    function cannot register as calls -- this file's own docstring names the broken one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "jwt"
    }


def test_the_installed_library_is_pyjwt_not_jose():
    """The premise of every assertion below."""
    assert hasattr(jwt, "PyJWTError"), "not PyJWT; the checks below assume the wrong library"
    assert not hasattr(jwt, "get_unverified_claims"), "python-jose is installed"


def test_some_module_actually_uses_jwt():
    """Guards the sweep itself: a discovery bug would otherwise pass silently."""
    assert _modules_using_jwt()


@pytest.mark.parametrize("path", _modules_using_jwt(), ids=lambda p: p.name)
def test_every_jwt_call_exists_on_the_installed_library(path):
    """The regression: `jwt.get_unverified_claims` is python-jose, and we ship PyJWT."""
    missing = sorted(attr for attr in _jwt_attributes(path) if not hasattr(jwt, attr))
    assert not missing, f"{path.relative_to(SRC.parent)} calls jwt.{{{', '.join(missing)}}}"


def test_the_callback_no_longer_decodes_the_provider_id_token():
    """`oauth_subject` has no reader anywhere -- no response schema, no query, no
    frontend. Decoding a provider token without verifying it, on the path between a
    successful exchange and status='active', was risk and a failure mode spent on a
    field with no consumer."""
    source = (SRC / "api" / "routers" / "connectors.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "verify_signature" not in body
    assert "id_token" not in body
