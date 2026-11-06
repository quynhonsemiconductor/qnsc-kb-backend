"""Every account and invitation create path stores one canonical email."""

from datetime import datetime, timedelta

from src.models.user import Invitation, User


def test_user_email_is_stripped_and_lowercased_on_create():
    user = User(
        email="  MiXeD.User@Example.TEST ",
        name="Mixed",
        password_hash="x",
        company_domain="example.test",
        role="Staff",
        active=True,
    )
    assert user.email == "mixed.user@example.test"


def test_user_email_is_normalised_on_assignment_too():
    user = User(
        email="first@example.test",
        name="Mixed",
        password_hash="x",
        company_domain="example.test",
        role="Staff",
        active=True,
    )
    user.email = " Second.User@Example.TEST "
    assert user.email == "second.user@example.test"


def test_invitation_email_is_normalised():
    invitation = Invitation(
        email=" Invitee@Example.TEST ",
        name="Invitee",
        role="Staff",
        company_domain="example.test",
        token_hash="h",
        expires_at=datetime.utcnow() + timedelta(days=1),
    )
    assert invitation.email == "invitee@example.test"


def test_no_bulk_update_bypasses_the_email_validator():
    """A core UPDATE does not run ORM validators, so none may write `email` in bulk."""
    from pathlib import Path

    offenders = [
        str(path)
        for path in Path("src").rglob("*.py")
        if ".values(email=" in path.read_text(encoding="utf-8")
        or '.values({"email"' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
