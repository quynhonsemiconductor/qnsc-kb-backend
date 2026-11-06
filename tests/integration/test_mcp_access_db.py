"""Real-DB proof for the Access email lookup.

An IdP may return a different local-part case for the same mailbox, and the KB stores
whatever the account was created with. `UserRepository.get_by_email` tries the exact
address first, then `lower(email)`; two rows that collide case-insensitively are an
explicit refusal, never SQLAlchemy's `MultipleResultsFound`.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.deps import set_database_context
from src.domain.rbac import bootstrap_rbac
from src.models.user import User
from src.repositories.user import AmbiguousEmailError, UserRepository

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="needs the migrated PostgreSQL test database",
)


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine(os.environ["DATABASE_URL"])
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield maker
    finally:
        await engine.dispose()


async def _add(db, *, email: str, user_id=None) -> uuid.UUID:
    # Core INSERT, deliberately: the ORM validator lowercases `email`, and these tests
    # exist to prove the lookup against legacy rows that were stored with mixed case.
    user_id = user_id or uuid.uuid4()
    await db.execute(
        insert(User).values(
            id=user_id,
            email=email,
            name="MCP Test",
            password_hash="x",
            company_domain="example.test",
            role="Staff",
            active=True,
            auth_version=0,
        )
    )
    return user_id


async def _cleanup(db, *user_ids: uuid.UUID) -> None:
    for user_id in user_ids:
        row = await db.get(User, user_id)
        if row is not None:
            await db.delete(row)
    await db.commit()


@pytest.mark.asyncio
async def test_a_mixed_case_stored_email_is_found_by_its_lowercase_form(factory):
    email = f"MiXeD.{uuid.uuid4().hex[:10]}@Example.TEST"
    async with factory() as db:
        # The insert and the cleanup are maintenance operations, not a tenant request.
        await set_database_context(db, None, True)
        user_id = await _add(db, email=email)
        await db.commit()
        try:
            found = await UserRepository(db).get_by_email(email.lower())
            assert found is not None and found.id == user_id
            assert found.email == email
            # A different case still finds the same single row.
            assert (await UserRepository(db).get_by_email(email.upper())).id == user_id
        finally:
            await _cleanup(db, user_id)


@pytest.mark.asyncio
async def test_the_exact_address_wins_and_a_lowercase_collision_is_an_explicit_error(
    factory,
):
    token = uuid.uuid4().hex[:10]
    lower = f"exact.{token}@example.test"
    mixed = f"Exact.{token}@Example.TEST"
    dup_a = f"Dup.{token}@Example.TEST"
    dup_b = f"dUp.{token}@Example.TEST"
    async with factory() as db:
        await set_database_context(db, None, True)
        lower_id = await _add(db, email=lower)
        mixed_id = await _add(db, email=mixed)
        dup_a_id = await _add(db, email=dup_a)
        dup_b_id = await _add(db, email=dup_b)
        await db.commit()
        try:
            found = await UserRepository(db).get_by_email(lower)
            assert found is not None and found.id == lower_id, "exact match must win"
            # The address exactly as given wins too, even though its lower-cased form
            # also matches the other row.
            found = await UserRepository(db).get_by_email(mixed)
            assert found is not None and found.id == mixed_id

            # Neither stored row is the all-lowercase form, so the lower() fallback sees
            # the collision and refuses explicitly.
            with pytest.raises(AmbiguousEmailError):
                await UserRepository(db).get_by_email(f"dup.{token}@example.test")
        finally:
            await _cleanup(db, lower_id, mixed_id, dup_a_id, dup_b_id)


@pytest.mark.asyncio
async def test_two_sessions_cannot_bind_the_same_account_to_different_subjects(factory):
    """The first-bind race: the loser must see the winner's row, not its own stale object.

    Both sessions load the unbound row into their identity maps before either writes.
    `expire_on_commit=False` plus an ORM-synchronised UPDATE used to leave the loser with
    an object that looked bound to its own `sub` even though its UPDATE matched zero rows.
    """
    from src.api.routers import mcp
    from src.domain.access_jwt import AccessIdentity

    token = uuid.uuid4().hex[:10]
    email = f"race.{token}@example.test"
    winner = f"race-winner-{token}"
    loser_sub = f"race-loser-{token}"
    async with factory() as setup:
        await set_database_context(setup, None, True)
        user_id = await _add(setup, email=email)
        await setup.commit()
        # Bootstrap now, so neither session runs it during the race; that bootstrap is a
        # separate concurrent-writer problem, not the binding under test. It backfills
        # the user with a Staff role, so `active_user_with_context` skips its own call.
        await bootstrap_rbac(setup)
        await setup.commit()

    async with factory() as a, factory() as b:
        writer = await UserRepository(a).get_by_email(email)
        loser = await UserRepository(b).get_by_email(email)
        assert writer is not None and loser is not None
        assert writer.access_subject is None and loser.access_subject is None

        await mcp.resolve_access_principal(
            a, AccessIdentity(sub=winner, email=email, oid=None)
        )
        with pytest.raises(mcp.AccessDeniedError):
            await mcp.resolve_access_principal(
                b, AccessIdentity(sub=loser_sub, email=email, oid=None)
            )

    async with factory() as check:
        await set_database_context(check, None, True)
        row = await check.get(User, user_id)
        assert row is not None and row.access_subject == winner
        await _cleanup(check, user_id)
