"""Real-DB proof that the Access email lookup is case-insensitive.

An IdP may return a different local-part case for the same mailbox, and the KB stores
whatever the account was created with. `UserRepository.get_by_email` normalises the input
and compares both the exact and lower-cased column, so a mixed-case stored row is found.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.deps import set_database_context
from src.models.user import User
from src.repositories.user import UserRepository

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


@pytest.mark.asyncio
async def test_a_mixed_case_stored_email_is_found_by_its_lowercase_form(factory):
    email = f"MiXeD.{uuid.uuid4().hex[:10]}@Example.TEST"
    user_id = uuid.uuid4()
    async with factory() as db:
        # The insert and the cleanup are maintenance operations, not a tenant request.
        await set_database_context(db, None, True)
        db.add(
            User(
                id=user_id,
                email=email,
                name="Mixed Case",
                password_hash="x",
                company_domain="example.test",
                role="Staff",
                active=True,
                auth_version=0,
            )
        )
        await db.commit()
        try:
            found = await UserRepository(db).get_by_email(email.lower())
            assert found is not None and found.id == user_id
            assert found.email == email
        finally:
            row = await db.get(User, user_id)
            if row is not None:
                await db.delete(row)
                await db.commit()
