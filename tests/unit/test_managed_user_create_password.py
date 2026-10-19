import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routers.auth import ManagedUserCreate, create_managed_user


def test_managed_create_without_a_password_is_a_422_before_any_database_work():
    # db is None on purpose: the refusal must come before the first query.
    user_in = ManagedUserCreate(email="new.person@qnsc.vn", name="New Person")
    current_user = SimpleNamespace(company_domain="qnsc.vn", roles=[])

    with pytest.raises(HTTPException) as refused:
        asyncio.run(create_managed_user(user_in, current_user=current_user, db=None))

    assert refused.value.status_code == 422
    assert refused.value.detail == "A password is required to create a user"
