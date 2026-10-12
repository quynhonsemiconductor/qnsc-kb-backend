import asyncio
import uuid
from types import SimpleNamespace

import pytest

from src.repositories.ai import AIRepository, _is_retryable_failed_pair


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Scalars:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _RowsResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return _Scalars(self.values)


class _Session:
    def __init__(self, conversation, latest=None):
        self.results = [_ScalarResult(conversation)]
        if latest is not None:
            self.results.append(_RowsResult(latest))
        self.added = []
        self.deleted = []
        self.commits = 0
        self.refreshed = []

    async def execute(self, _statement):
        return self.results.pop(0)

    def add(self, value):
        self.added.append(value)

    async def delete(self, value):
        self.deleted.append(value)

    async def commit(self):
        self.commits += 1

    async def refresh(self, value):
        self.refreshed.append(value)


def _message(role, content, *, failed=False):
    return SimpleNamespace(
        id=uuid.uuid4(),
        role=role,
        content=content,
        action_data={"failed": True} if failed else None,
    )


def test_retry_predicate_requires_the_latest_matching_failed_pair():
    question = "Có bao nhiêu nội dung trong Luật PTDNNVV"
    user = _message("user", question)
    failure = _message("assistant", "AI generation failed", failed=True)

    assert _is_retryable_failed_pair(failure, user, f"  {question}  ")
    assert not _is_retryable_failed_pair(_message("assistant", "ok"), user, question)
    assert not _is_retryable_failed_pair(failure, user, "another question")


def test_retry_reopens_one_turn_without_duplicating_the_user_question():
    conversation = SimpleNamespace(id=uuid.uuid4(), updated_at=None)
    question = "Có bao nhiêu nội dung trong Luật PTDNNVV"
    user = _message("user", question)
    failure = _message("assistant", "AI generation failed", failed=True)
    session = _Session(conversation, [failure, user])

    result = asyncio.run(
        AIRepository(session).start_user_turn(
            conversation.id, uuid.uuid4(), question, retry_failed=True
        )
    )

    assert result is user
    assert session.deleted == [failure]
    assert session.added == []
    assert session.commits == 1


def test_stale_or_concurrent_retry_does_not_delete_or_append_anything():
    conversation = SimpleNamespace(id=uuid.uuid4(), updated_at=None)
    in_progress = _message("user", "same question")
    previous = _message("assistant", "older answer")
    session = _Session(conversation, [in_progress, previous])

    with pytest.raises(ValueError, match="latest failed turn"):
        asyncio.run(
            AIRepository(session).start_user_turn(
                conversation.id, uuid.uuid4(), "same question", retry_failed=True
            )
        )

    assert session.deleted == []
    assert session.added == []
    assert session.commits == 0


def test_ordinary_send_still_appends_one_user_message():
    conversation = SimpleNamespace(id=uuid.uuid4(), updated_at=None)
    session = _Session(conversation)

    message = asyncio.run(
        AIRepository(session).start_user_turn(
            conversation.id, uuid.uuid4(), "new question"
        )
    )

    assert message.role == "user"
    assert message.content == "new question"
    assert session.added == [message]
    assert session.deleted == []
    assert session.commits == 1
