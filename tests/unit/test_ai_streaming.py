import asyncio
import json
import uuid
from types import SimpleNamespace


def test_streaming_answer_does_not_use_request_user_after_response_starts(monkeypatch):
    """SSE generators run after request dependencies have been released."""
    from src.api.routers import ai

    class RequestUser:
        def __init__(self):
            self._closed = False
            self._id = uuid.uuid4()
            self._company_domain = "acme.test"
            self.roles = []
            self.role = "Staff"

        @property
        def id(self):
            if self._closed:
                raise AssertionError("The streaming generator accessed a detached request user")
            return self._id

        @property
        def company_domain(self):
            if self._closed:
                raise AssertionError("The streaming generator accessed a detached request user")
            return self._company_domain

    class FakeRepository:
        def __init__(self, _db):
            self.conversation = SimpleNamespace(id=uuid.uuid4(), title="Question")

        async def get_conversation(self, *_args):
            return None

        async def create_conversation(self, *_args):
            return self.conversation

        async def start_user_turn(self, *_args, **_kwargs):
            return None

        async def add_message(self, *_args, **_kwargs):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeUserRepository:
        def __init__(self, _db):
            pass

        async def get_by_id(self, _user_id):
            return SimpleNamespace(id=uuid.uuid4(), company_domain="acme.test", roles=[], role="Staff")

    class FakeAIService:
        async def ask(self, _user, _question, **kwargs):
            await kwargs["on_token"]("Answer")
            return {"answer": "Answer", "citations": [], "log_id": None}

    async def allow(_key):
        return True, 0

    async def set_context(*_args):
        return None

    monkeypatch.setattr(ai, "AIRepository", FakeRepository)
    monkeypatch.setattr(ai, "SessionLocal", FakeSession)
    monkeypatch.setattr(ai, "UserRepository", FakeUserRepository)
    monkeypatch.setattr(ai, "get_ai_service", lambda _db: FakeAIService())
    monkeypatch.setattr(ai, "set_database_context", set_context)
    monkeypatch.setattr(ai.ai_rate_limiter, "allow", allow)

    async def run_stream():
        request_user = RequestUser()
        response = await ai.ask_question_stream(
            ai.AskRequest(question="What is the current policy?"),
            current_user=request_user,
            db=object(),
        )
        request_user._closed = True
        return [chunk.decode() if isinstance(chunk, bytes) else chunk async for chunk in response.body_iterator]

    events = asyncio.run(run_stream())
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events]
    assert {payload["type"] for payload in payloads} == {"token", "sources", "done"}
