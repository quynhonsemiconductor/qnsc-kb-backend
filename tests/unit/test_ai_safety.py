import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.domain.ai_service import AIService, _authorized_conversation_history
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import User


def make_ai_user() -> User:
    user = User(
        id=uuid.uuid4(),
        email="staff@acme.test",
        name="Staff",
        company_domain="acme.test",
        role="Staff",
        active=True,
    )
    role = Role(name="Staff", company_domain="acme.test", active=True)
    role.permissions.append(
        RolePermission(
            permission=Permission(key="ai.ask", name="Ask AI"), scope="company"
        )
    )
    user.roles.append(role)
    return user


class FakeAIRepository:
    def __init__(self):
        self.logged = []
        self.cached = []

    async def get_cached(self, *_args, **_kwargs):
        return None

    async def log_usage(self, log):
        self.logged.append(log)

    async def cache_answer(self, cache):
        self.cached.append(cache)

    async def get_usage_log(self, *_args, **_kwargs):
        return None


class ConversationAwareAIRepository(FakeAIRepository):
    def __init__(self, messages):
        super().__init__()
        self.messages = messages
        self.cache_lookups = 0

    async def list_messages(self, *_args, **_kwargs):
        return self.messages

    async def get_cached(self, *_args, **_kwargs):
        self.cache_lookups += 1
        return None


class FakeSearchService:
    def __init__(self, results):
        self.results = results
        self.authorized_ids = {str(item["chunk_id"]) for item in results}
        # `ask` reads this to tell "the knowledge base has nothing" apart from "vector
        # retrieval is broken", which the real service reports rather than swallowing.
        self.vector_search_degraded = False
        self.chunk_repo = SimpleNamespace(
            authorized_chunk_ids=self.authorized_chunk_ids
        )

    async def search(self, *_args, **_kwargs):
        return self.results

    async def authorized_chunk_ids(self, _user, _chunk_ids):
        return self.authorized_ids


def ask_with(results):
    service = AIService(FakeAIRepository(), FakeSearchService(results), object())
    return asyncio.run(
        service.ask(make_ai_user(), "What is the unsupported retention exception?")
    )


def _retrieved_result():
    chunk_id = str(uuid.uuid4())
    return {
        "score": 0.95,
        "chunk_id": chunk_id,
        "parent_chunk_id": chunk_id,
        "article_id": str(uuid.uuid4()),
        "title": "Retention policy",
        "chunk_text": "The retention period is thirty days.",
        "parent_text": "The retention period is thirty days.",
        "section_ref": "Retention",
        "source_url": "/api/v1/articles/source",
    }


def _ask_with_provider(monkeypatch, answer):
    monkeypatch.setattr(
        "src.domain.ai_service.resolve_provider",
        lambda: SimpleNamespace(name="test", model="test-model"),
    )

    async def fake_complete(*_args, **_kwargs):
        return answer, 1, "test-model", "test"

    monkeypatch.setattr("src.domain.ai_service.complete", fake_complete)
    service = AIService(
        FakeAIRepository(), FakeSearchService([_retrieved_result()]), object()
    )
    return asyncio.run(service.ask(make_ai_user(), "What is the retention period?"))


def test_ai_refuses_when_no_authorized_context_is_retrieved():
    result = ask_with([])

    assert result["citations"] == []
    assert "could not find any authorized documents" in result["answer_grounded"]


def test_ai_refuses_when_retrieval_is_below_confidence_threshold():
    result = ask_with([{"score": 0.1, "chunk_id": str(uuid.uuid4())}])

    assert result["citations"] == []
    assert "not find enough relevant" in result["answer_grounded"]


def test_ai_refuses_when_provider_cites_unretrieved_source(monkeypatch):
    result = _ask_with_provider(
        monkeypatch, "The retention period is thirty days. [C999]"
    )

    assert result["citations"] == []
    assert "C999" not in result["answer"]
    assert "could not produce a grounded answer" in result["answer_grounded"]


def test_ai_refuses_when_provider_omits_grounded_citation(monkeypatch):
    result = _ask_with_provider(monkeypatch, "The retention period is thirty days.")

    assert result["citations"] == []
    assert "[C1]" not in result["answer"]
    assert "could not produce a grounded answer" in result["answer_grounded"]


def test_ai_invalidates_cached_answer_with_unretrieved_marker(monkeypatch):
    retrieved = _retrieved_result()

    class CachedRepository(FakeAIRepository):
        async def get_cached(self, *_args, **_kwargs):
            return SimpleNamespace(
                answer="Cached claim [C999]",
                citations=json.dumps(
                    [{"source_id": "C1", "chunk_id": retrieved["chunk_id"]}]
                ),
            )

    monkeypatch.setattr(
        "src.domain.ai_service.resolve_provider",
        lambda: SimpleNamespace(name="test", model="test-model"),
    )

    async def fake_complete(*_args, **_kwargs):
        return "Fresh claim [C1]", 1, "test-model", "test"

    monkeypatch.setattr("src.domain.ai_service.complete", fake_complete)
    service = AIService(CachedRepository(), FakeSearchService([retrieved]), object())
    result = asyncio.run(service.ask(make_ai_user(), "What is the retention period?"))

    assert result["answer_grounded"] == "Fresh claim [C1]"
    assert "C999" not in result["answer"]
    assert {item["source_id"] for item in result["citations"]} == {"C1"}


def test_ai_feedback_rejects_log_outside_actor_scope():
    service = AIService(FakeAIRepository(), FakeSearchService([]), object())

    with pytest.raises(HTTPException) as error:
        asyncio.run(service.submit_feedback(make_ai_user(), uuid.uuid4(), 1))

    assert error.value.status_code == 403


def test_first_turn_with_persisted_current_message_can_use_cache():
    question = "What is the unsupported retention exception?"
    repo = ConversationAwareAIRepository(
        [SimpleNamespace(role="user", content=question)]
    )
    service = AIService(repo, FakeSearchService([]), object())

    asyncio.run(service.ask(make_ai_user(), question, conversation_id=uuid.uuid4()))

    assert repo.cache_lookups == 1


def test_follow_up_with_prior_history_bypasses_cache():
    question = "What is the unsupported retention exception?"
    repo = ConversationAwareAIRepository(
        [
            SimpleNamespace(role="user", content="What is the retention policy?"),
            SimpleNamespace(role="assistant", content="The policy is documented."),
            SimpleNamespace(role="user", content=question),
        ]
    )
    service = AIService(repo, FakeSearchService([]), object())

    asyncio.run(service.ask(make_ai_user(), question, conversation_id=uuid.uuid4()))

    assert repo.cache_lookups == 0


def test_follow_up_does_not_reuse_assistant_text_after_citation_access_is_revoked(
    monkeypatch,
):
    revoked_chunk_id = uuid.uuid4()
    current_result = _retrieved_result()
    question = "What about that?"
    repo = ConversationAwareAIRepository(
        [
            SimpleNamespace(role="user", content="What is the restricted policy?"),
            SimpleNamespace(
                role="assistant",
                content="The restricted procedure is BLACK-BOX-SECRET [C1].",
                citations=json.dumps([{"chunk_id": str(revoked_chunk_id)}]),
            ),
            SimpleNamespace(role="user", content=question),
        ]
    )
    prompts = []

    monkeypatch.setattr(
        "src.domain.ai_service.resolve_provider",
        lambda: SimpleNamespace(name="test", model="test-model"),
    )

    async def fake_complete(messages, **_kwargs):
        prompts.append(messages[1]["content"])
        return (
            "<<<GROUNDED>>>\nThe authorized answer is available [C1].",
            1,
            "test-model",
            "test",
        )

    monkeypatch.setattr("src.domain.ai_service.complete", fake_complete)
    service = AIService(repo, FakeSearchService([current_result]), object())

    asyncio.run(service.ask(make_ai_user(), question, conversation_id=uuid.uuid4()))

    assert prompts
    assert "BLACK-BOX-SECRET" not in prompts[0]
    assert "What is the restricted policy?" in prompts[0]


def test_conversation_history_keeps_assistant_turn_with_currently_authorized_citation():
    chunk_id = uuid.uuid4()
    message = SimpleNamespace(
        role="assistant",
        content="The authorized procedure is available [C1].",
        citations=json.dumps([{"source_id": "C1", "chunk_id": str(chunk_id)}]),
    )
    search_service = FakeSearchService([{"chunk_id": str(chunk_id)}])

    safe_messages = asyncio.run(
        _authorized_conversation_history(search_service, make_ai_user(), [message])
    )

    assert safe_messages == [message]


def test_follow_up_history_drops_citation_marker_mismatch():
    chunk_id = uuid.uuid4()
    message = SimpleNamespace(
        role="assistant",
        content="The answer cites [C1].",
        citations=json.dumps([{"source_id": "C2", "chunk_id": str(chunk_id)}]),
    )
    search_service = FakeSearchService([{"chunk_id": str(chunk_id)}])

    safe_messages = asyncio.run(
        _authorized_conversation_history(search_service, make_ai_user(), [message])
    )

    assert safe_messages == []


def test_revoked_conversation_answer_cannot_leak_through_grounded_field(monkeypatch):
    from src.api.routers import ai as ai_router

    class FakeRepository:
        def __init__(self, _db):
            pass

        async def get_conversation(self, *_args, **_kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def list_messages(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    id=uuid.uuid4(),
                    role="assistant",
                    content="REVOKED SECRET ANSWER",
                    grounded_content="REVOKED GROUNDED SECRET",
                    extended_content="REVOKED EXTENDED SECRET",
                    citations=json.dumps([{"chunk_id": str(uuid.uuid4())}]),
                    usage_log_id=None,
                    created_at=None,
                )
            ]

    async def no_authorized_citations(*_args, **_kwargs):
        return []

    monkeypatch.setattr(ai_router, "AIRepository", FakeRepository)
    monkeypatch.setattr(ai_router, "_hydrate_citations", no_authorized_citations)

    response = asyncio.run(
        ai_router.get_conversation_messages(
            uuid.uuid4(), make_ai_user(), object()
        )
    )

    assert response[0]["content"].startswith("This historical answer is no longer")
    assert response[0]["answer_grounded"] == ""
    assert response[0]["answer_extended"] == ""
    assert response[0]["has_extended"] is False


def test_malformed_historical_citation_cannot_leak_metadata_or_answer(monkeypatch):
    from src.api.routers import ai as ai_router

    class FakeRepository:
        def __init__(self, _db):
            pass

        async def get_conversation(self, *_args, **_kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def list_messages(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    id=uuid.uuid4(),
                    role="assistant",
                    content="FORGED PRIVATE ANSWER",
                    grounded_content="FORGED GROUNDED ANSWER",
                    extended_content="FORGED EXTENDED ANSWER",
                    citations=json.dumps([
                        "not-a-citation",
                        {
                            "title": "Private document",
                            "excerpt": "FORGED SOURCE EXCERPT",
                            "source_url": "/api/v1/articles/private/source",
                        },
                    ]),
                    usage_log_id=None,
                    created_at=None,
                )
            ]

    monkeypatch.setattr(ai_router, "AIRepository", FakeRepository)

    response = asyncio.run(
        ai_router.get_conversation_messages(
            uuid.uuid4(), make_ai_user(), object()
        )
    )

    assert response[0]["content"].startswith("This historical answer is no longer")
    assert response[0]["answer_grounded"] == ""
    assert response[0]["answer_extended"] == ""
    assert response[0]["citations"] == []


def test_uncited_historical_assistant_answer_fails_closed(monkeypatch):
    from src.api.routers import ai as ai_router

    class FakeRepository:
        def __init__(self, _db):
            pass

        async def get_conversation(self, *_args, **_kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def list_messages(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    id=uuid.uuid4(),
                    role="assistant",
                    content="UNCITED PRIVATE ANSWER",
                    grounded_content="UNCITED GROUNDED ANSWER",
                    extended_content="UNCITED EXTENDED ANSWER",
                    citations="[]",
                    usage_log_id=None,
                    created_at=None,
                )
            ]

    monkeypatch.setattr(ai_router, "AIRepository", FakeRepository)

    response = asyncio.run(
        ai_router.get_conversation_messages(
            uuid.uuid4(), make_ai_user(), object()
        )
    )

    assert response[0]["content"].startswith("This historical answer is no longer")
    assert response[0]["answer_grounded"] == ""
    assert response[0]["answer_extended"] == ""
    assert response[0]["citations"] == []


def test_historical_citation_marker_mismatch_fails_closed(monkeypatch):
    from src.api.routers import ai as ai_router

    class FakeRepository:
        def __init__(self, _db):
            pass

        async def get_conversation(self, *_args, **_kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def list_messages(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    id=uuid.uuid4(),
                    role="assistant",
                    content="ANSWER WITH [C1]",
                    grounded_content="ANSWER WITH [C1]",
                    extended_content="",
                    citations=json.dumps([{
                        "source_id": "C2",
                        "chunk_id": str(uuid.uuid4()),
                    }]),
                    usage_log_id=None,
                    created_at=None,
                )
            ]

    async def authorized_citation(*_args, **_kwargs):
        return [{"source_id": "C2", "chunk_id": "authorized"}]

    monkeypatch.setattr(ai_router, "AIRepository", FakeRepository)
    monkeypatch.setattr(ai_router, "_hydrate_citations", authorized_citation)

    response = asyncio.run(
        ai_router.get_conversation_messages(
            uuid.uuid4(), make_ai_user(), object()
        )
    )

    assert response[0]["content"].startswith("This historical answer is no longer")
    assert response[0]["answer_grounded"] == ""
    assert response[0]["citations"] == []
