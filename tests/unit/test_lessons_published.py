"""Team lessons are published at once, without a reviewer, kept apart from the reviewed documents
(src/domain/lessons.py), and search and the assistant can be limited to either."""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import src.domain.llm_client as llm_client
from src.api.routers import ai as ai_router
from src.api.routers import knowledge
from src.core.config import settings
from src.domain import lessons
from src.domain.ai_service import AIService
from src.domain.search_service import SearchService
from src.models.article import LESSON_TYPE
from tests.unit.test_search_bounds_the_cross_encoder import (
    QUERY,
    _admin,
    _article,
    _chunk,
    cross_encoder_on,  # noqa: F401 - fixture
)


def _run(coro):
    return asyncio.run(coro)


USER = SimpleNamespace(
    id=uuid.uuid4(), company_domain="acme.test", dept="Engineering", roles=[]
)


# -- the decision: a lesson is merged in place, a document is not ---------------------------------


class _Article:
    def __init__(self, title, body, type_=None):
        self.title, self.body_md, self.type = title, body, type_


class _Db:
    def __init__(self, lesson_ids=()):
        self.lesson_ids = set(lesson_ids)

    async def get(self, _model, article_id):
        kind = LESSON_TYPE if article_id in self.lesson_ids else "SOP"
        return _Article(f"Title {article_id}", f"Full text of {article_id}", kind)


def _decide_with(monkeypatch, reply, lesson_ids=(), found=(("a1", 0.5),)):
    async def similar(_db, _user, _content):
        return [{"article_id": a, "title": f"Title {a}", "score": s} for a, s in found]

    async def no_hits(self, user, query, filters=None, limit=5):
        return []

    seen = []
    monkeypatch.setattr(lessons, "find_similar_documents", similar)
    monkeypatch.setattr(SearchService, "search", no_hits)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: "p")

    async def complete(messages, **_k):
        seen.append(messages[1]["content"])
        return reply, 1, "m", "p"

    monkeypatch.setattr(llm_client, "complete", complete)
    decision = _run(
        lessons.decide_lesson(
            _Db(lesson_ids), USER, "Console baud", "## Lesson\n\n921600"
        )
    )
    return decision, seen


def test_an_update_of_a_lesson_carries_the_merged_text(monkeypatch):
    reply = '{"decision": "update", "target": 1, "reason": "newer", "merged": "## Lesson\\n\\nNow 921600."}'

    decision, seen = _decide_with(monkeypatch, reply, lesson_ids={"a1"})

    assert decision.action == "update" and decision.is_lesson is True
    assert decision.merged == "## Lesson\n\nNow 921600."
    assert "EXISTING LESSON 1" in seen[0] and "Full text of a1" in seen[0]


def test_an_update_of_a_reviewed_document_is_never_merged(monkeypatch):
    reply = (
        '{"decision": "update", "target": 1, "reason": "x", "merged": "rewritten spec"}'
    )

    decision, seen = _decide_with(monkeypatch, reply)

    assert decision.action == "update" and decision.is_lesson is False
    assert decision.merged is None
    assert "EXISTING DOCUMENT 1" in seen[0]


def test_an_update_without_merged_text_leaves_the_merge_to_the_caller(monkeypatch):
    decision, _ = _decide_with(
        monkeypatch,
        '{"decision": "update", "target": 1, "reason": "x"}',
        lesson_ids={"a1"},
    )

    assert decision.is_lesson is True and decision.merged is None


def test_without_a_model_a_close_copy_of_a_lesson_is_an_update_of_that_lesson(
    monkeypatch,
):
    async def similar(_db, _user, _content):
        return [{"article_id": "l1", "title": "T", "score": 0.9}]

    async def no_hits(self, user, query, filters=None, limit=5):
        return []

    monkeypatch.setattr(lessons, "find_similar_documents", similar)
    monkeypatch.setattr(SearchService, "search", no_hits)
    monkeypatch.setattr(llm_client, "resolve_provider", lambda: None)

    decision = _run(lessons.decide_lesson(_Db({"l1"}), USER, "T", "B"))

    assert decision.action == "update" and decision.is_lesson is True


def test_parse_merged():
    assert lessons.parse_merged('{"merged": " text "}') == "text"
    assert lessons.parse_merged('{"merged": null}') is None
    assert lessons.parse_merged('{"merged": "   "}') is None
    assert lessons.parse_merged("not json") is None
    assert lessons.parse_merged('```json\n{"merged": "x"}\n```') == "x"


# -- saving ----------------------------------------------------------------------------------------


class _CountDb:
    def __init__(self, saved_today):
        self.saved_today = saved_today

    async def scalar(self, _statement):
        return self.saved_today


def _save(monkeypatch, db, **extra):
    from src.domain import departments
    from src.domain.articles import ArticleService

    captured = {}

    async def resolve(_db, _domain, name, *, required=True):
        if not name:
            raise HTTPException(status_code=422, detail="A department is required")
        return SimpleNamespace(name=name)

    async def create(self, user, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id=uuid.uuid4(),
            title=kwargs["title"],
            status="published",
            dept=kwargs["dept"],
            version=1,
        )

    monkeypatch.setattr(departments, "resolve_active_department", resolve)
    monkeypatch.setattr(ArticleService, "create_article", create)
    args = dict(
        title="Console", body="## Lesson\n\nx", dept=None, language="vi", tags=["chip"]
    )
    args.update(extra)
    result = _run(lessons.save_new_lesson(db, USER, **args))
    return result, captured


def test_a_new_lesson_is_published_by_the_article_service_without_a_reviewer(
    monkeypatch,
):
    result, captured = _save(monkeypatch, _CountDb(0))

    assert captured["publish_without_review"] is True
    assert captured["type_"] == LESSON_TYPE and captured["sensitivity"] == "public"
    assert (
        captured["dept"] == "Engineering"
    ), "the author's own department when none is given"
    assert captured["tags"] == ["chip", lessons.LESSON_TAG]
    assert result["status"] == "published" and "Engineering" in result["shared_with"]


def test_a_lesson_tag_is_never_doubled(monkeypatch):
    _, captured = _save(monkeypatch, _CountDb(0), tags=[lessons.LESSON_TAG, "chip"])

    assert captured["tags"] == ["chip", lessons.LESSON_TAG]


def test_the_daily_quota_stops_a_runaway_assistant(monkeypatch):
    monkeypatch.setattr(settings, "LESSONS_MAX_PER_USER_PER_DAY", 3)

    with pytest.raises(HTTPException) as caught:
        _save(monkeypatch, _CountDb(3))

    assert caught.value.status_code == 429 and "3 lessons a day" in caught.value.detail
    _save(monkeypatch, _CountDb(2))  # still under it


def test_a_quota_of_zero_means_no_limit(monkeypatch):
    monkeypatch.setattr(settings, "LESSONS_MAX_PER_USER_PER_DAY", 0)

    _save(monkeypatch, _CountDb(10_000))


def test_without_any_department_it_says_so(monkeypatch):
    monkeypatch.setattr(USER, "dept", None)
    with pytest.raises(HTTPException) as caught:
        _save(monkeypatch, _CountDb(0))
    assert "department is required" in caught.value.detail.lower()


# -- changing and retracting ----------------------------------------------------------------------


class _WriteDb:
    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        self.commits += 1


def _lesson(owner=None, type_=LESSON_TYPE, status="published"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        type=type_,
        status=status,
        lifecycle_status="active",
        title="Console",
        body_md="## Lesson\n\nUART0 at 115200.",
        version=1,
        dept="Engineering",
        sensitivity="public",
        visibility="department",
        language="vi",
        tags=[SimpleNamespace(tag="session-lesson")],
        owner_id=owner or uuid.uuid4(),
        index_status="ready",
        index_error=None,
        last_reviewed=None,
        updated_at=None,
    )


@pytest.fixture
def lesson_store(monkeypatch):
    from src.domain import events
    from src.repositories.article import ArticleRepository

    from src.domain.articles import ArticleService

    holder = SimpleNamespace(article=None, published=[], deleted=[], checked=[])

    def can_create(self, user, dept):
        holder.checked.append(dept)

    monkeypatch.setattr(ArticleService, "ensure_can_create", can_create)

    async def get_by_id(self, article_id, user=None):
        return holder.article

    async def soft_delete(self, article_id, user=None, commit=True):
        holder.deleted.append(article_id)
        return True

    async def publish(event, payload):
        holder.published.append((event, payload["article_id"]))

    monkeypatch.setattr(ArticleRepository, "get_by_id", get_by_id)
    monkeypatch.setattr(ArticleRepository, "soft_delete", soft_delete)
    monkeypatch.setattr(events.event_bus, "publish", publish)
    return holder


def test_an_update_merges_into_the_lesson_as_a_new_version_and_is_indexed_again(
    lesson_store,
):
    lesson_store.article = _lesson()
    db = _WriteDb()

    result = _run(
        lessons.update_lesson(
            db,
            USER,
            str(lesson_store.article.id),
            merged="## Lesson\n\nNow 921600.",
            lesson="x",
            context=None,
        )
    )

    article = lesson_store.article
    assert article.body_md == "## Lesson\n\nNow 921600." and article.version == 2
    assert article.index_status == "pending"
    assert result["version"] == 2 and db.commits == 1
    kinds = {type(item).__name__ for item in db.added}
    assert kinds == {"ArticleVersion", "AuditLog"}
    version = next(item for item in db.added if type(item).__name__ == "ArticleVersion")
    assert version.version == 2 and version.edited_by == USER.id
    assert lesson_store.published == [("ArticleUpdated", str(article.id))]


def test_changing_a_lesson_needs_the_right_to_create_articles_in_its_department(
    lesson_store, monkeypatch
):
    """Being able to READ a lesson is not enough to rewrite it: it is published and indexed at once."""
    from src.domain.articles import ArticleService

    lesson_store.article = _lesson()
    original = lesson_store.article.body_md

    def refuse(self, user, dept):
        lesson_store.checked.append(dept)
        raise HTTPException(status_code=403, detail="Not authorized to create articles")

    monkeypatch.setattr(ArticleService, "ensure_can_create", refuse)
    db = _WriteDb()

    with pytest.raises(HTTPException) as caught:
        _run(
            lessons.update_lesson(
                db,
                USER,
                str(lesson_store.article.id),
                merged="rewritten",
                lesson="x",
                context=None,
            )
        )

    assert caught.value.status_code == 403
    assert lesson_store.checked == [
        "Engineering"
    ], "checked against the lesson's own department"
    assert (
        lesson_store.article.body_md == original and lesson_store.article.version == 1
    )
    assert db.commits == 0 and db.added == [] and lesson_store.published == []


def test_a_person_who_may_not_save_a_lesson_is_refused_before_anything_is_looked_up(
    monkeypatch,
):
    from src.domain.articles import ArticleService

    def refuse(self, user, dept):
        raise HTTPException(status_code=403, detail="Not authorized to create articles")

    async def decide(*_a, **_k):
        raise AssertionError("no retrieval or model call for someone who may not save")

    monkeypatch.setattr(ArticleService, "ensure_can_create", refuse)
    monkeypatch.setattr(lessons, "decide_lesson", decide)

    with pytest.raises(HTTPException) as caught:
        _run(
            knowledge.submit_lesson(
                knowledge.LessonRequest(title="t", lesson="l"),
                current_user=USER,
                db=object(),
            )
        )

    assert caught.value.status_code == 403


def test_without_merged_text_the_new_lesson_is_appended_as_a_dated_section(
    lesson_store,
):
    lesson_store.article = _lesson()

    _run(
        lessons.update_lesson(
            _WriteDb(),
            USER,
            str(lesson_store.article.id),
            merged=None,
            lesson="Now 921600.",
            context="After the rework.",
        )
    )

    body = lesson_store.article.body_md
    assert body.startswith("## Lesson\n\nUART0 at 115200.")
    assert f"## Update {datetime.utcnow():%Y-%m-%d}" in body
    assert "Now 921600." in body and "After the rework." in body


@pytest.mark.parametrize("kind", ["SOP", None])
def test_only_a_lesson_can_be_updated_this_way(lesson_store, kind):
    lesson_store.article = _lesson(type_=kind)

    with pytest.raises(HTTPException) as caught:
        _run(
            lessons.update_lesson(
                _WriteDb(),
                USER,
                str(uuid.uuid4()),
                merged="x",
                lesson="x",
                context=None,
            )
        )

    assert caught.value.status_code == 404


def test_the_author_can_retract_a_lesson(lesson_store):
    lesson_store.article = _lesson(owner=USER.id)
    db = _WriteDb()

    result = _run(lessons.retract_lesson(db, USER, str(lesson_store.article.id)))

    assert result["retracted"] is True and lesson_store.deleted == [
        lesson_store.article.id
    ]
    assert lesson_store.published == [("ArticleDeleted", str(lesson_store.article.id))]
    assert any(type(item).__name__ == "AuditLog" for item in db.added)


def test_someone_else_cannot_retract_it_without_the_right_to_delete(
    lesson_store, monkeypatch
):
    from src.domain.permissions import PermissionService

    lesson_store.article = _lesson()
    monkeypatch.setattr(
        PermissionService, "can_delete_article", classmethod(lambda cls, u, a: False)
    )

    with pytest.raises(HTTPException) as caught:
        _run(lessons.retract_lesson(_WriteDb(), USER, str(lesson_store.article.id)))

    assert caught.value.status_code == 403 and lesson_store.deleted == []


def test_whoever_may_delete_the_departments_articles_can_retract_it(
    lesson_store, monkeypatch
):
    from src.domain.permissions import PermissionService

    lesson_store.article = _lesson()
    monkeypatch.setattr(
        PermissionService, "can_delete_article", classmethod(lambda cls, u, a: True)
    )

    _run(lessons.retract_lesson(_WriteDb(), USER, str(lesson_store.article.id)))

    assert lesson_store.deleted == [lesson_store.article.id]


def test_a_reviewed_document_cannot_be_retracted_as_a_lesson(lesson_store):
    lesson_store.article = _lesson(owner=USER.id, type_="POLICY")

    with pytest.raises(HTTPException) as caught:
        _run(lessons.retract_lesson(_WriteDb(), USER, str(uuid.uuid4())))

    assert caught.value.status_code == 404 and lesson_store.deleted == []


# -- search and the assistant can leave lessons out -------------------------------------------------


class _ScopeRepo:
    def __init__(self, chunks):
        self.chunks, self.db, self.filters = chunks, None, []

    async def hybrid_search(
        self, *, query, query_embedding, user, limit, filters, identifiers=None
    ):
        self.filters.append(dict(filters))
        return list(self.chunks)


def _search_filters(monkeypatch, scope=None, filters=None):
    article = _article()
    repo = _ScopeRepo([_chunk(article, 0.9, "x")])
    service = SearchService(chunk_repo=repo, gov_repo=None, feature_flags=None)
    if scope is not None:
        service.source_scope = scope
    _run(service.search(_admin(), QUERY, filters=filters, limit=5))
    return repo.filters[0]


def test_search_covers_both_kinds_unless_told_otherwise(cross_encoder_on, monkeypatch):
    assert "source_scope" not in _search_filters(monkeypatch)


def test_a_scoped_search_filters_on_the_source_kind(cross_encoder_on, monkeypatch):
    assert (
        _search_filters(monkeypatch, scope="documents")["source_scope"] == "documents"
    )
    assert _search_filters(monkeypatch, scope="lessons")["source_scope"] == "lessons"


def test_an_explicit_filter_wins_over_the_services_scope(cross_encoder_on, monkeypatch):
    got = _search_filters(
        monkeypatch, scope="documents", filters={"source_scope": "lessons"}
    )
    assert got["source_scope"] == "lessons"


def test_the_chunk_filter_excludes_or_selects_lessons():
    from src.repositories.chunk import (
        ChunkRepository,
    )  # noqa: F401 - the constant it filters on
    import inspect

    source = inspect.getsource(ChunkRepository.hybrid_search)
    assert 'filters.get("source_scope") == "documents"' in source
    assert (
        "Article.type != LESSON_TYPE" in source
        and "Article.type == LESSON_TYPE" in source
    )


def test_the_assistant_reads_lessons_unless_the_request_or_the_default_says_not(
    monkeypatch,
):
    db = object()
    monkeypatch.setattr(settings, "AI_INCLUDE_LESSONS_DEFAULT", True)
    assert ai_router.get_ai_service(db).search_service.source_scope == "all"
    assert (
        ai_router.get_ai_service(db, "documents").search_service.source_scope
        == "documents"
    )
    assert ai_router.get_ai_service(db, "all").search_service.source_scope == "all"
    monkeypatch.setattr(settings, "AI_INCLUDE_LESSONS_DEFAULT", False)
    assert ai_router.get_ai_service(db).search_service.source_scope == "documents"
    assert ai_router.get_ai_service(db, "all").search_service.source_scope == "all"


def test_the_answer_cache_key_follows_the_sources_and_the_lessons_that_exist():
    class _Repo:
        def __init__(self, row):
            self.row = row
            self.db = self

        async def execute(self, _statement):
            return SimpleNamespace(one=lambda: self.row)

    def epoch(scope, row):
        service = SearchService(chunk_repo=None, gov_repo=None, feature_flags=None)
        service.source_scope = scope
        return _run(AIService(_Repo(row), service, None)._sources_epoch(USER))

    assert epoch("documents", (3, "t1")) == "documents"
    assert epoch("all", (3, "t1")) != epoch(
        "all", (4, "t2")
    ), "a new lesson must not be answered from the cache"
    assert epoch("all", (3, "t1")) == epoch("all", (3, "t1"))


def test_the_listing_is_limited_to_published_lessons_the_caller_may_read():
    class _Db:
        statements = []

        async def execute(self, statement):
            self.statements.append(str(statement))
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

    db = _Db()
    _run(
        knowledge.list_lessons(
            q="uart_%", mine=True, limit=5, current_user=_admin(), db=db
        )
    )

    sql = db.statements[0]
    assert (
        "articles.type" in sql
        and "articles.status" in sql
        and "articles.owner_id" in sql
    )
    assert "LIKE" in sql.upper()
