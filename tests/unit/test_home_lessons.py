"""The home page speaks about reviewed documents; team lessons (published without a reviewer) are counted apart."""

import asyncio
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from src.api.routers import knowledge
from src.models.article import LESSON_TYPE
from src.repositories.article import ArticleRepository
from src.repositories.governance import GovernanceRepository
from tests.unit.test_query_authorization import _DB, _reader


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect())).lower()


class _RecordingDb(_DB):
    """Answers every count with a number and remembers each statement."""

    def __init__(self):
        super().__init__()
        self.statements = []

    async def scalar(self, statement):
        self.statements.append(statement)
        return 4


def test_a_list_can_leave_one_article_type_out():
    db = _DB()

    asyncio.run(
        ArticleRepository(db).list_articles(
            _reader(), status="published", exclude_type=LESSON_TYPE
        )
    )

    sql = _sql(db.statement)
    assert "articles.type is distinct from" in sql, "a missing type is not a lesson"
    assert "articles.status =" in sql


def test_a_list_without_it_keeps_every_type():
    db = _DB()

    asyncio.run(ArticleRepository(db).list_articles(_reader(), status="published"))

    assert "is distinct from" not in _sql(db.statement)


def _home(monkeypatch):
    asked = {}

    async def list_articles(self, user, **kwargs):
        asked.update(kwargs)
        return []

    async def pending(self, user):
        return 0

    monkeypatch.setattr(ArticleRepository, "list_articles", list_articles)
    monkeypatch.setattr(GovernanceRepository, "count_pending_for_user", pending)
    db = _RecordingDb()
    result = asyncio.run(knowledge.home_summary(current_user=_reader(), db=db))
    return result, asked, [_sql(statement) for statement in db.statements]


def test_the_recent_list_is_reviewed_documents_only(monkeypatch):
    _, asked, _ = _home(monkeypatch)

    assert asked["exclude_type"] == LESSON_TYPE and asked["status"] == "published"


def test_the_figures_are_about_documents_and_lessons_are_counted_apart(monkeypatch):
    result, _, statements = _home(monkeypatch)

    about_articles = [s for s in statements if "from articles" in s and "count(" in s]
    assert about_articles, "the home page counts articles"
    owner, lessons = None, None
    for sql in about_articles:
        if "articles.type = " in sql:
            lessons = sql
        elif "owner_id is not null" in sql:
            owner = sql
    assert lessons is not None and "is distinct from" not in lessons
    assert owner is not None and "articles.type is distinct from" in owner
    others = [s for s in about_articles if s not in (lessons, owner)]
    assert others and all("articles.type is distinct from" in s for s in others)
    assert result["lessons"] == 4 and "total_articles" in result


def test_a_card_says_what_kind_of_article_it_is():
    article = SimpleNamespace(
        id="a1",
        title="T",
        dept="Engineering",
        company_domain="acme.test",
        departments=[],
        external_id=None,
        type=LESSON_TYPE,
        status="published",
        language="en",
        version=1,
        owner=None,
        owner_id=None,
        tags=[],
        related_article_ids=None,
        next_review=None,
        last_reviewed=None,
        needs_update=False,
        source_changed=False,
        sources=[],
    )

    assert knowledge._article_card(article)["type"] == LESSON_TYPE


def test_the_governance_health_figures_leave_team_lessons_out():
    class Result:
        def __init__(self, columns):
            self.columns = columns

        def scalar_one(self):
            return 4

        def one(self):
            return tuple(0 for _ in range(self.columns))

    class Db:
        def __init__(self):
            self.statements = []

        async def execute(self, statement):
            self.statements.append(statement)
            return Result(len(statement.selected_columns))

    db = Db()

    result = asyncio.run(GovernanceRepository(db).get_health_metrics())

    counted = [_sql(s) for s in db.statements if "from articles" in _sql(s)]
    assert len(counted) == 3, "documents, owned documents, overdue documents"
    assert all("articles.type is distinct from" in sql for sql in counted)
    assert result["total_articles"] == 4
