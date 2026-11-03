import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from src.domain.articles import ArticleService
from src.domain.permissions import PermissionService
from src.models.article import Article, ArticleUserPermission, DocumentSource
from src.models.interaction import Vote
from src.models.governance import PendingDraft
from src.models.rbac import Permission, Role, RolePermission
from src.models.user import Department, User
from src.repositories.article import ArticleRepository
from src.repositories.user import UserRepository
from src.repositories.governance import GovernanceRepository


def test_historical_citation_hydration_uses_sql_article_authorization():
    from src.api.routers.ai import _hydrate_citations

    db = _DB()
    user = _reader()
    asyncio.run(_hydrate_citations(db, user, [{"chunk_id": str(uuid.uuid4())}]))
    sql = _compiled(db.statement).lower()

    assert "articles.company_domain" in sql
    assert "articles.visibility" in sql
    assert "article_user_permissions" in sql
    assert "articles.lifecycle_status" in sql


def test_historical_citation_hydration_rebuilds_metadata_from_authorized_rows():
    from src.api.routers.ai import _hydrate_citations

    chunk_id = uuid.uuid4()
    article_id = uuid.uuid4()
    child = SimpleNamespace(
        id=uuid.uuid4(), chunk_index=0, chunk_text="authorized child"
    )
    parent = SimpleNamespace(
        id=uuid.uuid4(),
        text="authorized parent passage",
        section_ref="Section 1",
        heading="Authorized heading",
        page_number=4,
        child_chunks=[child],
    )
    chunk = SimpleNamespace(
        id=chunk_id,
        article_id=article_id,
        chunk_text="authorized child",
        heading=None,
        page_number=None,
        parent_chunk=parent,
        article=SimpleNamespace(id=article_id, title="Authorized Article"),
    )
    db = _DB([chunk])

    hydrated = asyncio.run(
        _hydrate_citations(
            db,
            _reader(),
            [
                {
                    "source_id": "C1",
                    "chunk_id": str(chunk_id),
                    "title": "UNTRUSTED TITLE",
                    "excerpt": "UNTRUSTED EXCERPT",
                    "source_url": "/api/v1/articles/unauthorized/source",
                }
            ],
        )
    )

    assert hydrated[0]["title"] == "Authorized Article"
    assert hydrated[0]["excerpt"] == "authorized parent passage"
    assert hydrated[0]["source_url"] == f"/api/v1/articles/{article_id}/source?page=4"
    assert hydrated[0]["source_ref"] == "Authorized Article - Authorized heading"


def test_ai_feedback_usage_log_lookup_is_user_scoped_in_sql():
    from src.repositories.ai import AIRepository

    db = _DB()
    user = _reader()

    asyncio.run(AIRepository(db).get_usage_log(uuid.uuid4(), user.id))
    sql = _compiled(db.statement).lower()

    assert "ai_usage_logs.id" in sql
    assert "ai_usage_logs.user_id" in sql


class _Result:
    rowcount = 0

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def scalars(self):
        return self

    def all(self):
        return self.rows

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _DB:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _Result(self.rows)

    async def scalar(self, statement):
        self.statement = statement
        return 0


class _MutationDB:
    def __init__(self):
        self.statements = []
        self.added = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result()

    async def commit(self):
        return None

    async def refresh(self, _item):
        return None

    def add(self, item):
        self.added.append(item)


def _compiled(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _reader(*, company="acme.test", department="Engineering") -> User:
    user = User(id=uuid.uuid4(), role="Staff", company_domain=company, dept=department)
    user.departments = [
        Department(
            id=uuid.uuid4(), company_domain=company, name=department, active=True
        )
    ]
    permission = Permission(key="article.read", name="Read")
    role = Role(name="Reader", company_domain=company)
    role.permissions = [RolePermission(permission=permission, scope="company")]
    user.roles = [role]
    return user


def test_interaction_reads_include_article_authorization():
    from src.repositories.interaction import InteractionRepository

    db = _DB()
    user = _reader()
    repo = InteractionRepository(db)

    asyncio.run(repo.get_comments(uuid.uuid4(), user))
    sql = _compiled(db.statement).lower()
    assert "articles.company_domain" in sql
    assert "article_user_permissions" in sql

    asyncio.run(repo.get_votes_summary(uuid.uuid4(), user))
    assert "articles.visibility" in _compiled(db.statement).lower()

    asyncio.run(repo.get_user_vote(uuid.uuid4(), user.id, user))
    assert "articles.lifecycle_status" in _compiled(db.statement).lower()


def test_interaction_mutations_scope_vote_and_bookmark_queries_to_article_access():
    from src.repositories.interaction import InteractionRepository

    user = _reader()
    article_id = uuid.uuid4()

    vote_db = _MutationDB()
    repo = InteractionRepository(vote_db)
    asyncio.run(
        repo.cast_vote(Vote(article_id=article_id, user_id=user.id, value=1), user)
    )
    vote_sql = _compiled(vote_db.statements[0]).lower()
    assert "articles.company_domain" in vote_sql
    assert "article_user_permissions" in vote_sql
    assert "articles.lifecycle_status" in vote_sql

    bookmark_db = _MutationDB()
    repo = InteractionRepository(bookmark_db)
    asyncio.run(repo.remove_bookmark(user, article_id))
    delete_sql = _compiled(bookmark_db.statements[0]).lower()
    assert "articles.company_domain" in delete_sql
    assert "article_user_permissions" in delete_sql
    assert "articles.lifecycle_status" in delete_sql

    asyncio.run(repo.is_bookmarked(user, article_id))
    read_sql = _compiled(bookmark_db.statements[1]).lower()
    assert "articles.company_domain" in read_sql
    assert "article_user_permissions" in read_sql
    assert "articles.lifecycle_status" in read_sql


def test_article_get_by_id_returns_no_row_for_user_without_read_permission():
    db = _DB()
    user = User(id=uuid.uuid4(), role="Unassigned", company_domain="acme.test")

    article = asyncio.run(ArticleRepository(db).get_by_id(uuid.uuid4(), user=user))

    assert article is None
    assert "false" in _compiled(db.statement).lower()


def test_article_soft_delete_update_contains_authorization_predicate():
    db = _MutationDB()
    user = _reader()

    asyncio.run(ArticleRepository(db).soft_delete(uuid.uuid4(), user=user))
    sql = _compiled(db.statements[0]).lower()

    assert "articles.company_domain" in sql
    assert "article_user_permissions" in sql
    assert "articles.lifecycle_status" in sql


def test_article_version_queries_include_authorization_predicate():
    db = _DB()
    user = _reader()
    repo = ArticleRepository(db)
    article_id = uuid.uuid4()

    asyncio.run(repo.get_versions(article_id, user=user))
    history_sql = _compiled(db.statement).lower()
    assert "article_versions.article_id" in history_sql
    assert "articles.company_domain" in history_sql
    assert "article_user_permissions" in history_sql
    assert "articles.lifecycle_status" in history_sql

    asyncio.run(repo.get_version_by_number(article_id, 2, user=user))
    version_sql = _compiled(db.statement).lower()
    assert "article_versions.version" in version_sql
    assert "articles.company_domain" in version_sql
    assert "article_user_permissions" in version_sql
    assert "articles.lifecycle_status" in version_sql


def test_article_list_query_contains_tenant_and_department_audience_predicates():
    db = _DB()
    user = _reader()
    user.departments.append(
        Department(
            id=uuid.uuid4(),
            company_domain="acme.test",
            name="Security",
            active=True,
        )
    )

    asyncio.run(ArticleRepository(db).list_articles(user))
    sql = _compiled(db.statement).lower()

    assert "articles.company_domain" in sql
    assert "article_departments" in sql
    assert "departments.name in" in sql
    assert "departments.id in" in sql
    assert "articles.status" in sql


def test_shared_audience_department_reads_across_the_primary_department():
    """Audience matching is an OR, not an AND.

    A department the user shares is sufficient even when its name differs from
    the article's primary ``dept``; a user outside every audience department of
    the article stays out.
    """
    user = _reader(department="Engineering")
    engineering = user.departments[0]
    article = Article(
        company_domain="acme.test",
        dept="Finance",
        domain="Security",
        visibility="department",
        sensitivity="internal",
        status="published",
        lifecycle_status="active",
        departments=[engineering],
    )

    assert PermissionService.can_view_article(user, article) is True

    article.departments = [
        Department(
            id=uuid.uuid4(),
            company_domain="acme.test",
            name="Finance",
            active=True,
        )
    ]
    assert PermissionService.can_view_article(user, article) is False


def test_article_list_query_keeps_sharepoint_acl_for_global_readers():
    db = _DB()
    user = User(
        id=uuid.uuid4(), role="Admin", company_domain="acme.test", dept="Engineering"
    )
    permission = Permission(key="article.read", name="Read")
    role = Role(name="Admin", company_domain=None)
    role.permissions = [RolePermission(permission=permission, scope="global")]
    user.roles = [role]
    user.departments = [
        Department(
            id=uuid.uuid4(),
            company_domain="acme.test",
            name="Security",
            active=True,
        )
    ]

    asyncio.run(ArticleRepository(db).list_articles(user))
    sql = _compiled(db.statement).lower()

    assert "document_sources" in sql
    assert "document_sources.source_system" in sql
    assert "article_departments" in sql
    assert "departments.id in" in sql


def test_similarity_query_uses_the_shared_sql_article_authorization_predicate():
    from src.domain.similarity import find_similar_documents

    db = _DB()
    user = _reader()

    asyncio.run(find_similar_documents(db, user, "policy text"))
    sql = _compiled(db.statement).lower()

    assert "articles.company_domain" in sql
    assert "article_user_permissions" in sql
    assert "articles.lifecycle_status" in sql


def test_search_applies_effective_department_and_owner_scope_in_sql():
    from src.repositories.chunk import ChunkRepository

    class SearchResult(_Result):
        def scalar_one(self):
            return 0

    class SearchDB:
        def __init__(self):
            self.statements = []

        async def execute(self, statement):
            self.statements.append(statement)
            return SearchResult()

    db = SearchDB()
    user = _reader()
    asyncio.run(
        ChunkRepository(db).hybrid_search(
            query="policy",
            query_embedding=None,
            user=user,
            filters={"departments": ["Finance"], "owner_id": user.id},
        )
    )
    sql = "\n".join(_compiled(statement).lower() for statement in db.statements)

    assert "articles.sensitivity" in sql
    assert "articles.dept in" in sql
    assert "articles.owner_id" in sql


def test_article_list_query_excludes_deleted_and_inactive_rows_in_sql():
    db = _DB()
    user = _reader()

    asyncio.run(ArticleRepository(db).list_articles(user))
    sql = _compiled(db.statement).lower()

    assert "articles.lifecycle_status" in sql
    assert "articles.status !=" in sql or "articles.status <>" in sql


def test_article_list_orders_home_cards_by_updated_then_created_time():
    db = _DB()
    user = _reader()

    asyncio.run(ArticleRepository(db).list_articles(user, status="published"))
    sql = _compiled(db.statement).lower()

    assert "articles.updated_at desc" in sql
    assert "articles.created_at desc" in sql


def test_pending_home_count_is_scoped_to_review_access_and_assignment():
    db = _DB()
    user = _reader()
    review = Permission(key="article.review", name="Review")
    user.roles[0].permissions.append(RolePermission(permission=review, scope="company"))

    asyncio.run(GovernanceRepository(db).count_pending_for_user(user))
    sql = _compiled(db.statement).lower()

    assert "pending_drafts.status" in sql
    assert "pending_drafts.company_domain" in sql
    assert "pending_drafts.assigned_approver_id" in sql
    assert "pending_drafts.dept" in sql


def test_pending_home_count_is_zero_for_users_without_review_access():
    db = _DB()
    user = _reader()

    assert asyncio.run(GovernanceRepository(db).count_pending_for_user(user)) == 0
    assert db.statement is None


def test_article_service_passes_actor_to_scoped_repository():
    user = _reader()

    class ScopedRepository:
        async def get_by_id(self, article_id, user=None):
            assert user is user_for_assertion
            return None

    user_for_assertion = user
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            ArticleService(ScopedRepository(), object()).get_article(user, uuid.uuid4())
        )

    assert error.value.status_code == 404


def test_user_list_query_is_company_scoped_for_company_manager():
    db = _DB()
    viewer = _reader()
    permission = Permission(key="user.read", name="Read users")
    viewer.roles[0].permissions.append(
        RolePermission(permission=permission, scope="company")
    )

    asyncio.run(UserRepository(db).list_users(viewer=viewer))
    sql = _compiled(db.statement)

    assert "users.company_domain" in sql


def test_eligible_approver_user_query_pushes_resource_predicates_into_sql():
    db = _DB()

    asyncio.run(
        UserRepository(db).list_users(
            limit=500,
            company_domain="acme.test",
            active=True,
        )
    )
    sql = _compiled(db.statement).lower()

    assert "users.company_domain" in sql
    assert "users.active" in sql


def test_department_member_queries_are_company_scoped():
    db = _DB()
    repo = UserRepository(db)

    asyncio.run(
        repo.get_departments_by_ids([uuid.uuid4()], company_domain="acme.test")
    )
    assert "departments.company_domain" in _compiled(db.statement).lower()

    asyncio.run(repo.get_by_ids([uuid.uuid4()], company_domain="acme.test"))
    assert "users.company_domain" in _compiled(db.statement).lower()


def test_connector_management_lookup_is_company_scoped():
    from src.api.routers.connectors import _connector_for_user

    db = _DB()
    user = _reader()
    permission = Permission(key="connector.manage", name="Manage connectors")
    user.roles[0].permissions.append(
        RolePermission(permission=permission, scope="company")
    )

    asyncio.run(_connector_for_user(db, uuid.uuid4(), user))
    sql = _compiled(db.statement).lower()
    assert "connectors.company_domain" in sql


def test_role_management_queries_are_company_scoped():
    from src.api.routers import auth

    db = _DB()
    user = _reader()
    permission = Permission(key="role.manage", name="Manage roles")
    user.roles[0].permissions.append(
        RolePermission(permission=permission, scope="company")
    )

    with pytest.raises(HTTPException):
        asyncio.run(
            auth.update_role(uuid.uuid4(), auth.RoleUpdate(), current_user=user, db=db)
        )
    assert "roles.company_domain" in _compiled(db.statement).lower()


def test_explicit_user_visibility_grants_cross_department_read():
    user = _reader(department="Engineering")
    article = Article(
        company_domain="acme.test",
        dept="Finance",
        visibility="users",
        sensitivity="restricted",
        user_permissions=[ArticleUserPermission(user_id=user.id, effect="allow")],
    )

    assert PermissionService.can_view_article(user, article) is True


def test_explicit_deny_wins_over_public_and_role_access():
    user = _reader()
    article = Article(
        company_domain="acme.test",
        dept="Engineering",
        visibility="public",
        sensitivity="public",
        user_permissions=[ArticleUserPermission(user_id=user.id, effect="deny")],
    )

    assert PermissionService.can_view_article(user, article) is False


def test_sharepoint_acl_restricts_global_reader_to_mapped_department_or_source_user():
    department = Department(
        id=uuid.uuid4(), company_domain="acme.test", name="Security", active=True
    )
    user = User(
        id=uuid.uuid4(), role="Admin", company_domain="acme.test", dept="Engineering"
    )
    permission = Permission(key="article.read", name="Read")
    role = Role(name="Admin", company_domain=None)
    role.permissions = [RolePermission(permission=permission, scope="global")]
    user.roles = [role]
    article = Article(
        title="Provider policy",
        body_md="body",
        company_domain="acme.test",
        dept="Engineering",
        domain="Security",
        type="POLICY",
        sensitivity="restricted",
        visibility="department",
        status="published",
        lifecycle_status="active",
        departments=[department],
        sources=[
            DocumentSource(
                source_system="sharepoint", source_ref="file-1", source_hash="hash"
            )
        ],
    )

    assert PermissionService.can_view_article(user, article) is False
    user.departments = [department]
    assert PermissionService.can_view_article(user, article) is True

    user.departments = []
    article.departments = []
    article.visibility = "users"
    article.user_permissions = [
        ArticleUserPermission(user_id=user.id, effect="allow", source="sharepoint")
    ]
    assert PermissionService.can_view_article(user, article) is True


def test_pending_draft_lookup_contains_actor_tenant_and_assignment_scope():
    db = _DB()
    user = _reader()

    import asyncio

    asyncio.run(GovernanceRepository(db).get_draft_for_user(uuid.uuid4(), user))
    sql = _compiled(db.statement)

    assert "pending_drafts.company_domain" in sql
    assert "pending_drafts.assigned_approver_id" in sql
    assert "pending_drafts.dept" in sql
