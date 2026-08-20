import uuid
from typing import Sequence, Any
from sqlalchemy import select, update, and_, or_, exists, false, true, not_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.article import (
    Article,
    ArticleVersion,
    ArticleUserPermission,
    ArticleTag,
    DocumentSource,
    article_access,
)
from src.models.user import User, Department
from src.domain.rbac import AuthorizationService
from src.domain.permissions import PermissionService


class ArticleRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _has_article_read_permission(user: User) -> bool:
        return any(
            AuthorizationService.has_permission(
                user, "article.read", requested_scope=scope
            )
            for scope in ("own", "department", "company", "global")
        )

    @classmethod
    def _authorized_article_filters(cls, user: User) -> list[Any]:
        """Build the Article visibility predicate used by every read query.

        This predicate intentionally contains the same tenant, department,
        group, ownership, and workflow boundaries as ``PermissionService``.
        The repository never relies on fetching a broad candidate set and
        filtering it in Python, which would expose private rows to the
        application process and violate the data-layer authorization rule.
        """
        filters: list[Any] = [
            Article.lifecycle_status == "active",
            Article.status != "deleted",
            exists(
                select(Department.id).where(
                    Department.company_domain == Article.company_domain,
                    Department.name == Article.dept,
                    Department.active.is_(True),
                )
            ),
        ]
        explicit_allow = Article.user_permissions.any(
            and_(
                ArticleUserPermission.user_id == user.id,
                ArticleUserPermission.effect == "allow",
            )
        )
        explicit_deny = Article.user_permissions.any(
            and_(
                ArticleUserPermission.user_id == user.id,
                ArticleUserPermission.effect == "deny",
            )
        )
        source_explicit_allow = Article.user_permissions.any(
            and_(
                ArticleUserPermission.user_id == user.id,
                ArticleUserPermission.effect == "allow",
                ArticleUserPermission.source == "sharepoint",
            )
        )
        # An explicit deny is evaluated before every role, department, group,
        # and public allow. This predicate must stay in SQL so denied content
        # never enters the application result set.
        filters.append(not_(explicit_deny))
        if not cls._has_article_read_permission(user):
            return [false()]

        global_read = AuthorizationService.has_permission(
            user, "article.read", requested_scope="global"
        )
        full_company_access = AuthorizationService.has_full_company_article_access(user)
        if not global_read:
            filters.append(Article.company_domain == user.company_domain)

        if not (global_read or full_company_access):
            member_departments = AuthorizationService.member_department_names(user)
            if user.dept:
                member_departments.add(user.dept)
            if not member_departments:
                department_scope = false()
            else:
                department_scope = or_(
                    Article.dept.in_(member_departments),
                    Article.departments.any(Department.name.in_(member_departments)),
                )
            permission_conditions: list[Any] = [
                Article.sensitivity == "public",
                Article.owner_id == user.id,
                explicit_allow,
            ]
            group_ids = [group.id for group in getattr(user, "groups", [])]
            access_audience_ids = [department.id for department in getattr(user, "departments", []) if getattr(department, "kind", "org") == "access"]
            if group_ids:
                permission_conditions.append(
                    Article.access_groups.any(article_access.c.group_id.in_(group_ids))
                )
            if access_audience_ids:
                permission_conditions.append(Article.departments.any(Department.id.in_(access_audience_ids)))
            if AuthorizationService.has_permission(
                user, "article.read", requested_scope="department"
            ):
                owned_departments = AuthorizationService.owned_department_names(user)
                if owned_departments:
                    permission_conditions.append(
                        or_(
                            Article.dept.in_(owned_departments),
                            Article.departments.any(
                                Department.name.in_(owned_departments)
                            ),
                        )
                    )
            # Departments and access groups are the same read-time audience
            # concept.  Do not require both, otherwise a cross-department
            # access-group member is incorrectly denied.
            non_user_scope = and_(
                Article.visibility != "users",
                or_(department_scope, *permission_conditions),
            )
            user_scope = and_(Article.visibility == "users", explicit_allow)
            filters.append(or_(non_user_scope, user_scope))

        else:
            # Global/company-wide readers may see all normal visibility modes,
            # but explicit-user mode remains an explicit allow-list and deny
            # rows still win.
            filters.append(or_(Article.visibility != "users", explicit_allow))

        # SharePoint ACLs are an intersection with the internal policy. Keep
        # this predicate in every Article query so a global/company reader
        # cannot bypass a mapped group, mapped direct user, or fail-closed
        # provider ACL through the broad internal branch above.
        source_acl_article = Article.sources.any(
            DocumentSource.source_system == "sharepoint"
        )
        source_acl_allows: list[Any] = [source_explicit_allow]
        group_ids = [group.id for group in getattr(user, "groups", [])]
        access_audience_ids = [department.id for department in getattr(user, "departments", []) if getattr(department, "kind", "org") == "access"]
        if group_ids:
            source_acl_allows.append(
                Article.access_groups.any(article_access.c.group_id.in_(group_ids))
            )
        if access_audience_ids:
            source_acl_allows.append(Article.departments.any(Department.id.in_(access_audience_ids)))
        filters.append(
            or_(
                not_(source_acl_article),
                and_(source_acl_article, or_(*source_acl_allows)),
            )
        )

        # Published content is the only ordinary knowledge-base surface.
        # Owners and governance users may inspect working states, but only
        # within the same SQL scope used for the article itself.
        unpublished_conditions: list[Any] = []
        if AuthorizationService.has_permission(
            user, "article.review", requested_scope="global"
        ) or AuthorizationService.has_permission(
            user, "article.publish", requested_scope="global"
        ):
            unpublished_conditions.append(true())
        elif AuthorizationService.has_permission(
            user, "article.review", requested_scope="company"
        ) or AuthorizationService.has_permission(
            user, "article.publish", requested_scope="company"
        ):
            unpublished_conditions.append(true())
        else:
            if AuthorizationService.has_permission(
                user, "article.review", requested_scope="department"
            ) or AuthorizationService.has_permission(
                user, "article.publish", requested_scope="department"
            ):
                owned_departments = AuthorizationService.owned_department_names(user)
                if owned_departments:
                    unpublished_conditions.append(
                        or_(
                            Article.dept.in_(owned_departments),
                            Article.departments.any(
                                Department.name.in_(owned_departments)
                            ),
                        )
                    )
            if any(
                AuthorizationService.has_permission(user, key, requested_scope="own")
                for key in (
                    "article.review",
                    "article.publish",
                    "article.edit",
                    "article.delete",
                )
            ):
                unpublished_conditions.append(Article.owner_id == user.id)

        if unpublished_conditions:
            filters.append(or_(Article.status == "published", *unpublished_conditions))
        else:
            filters.append(Article.status == "published")
        return filters

    async def get_by_id(
        self, article_id: uuid.UUID, user: User | None = None
    ) -> Article | None:
        filters = [Article.id == article_id]
        if user is not None:
            filters.extend(self._authorized_article_filters(user))
        result = await self.db.execute(
            select(Article)
            .where(and_(*filters))
            .options(
                selectinload(Article.access_groups),
                selectinload(Article.departments),
                selectinload(Article.tags),
                selectinload(Article.owner),
                selectinload(Article.sources),
                selectinload(Article.user_permissions).selectinload(
                    ArticleUserPermission.user
                ),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_update(
        self, article_id: uuid.UUID, user: User | None = None
    ) -> Article | None:
        """Load an article row with a transaction lock for state transitions."""
        filters = [Article.id == article_id]
        if user is not None:
            filters.extend(self._authorized_article_filters(user))
        result = await self.db.execute(
            select(Article)
            .where(and_(*filters))
            .with_for_update()
            .options(
                selectinload(Article.access_groups),
                selectinload(Article.departments),
                selectinload(Article.tags),
                selectinload(Article.owner),
                selectinload(Article.sources),
                selectinload(Article.user_permissions).selectinload(
                    ArticleUserPermission.user
                ),
            )
        )
        return result.scalar_one_or_none()

    async def list_articles(
        self,
        user: User,
        dept: str | None = None,
        type_: str | None = None,
        sensitivity: str | None = None,
        status: str | None = None,
        topic: str | None = None,
        search_query: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Article]:
        stmt = select(Article).options(
            selectinload(Article.access_groups),
            selectinload(Article.departments),
            selectinload(Article.tags),
            selectinload(Article.owner),
            selectinload(Article.sources),
            selectinload(Article.user_permissions).selectinload(
                ArticleUserPermission.user
            ),
        )

        filters = self._authorized_article_filters(user)

        # Apply standard filters
        if dept:
            filters.append(
                or_(
                    Article.dept == dept,
                    Article.departments.any(Department.name == dept),
                )
            )
        if type_:
            filters.append(Article.type == type_)
        if sensitivity:
            filters.append(Article.sensitivity == sensitivity)
        if status:
            filters.append(Article.status == status)
        if topic:
            filters.append(
                ~Article.tags.any() if topic == "General knowledge"
                else Article.tags.any(ArticleTag.tag == topic)
            )
        if search_query:
            filters.append(Article.title.ilike(f"%{search_query}%"))

        if filters:
            stmt = stmt.where(and_(*filters))

        # Unbounded list queries load and serialize the whole authorized
        # corpus per call; the API layer always bounds the page size.
        if limit is None:
            limit = 200
        stmt = stmt.order_by(Article.updated_at.desc(), Article.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        articles = result.scalars().all()
        for article in articles:
            AuthorizationService.restrict_article_metadata(user, article)
        return articles

    async def list_related(
        self, user: User, article: Article, limit: int = 6
    ) -> Sequence[Article]:
        """Return published, authorized articles related by taxonomy or tags."""
        stmt = (
            select(Article)
            .options(
                selectinload(Article.access_groups),
                selectinload(Article.departments),
                selectinload(Article.tags),
                selectinload(Article.owner),
                selectinload(Article.sources),
            )
            .where(
                Article.id != article.id,
                Article.status == "published",
                Article.lifecycle_status == "active",
                exists(
                    select(Department.id).where(
                        Department.company_domain == Article.company_domain,
                        Department.name == Article.dept,
                        Department.active.is_(True),
                    )
                ),
                or_(
                    or_(
                        Article.dept == article.dept,
                        Article.departments.any(Department.name == article.dept),
                    ),
                    Article.domain == article.domain,
                    Article.tags.any(
                        ArticleTag.tag.in_(
                            select(ArticleTag.tag).where(
                                ArticleTag.article_id == article.id
                            )
                        )
                    ),
                ),
            )
        )
        stmt = stmt.where(and_(*self._authorized_article_filters(user)))
        result = await self.db.execute(
            stmt.order_by(Article.created_at.desc()).limit(limit * 3)
        )
        candidates = result.scalars().all()
        for candidate in candidates:
            AuthorizationService.restrict_article_metadata(user, candidate)
        return candidates[:limit]

    async def create(self, article: Article, *, commit: bool = True) -> Article:
        self.db.add(article)
        if commit:
            await self.db.commit()
        else:
            await self.db.flush()
        await self.db.refresh(article)
        return article

    async def update(self, article: Article, *, commit: bool = True) -> Article:
        self.db.add(article)
        if commit:
            await self.db.commit()
        else:
            await self.db.flush()
        await self.db.refresh(article)
        return article

    async def soft_delete(
        self, article_id: uuid.UUID, user: User | None = None, *, commit: bool = True
    ) -> bool:
        stmt = update(Article).where(Article.id == article_id)
        if user is not None:
            stmt = stmt.where(and_(*self._authorized_article_filters(user)))
        result = await self.db.execute(stmt.values(status="deleted"))
        if commit:
            await self.db.commit()
        return result.rowcount > 0

    async def create_version(
        self, version: ArticleVersion, *, commit: bool = True
    ) -> ArticleVersion:
        self.db.add(version)
        if commit:
            await self.db.commit()
        else:
            await self.db.flush()
        await self.db.refresh(version)
        return version

    async def get_versions(
        self, article_id: uuid.UUID, user: User | None = None
    ) -> Sequence[ArticleVersion]:
        filters: list[Any] = [ArticleVersion.article_id == article_id]
        if user is not None:
            # Historical snapshots contain prior Article body/title data. Keep
            # the version query itself inside the same SQL authorization
            # boundary as the current Article lookup instead of relying on the
            # caller's earlier check.
            filters.extend(self._authorized_article_filters(user))
        result = await self.db.execute(
            select(ArticleVersion)
            .join(Article, Article.id == ArticleVersion.article_id)
            .where(and_(*filters))
            .order_by(ArticleVersion.version.desc())
            .options(selectinload(ArticleVersion.editor))
        )
        return result.scalars().all()

    async def get_version_by_number(
        self,
        article_id: uuid.UUID,
        version_num: int,
        user: User | None = None,
    ) -> ArticleVersion | None:
        filters: list[Any] = [
            ArticleVersion.article_id == article_id,
            ArticleVersion.version == version_num,
        ]
        if user is not None:
            filters.extend(self._authorized_article_filters(user))
        result = await self.db.execute(
            select(ArticleVersion)
            .join(Article, Article.id == ArticleVersion.article_id)
            .where(and_(*filters))
            .options(selectinload(ArticleVersion.editor))
        )
        return result.scalar_one_or_none()

    async def sync_tags(
        self, article_id: uuid.UUID, tags: list[str], *, commit: bool = True
    ) -> None:
        # Delete existing tags
        from sqlalchemy import delete

        await self.db.execute(
            delete(ArticleTag).where(ArticleTag.article_id == article_id)
        )

        # Add new tags
        for t in tags:
            self.db.add(ArticleTag(article_id=article_id, tag=t))
        if commit:
            await self.db.commit()
        else:
            # Make the new rows visible to relationship reloads inside the
            # caller's transaction (sessions run with autoflush=False).
            await self.db.flush()
