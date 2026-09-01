import uuid
from src.models.user import User
from src.models.article import Article
from src.domain.connector_providers import SOURCE_ACL_PROVIDERS
from src.domain.rbac import AuthorizationService

class PermissionService:
    @staticmethod
    def _explicit_user_effect(user: User, article: Article) -> str | None:
        effects = {
            override.effect
            for override in getattr(article, "user_permissions", []) or []
            if override.user_id == user.id
        }
        # Source-managed denies can coexist with an internal allow. Deny must
        # remain authoritative regardless of relationship/load ordering.
        if "deny" in effects:
            return "deny"
        if "allow" in effects:
            return "allow"
        return None

    @staticmethod
    def _source_acl_allows(user: User, article: Article) -> bool:
        """Apply the provider ACL even to global/company internal readers.

        Provider permissions are an intersection with the internal policy; a
        global Article permission is not a provider-side ACL bypass. The sync
        path represents mapped direct users as source-qualified allows and
        mapped groups through ``Article.departments``. Empty or unmapped
        provider ACLs therefore fail closed here.

        Every remote provider counts, not just SharePoint. This used to compare
        against the literal ``"sharepoint"``, so a OneDrive or Google Drive
        Article — whose provenance rows are stamped with ``connector.system`` —
        fell straight through to ``return True`` and served content the provider
        had not shared with the reader.
        """
        source_systems = {
            getattr(source, "source_system", None)
            for source in (getattr(article, "sources", []) or [])
        }
        governed_by = source_systems & set(SOURCE_ACL_PROVIDERS)
        if not governed_by:
            return True
        source_user_allow = any(
            override.user_id == user.id
            and override.effect == "allow"
            and override.source in governed_by
            for override in (getattr(article, "user_permissions", []) or [])
        )
        if source_user_allow:
            return True
        user_department_ids = {department.id for department in (getattr(user, "departments", []) or [])}
        article_department_ids = {department.id for department in (getattr(article, "departments", []) or [])}
        return bool(user_department_ids & article_department_ids)

    @classmethod
    def can_view_article(cls, user: User, article: Article) -> bool:
        if article.status == "deleted" or getattr(article, "lifecycle_status", "active") not in (None, "active"):
            return False
        if not any(AuthorizationService.has_permission(user, "article.read", article, scope) for scope in ("own", "department", "company", "global")):
            return False
        explicit_effect = cls._explicit_user_effect(user, article)
        if explicit_effect == "deny":
            return False
        if not cls._source_acl_allows(user, article):
            return False
        if getattr(article, "visibility", None) == "users":
            return explicit_effect == "allow"
        if article.status in {"draft", "pending_review", "archived"}:
            # Unpublished content is never ordinary knowledge-base content.
            # Owners and governance users may inspect it for review/history,
            # but a reader must not discover it through direct IDs or search.
            governance_access = any(
                AuthorizationService.has_permission(user, permission, article, scope)
                for permission in ("article.review", "article.publish", "article.edit", "article.delete")
                for scope in ("own", "department", "company", "global")
            )
            if article.owner_id != user.id and not governance_access:
                return False
        if AuthorizationService.has_full_company_article_access(user):
            return True
        if explicit_effect == "allow":
            return True
        if article.sensitivity == "restricted":
            # Restricted content requires an explicit shared department, never
            # a name-based match through `Article.dept`.
            user_department_ids = {department.id for department in getattr(user, "departments", []) or []}
            article_department_ids = {department.id for department in getattr(article, "departments", []) or []}
            if not user_department_ids & article_department_ids:
                return False
        if AuthorizationService.can_access_article_departments(user, article):
            return True
        return False

    @classmethod
    def can_edit_article(cls, user: User, article: Article) -> bool:
        return any(AuthorizationService.has_permission(user, "article.edit", article, scope) for scope in ("own", "department", "company"))

    @classmethod
    def can_delete_article(cls, user: User, article: Article) -> bool:
        return any(AuthorizationService.has_permission(user, "article.delete", article, scope) for scope in ("own", "department", "company"))
