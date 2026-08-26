import uuid
from src.models.user import User, AccessGroup
from src.models.article import Article
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
    def _sharepoint_acl_allows(user: User, article: Article) -> bool:
        """Apply the provider ACL even to global/company internal readers.

        SharePoint permissions are an intersection with the internal policy;
        a global Article permission is not a provider-side ACL bypass. The
        sync path represents mapped direct users as source-qualified allows
        and mapped groups through ``Article.access_groups``. Empty or
        unmapped provider ACLs therefore fail closed here.
        """
        if not any(
            getattr(source, "source_system", None) == "sharepoint"
            for source in (getattr(article, "sources", []) or [])
        ):
            return True
        source_user_allow = any(
            override.user_id == user.id
            and override.effect == "allow"
            and override.source == "sharepoint"
            for override in (getattr(article, "user_permissions", []) or [])
        )
        if source_user_allow:
            return True
        user_group_ids = {group.id for group in (getattr(user, "groups", []) or [])}
        article_group_ids = {group.id for group in (getattr(article, "access_groups", []) or [])}
        if user_group_ids & article_group_ids:
            return True
        user_access_ids = {department.id for department in getattr(user, "departments", []) if getattr(department, "kind", "org") == "access"}
        article_access_ids = {department.id for department in getattr(article, "departments", []) if getattr(department, "kind", "org") == "access"}
        return bool(user_access_ids & article_access_ids)

    @staticmethod
    def get_public_bit() -> int:
        # Bit position 0 represents public access (always available to everyone)
        return 0

    @classmethod
    def calculate_user_bitmask(cls, user: User) -> int:
        """
        Calculates user's bitmask from their access groups.
        If user is Admin, we return a fully set bitmask (e.g. all 1s).
        All users get the public bit (position 0) automatically.
        """
        if AuthorizationService.has_permission(user, "article.read", requested_scope="global"):
            # Enable first 62 bits
            return (1 << 62) - 1
            
        bitmask = 1 << cls.get_public_bit()
        for group in user.groups:
            if group.bitmask_position is not None:
                bitmask |= (1 << group.bitmask_position)
        return bitmask

    @classmethod
    def calculate_article_bitmask(cls, article: Article) -> int:
        """
        Calculates an article's access group bitmask.
        If sensitivity is public, it returns just the public bit.
        Otherwise, it returns the bitwise OR of all allowed access group bitmask positions.
        """
        if article.sensitivity == "public":
            return 1 << cls.get_public_bit()

        bitmask = 0
        for group in article.access_groups:
            if group.bitmask_position is not None:
                bitmask |= (1 << group.bitmask_position)
        
        # Restricted/internal articles without an explicit access group must
        # fail closed. Treating them as public leaks documents whenever an
        # editor forgets to select a group.
        return bitmask

    @classmethod
    def can_view_article(cls, user: User, article: Article) -> bool:
        if article.status == "deleted" or getattr(article, "lifecycle_status", "active") not in (None, "active"):
            return False
        if not any(AuthorizationService.has_permission(user, "article.read", article, scope) for scope in ("own", "department", "company", "global")):
            return False
        explicit_effect = cls._explicit_user_effect(user, article)
        if explicit_effect == "deny":
            return False
        if not cls._sharepoint_acl_allows(user, article):
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
            user_group_ids = {group.id for group in getattr(user, "groups", []) or []}
            article_group_ids = {group.id for group in getattr(article, "access_groups", []) or []}
            if not user_group_ids & article_group_ids:
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
