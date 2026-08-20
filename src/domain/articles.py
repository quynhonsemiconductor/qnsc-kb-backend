import uuid
import structlog
from datetime import datetime
from typing import Sequence
from fastapi import HTTPException, status
from src.models.article import Article, ArticleVersion, ArticleTag
from src.models.user import User
from src.repositories.article import ArticleRepository
from src.repositories.user import UserRepository
from src.domain.permissions import PermissionService
from src.domain.rbac import AuthorizationService
from src.domain.events import event_bus
from src.repositories.audit import AuditRepository
from src.domain.departments import resolve_active_department

logger = structlog.get_logger()


class ArticleService:
    def __init__(
        self,
        article_repo: ArticleRepository,
        user_repo: UserRepository,
        audit_repo: AuditRepository | None = None,
    ):
        self.article_repo = article_repo
        self.user_repo = user_repo
        self.audit_repo = audit_repo

    async def _audit(
        self,
        user_id: uuid.UUID,
        action: str,
        article_id: uuid.UUID,
        *,
        commit: bool = True,
    ) -> None:
        if self.audit_repo:
            await self.audit_repo.record(
                user_id, action, "article", str(article_id), commit=commit
            )

    def ensure_can_create(self, user: User, dept: str) -> None:
        """Fail before any costly side effect, including AI restructuring."""
        draft_resource = Article(
            company_domain=user.company_domain, dept=dept, owner_id=user.id
        )
        if not any(
            AuthorizationService.has_permission(
                user, "article.create", draft_resource, scope
            )
            for scope in ("own", "department", "company", "global")
        ):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to create articles in this department",
            )

    async def create_article(
        self,
        user: User,
        title: str,
        body_md: str,
        dept: str,
        domain: str,
        type_: str,
        sensitivity: str,
        tags: list[str],
        language: str = "vi",
        access_group_ids: list[uuid.UUID] | None = None,
        next_review: datetime | None = None,
        external_id: str | None = None,
        original_body_md: str | None = None,
    ) -> Article:
        department = await resolve_active_department(
            self.article_repo.db, user.company_domain, dept
        )
        dept = department.name
        self.ensure_can_create(user, dept)
        draft_resource = Article(
            company_domain=user.company_domain, dept=dept, owner_id=user.id
        )

        # Resolve access groups
        groups = []
        if access_group_ids:
            groups = list(
                await self.user_repo.get_groups_by_ids(
                    access_group_ids, user.company_domain
                )
            )
            if len({group.id for group in groups}) != len(set(access_group_ids)):
                raise HTTPException(
                    status_code=422, detail="One or more access groups do not exist"
                )

        if sensitivity not in {"public", "internal", "confidential", "restricted"}:
            raise HTTPException(status_code=422, detail="Invalid article sensitivity")
        if sensitivity != "public" and not groups:
            raise HTTPException(
                status_code=422,
                detail="Non-public articles require at least one access group",
            )

        # Default sensitivity and status logic:
        # Department owners/admins can publish directly, staff create drafts
        can_publish = any(
            AuthorizationService.has_permission(
                user, "article.publish", draft_resource, scope
            )
            for scope in ("own", "department", "company", "global")
        )
        initial_status = "published" if can_publish else "draft"

        article = Article(
            title=title,
            external_id=external_id,
            body_md=body_md,
            dept=dept,
            domain=domain,
            company_domain=user.company_domain,
            type=type_,
            sensitivity=sensitivity,
            language=language,
            owner_id=user.id,
            status=initial_status,
            version=1,
            next_review=next_review,
            last_reviewed=datetime.utcnow() if initial_status == "published" else None,
            access_groups=groups,
            departments=[department],
        )

        # Single transaction: article row, tags, version snapshot, and audit
        # entries commit together (the approve_draft pattern). A crash between
        # the steps can no longer leave a published article without a version
        # snapshot or audit trail; events publish only after the commit.
        try:
            created_article = await self.article_repo.create(article, commit=False)
            logger.info(
                "Article created",
                article_id=str(created_article.id),
                status=created_article.status,
                owner_id=str(user.id),
                owner_role=user.role,
                sensitivity=created_article.sensitivity,
            )

            # Save tags
            if tags:
                await self.article_repo.sync_tags(
                    created_article.id, tags, commit=False
                )

            # Save initial version snapshot
            snapshot = {
                "title": created_article.title,
                "body_md": created_article.body_md,
                "original_body_md": original_body_md or created_article.body_md,
                "dept": created_article.dept,
                "domain": created_article.domain,
                "type": created_article.type,
                "sensitivity": created_article.sensitivity,
                "language": created_article.language,
                "tags": tags,
            }
            version = ArticleVersion(
                article_id=created_article.id,
                version=1,
                snapshot=snapshot,
                edited_by=user.id,
            )
            await self.article_repo.create_version(version, commit=False)
            await self._audit(user.id, "create", created_article.id, commit=False)
            if created_article.status == "published":
                await self._audit(user.id, "publish", created_article.id, commit=False)
            await self.article_repo.db.commit()
        except Exception:
            await self.article_repo.db.rollback()
            raise

        # Re-fetch article with tags for the response payload.
        updated_article = await self.article_repo.get_by_id(created_article.id)
        if not updated_article:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve created article"
            )

        # Trigger event if published
        if updated_article.status == "published":
            logger.info(
                "Published article indexing event queued",
                article_id=str(updated_article.id),
            )
            await event_bus.publish(
                "ArticlePublished", {"article_id": str(updated_article.id)}
            )

        return updated_article

    async def update_article(
        self,
        user: User,
        article_id: uuid.UUID,
        title: str | None = None,
        body_md: str | None = None,
        dept: str | None = None,
        domain: str | None = None,
        type_: str | None = None,
        sensitivity: str | None = None,
        language: str | None = None,
        status_: str | None = None,
        tags: list[str] | None = None,
        access_group_ids: list[uuid.UUID] | None = None,
        next_review: datetime | None = None,
    ) -> Article:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        # Authorization check
        if not PermissionService.can_edit_article(user, article):
            raise HTTPException(
                status_code=403, detail="Not authorized to edit this article"
            )

        if sensitivity is not None and sensitivity not in {
            "public",
            "internal",
            "confidential",
            "restricted",
        }:
            raise HTTPException(status_code=422, detail="Invalid article sensitivity")

        # Moving an article can silently turn a personal edit grant into an
        # organization-wide data-placement action.  Allow it only to someone
        # with edit authority over the destination department or higher.
        target_dept = dept if dept is not None else article.dept
        if target_dept != article.dept:
            target_resource = Article(
                company_domain=article.company_domain,
                dept=target_dept,
                owner_id=article.owner_id,
            )
            if not any(
                AuthorizationService.has_permission(
                    user, "article.edit", target_resource, scope
                )
                for scope in ("department", "company", "global")
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to move an article to this department",
                )

        proposed_groups = list(article.access_groups)
        if access_group_ids is not None:
            proposed_groups = list(
                await self.user_repo.get_groups_by_ids(
                    access_group_ids, article.company_domain
                )
            )
            if len({group.id for group in proposed_groups}) != len(
                set(access_group_ids)
            ):
                raise HTTPException(
                    status_code=422, detail="One or more access groups do not exist"
                )
        proposed_sensitivity = (
            sensitivity if sensitivity is not None else article.sensitivity
        )
        if proposed_sensitivity != "public" and not proposed_groups:
            raise HTTPException(
                status_code=422,
                detail="Non-public articles require at least one access group",
            )

        # Track changes for permission recalculation and version incrementing
        permissions_changed = False
        content_changed = False
        published_transition = False

        if sensitivity is not None and sensitivity != article.sensitivity:
            article.sensitivity = sensitivity
            permissions_changed = True

        if language is not None and language != article.language:
            article.language = language
            content_changed = True

        if access_group_ids is not None:
            new_groups = proposed_groups
            # Compare access groups
            old_group_ids = {g.id for g in article.access_groups}
            new_group_ids = {g.id for g in new_groups}
            if old_group_ids != new_group_ids:
                article.access_groups = new_groups
                permissions_changed = True

        if title is not None and title != article.title:
            article.title = title
            content_changed = True

        if body_md is not None and body_md != article.body_md:
            article.body_md = body_md
            content_changed = True

        if dept is not None and dept != article.dept:
            article.dept = dept
            permissions_changed = True
            content_changed = True

        if domain is not None and domain != article.domain:
            article.domain = domain
            content_changed = True

        if type_ is not None and type_ != article.type:
            article.type = type_
            content_changed = True

        if status_ is not None and status_ != article.status:
            # Status transitions are owned by the governance approval flow.
            # This legacy branch allowed a direct publish when wired up and is
            # intentionally rejected here; nothing reachable passes status_.
            raise HTTPException(
                status_code=400,
                detail="Article status changes must be approved through the review workflow",
            )

        if next_review is not None:
            article.next_review = next_review

        # If content changed, bump the version and save version snapshot
        if content_changed:
            article.version += 1

        # Single transaction across article update, tags, version snapshot,
        # and audit entries; events publish only after the commit.
        try:
            updated_article = await self.article_repo.update(article, commit=False)

            if tags is not None:
                await self.article_repo.sync_tags(
                    updated_article.id, tags, commit=False
                )
                # Reload the relationship so the version snapshot records the
                # new tag rows, not the pre-sync collection.
                await self.article_repo.db.refresh(
                    updated_article, attribute_names=["tags"]
                )

            if content_changed:
                tag_strings = [t.tag for t in updated_article.tags]
                snapshot = {
                    "title": updated_article.title,
                    "body_md": updated_article.body_md,
                    "dept": updated_article.dept,
                    "domain": updated_article.domain,
                    "type": updated_article.type,
                    "sensitivity": updated_article.sensitivity,
                    "language": updated_article.language,
                    "tags": tag_strings,
                }
                version = ArticleVersion(
                    article_id=updated_article.id,
                    version=updated_article.version,
                    snapshot=snapshot,
                    edited_by=user.id,
                )
                await self.article_repo.create_version(version, commit=False)

            if permissions_changed:
                await self._audit(
                    user.id, "permission_change", updated_article.id, commit=False
                )
            if content_changed:
                await self._audit(user.id, "update", updated_article.id, commit=False)
                if updated_article.status == "published" and published_transition:
                    await self._audit(
                        user.id, "publish", updated_article.id, commit=False
                    )
            await self.article_repo.db.commit()
        except Exception:
            await self.article_repo.db.rollback()
            raise

        if tags is not None:
            # Re-fetch so the response reflects the persisted tag rows.
            updated_article = await self.article_repo.get_by_id(updated_article.id)

        # Trigger events
        if permissions_changed:
            await event_bus.publish(
                "PermissionChanged", {"article_id": str(updated_article.id)}
            )

        if content_changed:
            if updated_article.status == "published":
                logger.info(
                    "Published article re-indexing event queued",
                    article_id=str(updated_article.id),
                )
                await event_bus.publish(
                    "ArticleUpdated", {"article_id": str(updated_article.id)}
                )

        return updated_article

    async def get_article(self, user: User, article_id: uuid.UUID) -> Article:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        # Department records are the source of truth. Archived/deactivated
        # departments must not remain reachable through an old article ID.
        if not await resolve_active_department(
            self.article_repo.db, article.company_domain, article.dept, required=False
        ):
            raise HTTPException(
                status_code=404,
                detail="Article department is inactive or no longer exists",
            )

        if not PermissionService.can_view_article(user, article):
            raise HTTPException(status_code=403, detail="Access denied")

        return article

    async def soft_delete_article(self, user: User, article_id: uuid.UUID) -> None:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        if not PermissionService.can_delete_article(user, article):
            raise HTTPException(
                status_code=403, detail="Not authorized to delete this article"
            )

        try:
            deleted = await self.article_repo.soft_delete(
                article_id, user=user, commit=False
            )
            if deleted:
                await self._audit(user.id, "delete", article_id, commit=False)
            await self.article_repo.db.commit()
        except Exception:
            await self.article_repo.db.rollback()
            raise
        await event_bus.publish("ArticleDeleted", {"article_id": str(article_id)})

    async def get_history(
        self, user: User, article_id: uuid.UUID
    ) -> Sequence[ArticleVersion]:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        if not PermissionService.can_view_article(user, article):
            raise HTTPException(status_code=403, detail="Access denied")

        return await self.article_repo.get_versions(article_id, user=user)

    async def get_version(
        self, user: User, article_id: uuid.UUID, version_num: int
    ) -> ArticleVersion:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        if not PermissionService.can_view_article(user, article):
            raise HTTPException(status_code=403, detail="Access denied")

        version = await self.article_repo.get_version_by_number(
            article_id, version_num, user=user
        )
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        return version

    async def restore_version(
        self, user: User, article_id: uuid.UUID, version_num: int
    ) -> Article:
        """Restore a historical snapshot as a new active version.

        Historical snapshots are never overwritten. Restoring version 2 of a
        v4 article creates v5 with v2's content and keeps v1-v4 available for
        audit and comparison.
        """
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")
        if not PermissionService.can_edit_article(user, article):
            raise HTTPException(
                status_code=403, detail="Not authorized to restore this article version"
            )

        snapshot_version = await self.article_repo.get_version_by_number(
            article_id, version_num, user=user
        )
        if not snapshot_version:
            raise HTTPException(status_code=404, detail="Version not found")
        if snapshot_version.version == article.version:
            raise HTTPException(
                status_code=409, detail="This version is already active"
            )

        snapshot = snapshot_version.snapshot or {}
        for field in (
            "title",
            "body_md",
            "dept",
            "domain",
            "type",
            "sensitivity",
            "language",
        ):
            if field in snapshot and snapshot[field] is not None:
                setattr(article, field, snapshot[field])

        article.version += 1
        article.lifecycle_status = "active"
        article.index_status = "pending"
        article.index_error = None

        # Single transaction: restored article, tags, new version snapshot,
        # and audit entry commit together; the event publishes after commit.
        try:
            restored_article = await self.article_repo.update(article, commit=False)

            tags = snapshot.get("tags")
            if isinstance(tags, list):
                await self.article_repo.sync_tags(
                    article.id, [str(tag) for tag in tags], commit=False
                )
                await self.article_repo.db.refresh(
                    restored_article, attribute_names=["tags"]
                )

            restored_snapshot = {
                "title": restored_article.title,
                "body_md": restored_article.body_md,
                "dept": restored_article.dept,
                "domain": restored_article.domain,
                "type": restored_article.type,
                "sensitivity": restored_article.sensitivity,
                "language": restored_article.language,
                "tags": [tag.tag for tag in restored_article.tags],
                "restored_from_version": version_num,
            }
            await self.article_repo.create_version(
                ArticleVersion(
                    article_id=restored_article.id,
                    version=restored_article.version,
                    snapshot=restored_snapshot,
                    edited_by=user.id,
                ),
                commit=False,
            )
            await self._audit(user.id, "restore_version", restored_article.id, commit=False)
            await self.article_repo.db.commit()
        except Exception:
            await self.article_repo.db.rollback()
            raise

        if restored_article.status == "published":
            await event_bus.publish(
                "ArticleUpdated", {"article_id": str(restored_article.id)}
            )
        return restored_article
