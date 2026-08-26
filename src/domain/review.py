import uuid
from datetime import datetime, timedelta
from sqlalchemy import select
from fastapi import HTTPException
from src.models.user import User
from src.models.article import Article
from src.repositories.article import ArticleRepository
from src.domain.permissions import PermissionService
from src.domain.events import event_bus

class ReviewService:
    def __init__(self, article_repo: ArticleRepository):
        self.article_repo = article_repo

    async def schedule_review(
        self,
        user: User,
        article_id: uuid.UUID,
        next_review_date: datetime
    ) -> Article:
        article = await self.article_repo.get_by_id(article_id, user=user)
        if not article or article.status == "deleted":
            raise HTTPException(status_code=404, detail="Article not found")

        # Must be Owner, Dept Owner, or Admin
        if not PermissionService.can_edit_article(user, article):
            raise HTTPException(status_code=403, detail="Not authorized to set review schedule")

        article.next_review = next_review_date
        article.needs_update = False
        updated_article = await self.article_repo.update(article)
        
        await event_bus.publish("ReviewScheduled", {"article_id": str(article_id), "next_review": next_review_date.isoformat()})
        return updated_article

    async def verify_review_deadlines(self, company_domain: str | None = None) -> list[str]:
        """
        Scans articles list for overdue reviews and fires ReviewExpired events.
        """
        # For simplicity, we can load active articles through repository
        # In a real setup, this is run by a Celery Beat task
        # We can implement a simple scan and return list of flagged IDs
        now = datetime.utcnow()
        filters = [Article.status == "published", Article.lifecycle_status == "active", Article.next_review < now]
        if company_domain:
            filters.append(Article.company_domain == company_domain)
        result = await self.article_repo.db.execute(select(Article).where(*filters).limit(1000))
        articles = result.scalars().all()
        
        overdue_ids = []
        for article in articles:
            overdue_ids.append(str(article.id))
            if not article.needs_update:
                article.needs_update = True
            await event_bus.publish("ReviewExpired", {"article_id": str(article.id)})
        if articles:
            await self.article_repo.db.commit()
                
        return overdue_ids
