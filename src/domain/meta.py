from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

logger = structlog.get_logger()
from src.models.article import Article, ArticleTag, TagCatalog
from src.models.user import User
from src.repositories.article import ArticleRepository

class MetaService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_tags(self, user: User) -> list[str]:
        stmt = select(TagCatalog.tag).where(TagCatalog.company_domain == user.company_domain, TagCatalog.active.is_(True)).order_by(TagCatalog.normalized_tag)
        tags = list((await self.db.execute(stmt)).scalars().all())
        logger.info("Meta tags loaded", tag_count=len(tags))
        return tags

    async def get_glossary(self) -> list[dict[str, str]]:
        # Glossary content is customer-owned.  Until a customer-loaded glossary
        # table is configured, return an empty catalogue rather than demo terms.
        logger.info("Meta glossary loaded", glossary_item_count=0)
        return []
