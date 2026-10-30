import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ops import FeatureFlag
from src.models.user import User


class FeatureFlagRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list(self):
        result = await self.db.execute(select(FeatureFlag).order_by(FeatureFlag.key))
        return result.scalars().all()

    async def get(self, key: str) -> FeatureFlag | None:
        result = await self.db.execute(
            select(FeatureFlag).where(FeatureFlag.key == key)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        key: str,
        enabled: bool,
        rollout_percent: int,
        role: str | None,
        department: str | None,
    ) -> FeatureFlag:
        flag = await self.get(key)
        if flag is None:
            flag = FeatureFlag(key=key)
            self.db.add(flag)
        flag.enabled = enabled
        flag.rollout_percent = rollout_percent
        flag.role = role
        flag.department = department
        await self.db.commit()
        await self.db.refresh(flag)
        return flag

    async def is_opted_in(self, key: str, user: User) -> bool:
        """For a feature that is OFF until someone switches it on.

        `is_enabled` treats a flag nobody created as on, which is right for a feature that
        ships on. A feature that must stay off by default needs the opposite: no flag, no
        feature; and an existing flag still follows its role, department and rollout rules.
        """
        flag = await self.get(key)
        if flag is None or not flag.enabled:
            return False
        return await self.is_enabled(key, user)

    async def is_enabled(self, key: str, user: User) -> bool:
        flag = await self.get(key)
        if flag is None:
            return True
        if not flag.enabled:
            return False
        if flag.role and flag.role != user.role:
            return False
        if flag.department and flag.department != user.dept:
            return False
        digest = hashlib.sha256(str(user.id).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        return bucket < max(0, min(100, flag.rollout_percent))
