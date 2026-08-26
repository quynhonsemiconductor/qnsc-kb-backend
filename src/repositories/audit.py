import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.governance import AuditLog


class AuditRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def record(self, user_id: uuid.UUID | None, action: str, target_type: str, target_id: str, outcome: str = "success", *, detail: dict | None = None, commit: bool = True) -> AuditLog:
        entry = AuditLog(user_id=user_id, action=action, target_type=target_type, target_id=target_id, outcome=outcome, detail_json=detail)
        self.db.add(entry)
        if commit:
            await self.db.commit()
            await self.db.refresh(entry)
        else:
            await self.db.flush()
        return entry
