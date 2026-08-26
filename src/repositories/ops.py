import uuid
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models.ops import Connector, ConnectorJob, NotificationQueue, DeadLetterJob, EvalQuestion, EvalRun

class OpsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # Connectors
    async def create_connector(self, conn: Connector) -> Connector:
        self.db.add(conn)
        await self.db.commit()
        await self.db.refresh(conn)
        return conn

    async def get_connector(self, conn_id: uuid.UUID) -> Connector | None:
        result = await self.db.execute(
            select(Connector).where(Connector.id == conn_id)
        )
        return result.scalar_one_or_none()

    async def list_connectors(self) -> Sequence[Connector]:
        result = await self.db.execute(select(Connector).order_by(Connector.created_at.desc()))
        return result.scalars().all()

    # Jobs
    async def create_job(self, job: ConnectorJob) -> ConnectorJob:
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def get_job(self, job_id: uuid.UUID) -> ConnectorJob | None:
        result = await self.db.execute(
            select(ConnectorJob).where(ConnectorJob.id == job_id).options(selectinload(ConnectorJob.connector))
        )
        return result.scalar_one_or_none()

    async def update_job(self, job: ConnectorJob) -> ConnectorJob:
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    # DLQ / Notifications
    async def log_dead_letter(self, dlj: DeadLetterJob) -> DeadLetterJob:
        self.db.add(dlj)
        await self.db.commit()
        await self.db.refresh(dlj)
        return dlj

    async def queue_notification(self, nq: NotificationQueue) -> NotificationQueue:
        self.db.add(nq)
        await self.db.commit()
        await self.db.refresh(nq)
        return nq

    # Evaluation
    async def create_eval_question(self, eq: EvalQuestion) -> EvalQuestion:
        self.db.add(eq)
        await self.db.commit()
        await self.db.refresh(eq)
        return eq

    async def list_eval_questions(self) -> Sequence[EvalQuestion]:
        result = await self.db.execute(select(EvalQuestion).limit(500))
        return result.scalars().all()

    async def get_eval_question(self, question_id: uuid.UUID) -> EvalQuestion | None:
        result = await self.db.execute(select(EvalQuestion).where(EvalQuestion.id == question_id))
        return result.scalar_one_or_none()

    async def create_eval_run(self, er: EvalRun) -> EvalRun:
        self.db.add(er)
        await self.db.commit()
        await self.db.refresh(er)
        return er

    async def list_eval_runs(self) -> Sequence[EvalRun]:
        result = await self.db.execute(
            select(EvalRun)
            .order_by(EvalRun.created_at.desc())
            .options(selectinload(EvalRun.question))
            .limit(200)
        )
        return result.scalars().all()
