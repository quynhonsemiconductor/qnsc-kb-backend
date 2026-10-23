"""One agentic answer: the object `AIService.ask` drives, step by step.

`ask` stays the one place that owns permissions, context selection, the prompt, citations and
the refusal rules. The agent only changes three things inside it: HOW the candidate passages
are found (plan, per-sub-question search, sufficiency check), WHAT extra material is attached
(a chart, a marked figure), and the final check of the numbers. Everything between is the
existing pipeline, untouched.

`AgentRun.start` returns None when the agent is off for this user, and `ask` then runs exactly
the pipeline it always has.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.core.config import settings
from src.domain.agent.core import Budget, EventSink, Tracer
from src.domain.agent.plan import (
    AgentPlan,
    allowed_department_names,
    build_plan,
    fallback_plan,
    is_simple_question,
)
from src.domain.agent.retrieve import RetrievalOutcome, agentic_retrieve
from src.domain.agent.tools import annotate_figure, build_chart
from src.domain.agent.verify import find_unverified_numbers
from src.models.user import User

logger = structlog.get_logger()


class AgentRun:
    def __init__(
        self, tracer: Tracer, budget: Budget, plan: AgentPlan, *, fast: bool
    ) -> None:
        self.tracer = tracer
        self.budget = budget
        self.plan = plan
        #: True when the question was simple enough to skip planning.
        self.fast = fast
        self.artifacts: list[dict[str, Any]] = []
        self.outcome: RetrievalOutcome | None = None

    @classmethod
    async def start(
        cls,
        *,
        db: Any,
        feature_flags: Any,
        user: User,
        question: str,
        history_text: str,
        has_history: bool,
        on_event: EventSink | None,
    ) -> "AgentRun | None":
        if not settings.AGENT_ENABLED:
            return None
        if feature_flags is not None and not await feature_flags.is_enabled(
            "ai.agent", user
        ):
            return None
        tracer = Tracer(on_event)
        budget = Budget(settings.AGENT_MAX_LLM_CALLS)
        if settings.AGENT_FAST_PATH_ENABLED and is_simple_question(
            question, has_history=has_history
        ):
            await tracer.emit("understand", "skipped", detail="fast")
            return cls(tracer, budget, fallback_plan(question), fast=True)
        await tracer.emit("understand", "running")
        try:
            departments = await allowed_department_names(db, user)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - the department list only narrows the plan
            logger.warning("Agent could not list departments", error=str(exc))
            departments = []
        plan = await build_plan(question, history_text, departments, budget)
        await tracer.emit(
            "understand",
            "done",
            detail=plan.summary or None,
            intent=plan.intent,
            sub_queries=plan.sub_queries,
            doc_types=plan.doc_types,
            departments=plan.departments,
        )
        return cls(tracer, budget, plan, fast=False)

    # ── clarification ──────────────────────────────────────────────────────────────────────

    def clarification(
        self, conversation_messages: list[Any]
    ) -> tuple[str, list[str]] | None:
        """The question to ask back, or None.

        Asked at most once in a row: if the assistant's last message already ended in a
        question, the user is answering it, and asking again would loop them.
        """
        plan = self.plan
        if not (
            settings.AGENT_CLARIFICATION_ENABLED
            and plan.needs_clarification
            and plan.clarification_question
        ):
            return None
        for message in reversed(conversation_messages):
            if getattr(message, "role", "") == "assistant":
                if str(getattr(message, "content", "")).rstrip().endswith("?"):
                    return None
                break
        return plan.clarification_question, plan.clarification_options

    # ── retrieval ──────────────────────────────────────────────────────────────────────────

    async def retrieve(
        self, search: Any, user: User, question: str
    ) -> RetrievalOutcome:
        self.outcome = await agentic_retrieve(
            search, user, question, self.plan, tracer=self.tracer, budget=self.budget
        )
        return self.outcome

    # ── tools ──────────────────────────────────────────────────────────────────────────────

    async def attach_artifacts(
        self, db: Any, user: User, question: str, context_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.plan.wants_chart:
            chart = await build_chart(
                question, context_results, tracer=self.tracer, budget=self.budget
            )
            if chart:
                self.artifacts.append(chart)
        if self.plan.wants_figure:
            figure = await annotate_figure(
                db,
                user,
                question,
                context_results,
                tracer=self.tracer,
                budget=self.budget,
            )
            if figure:
                self.artifacts.append(figure)
        return self.artifacts

    def artifact_notes(self) -> str:
        """Trusted lines telling the answer model what will sit beside its answer."""
        lines = []
        for artifact in self.artifacts:
            if artifact["type"] == "chart":
                lines.append(
                    f"<attached-chart source=\"{artifact['source_id']}\">A {artifact['chart_type']} chart titled "
                    f"\"{artifact['title']}\" is shown to the reader under your answer. Refer to it; do not redraw it.</attached-chart>"
                )
            elif artifact["type"] == "figure_annotation":
                lines.append(
                    f"<attached-figure source=\"{artifact['source_id']}\">The relevant part of the figure is marked on it "
                    "and shown to the reader under your answer. Refer to it.</attached-figure>"
                )
        return "\n".join(lines)

    # ── verification ───────────────────────────────────────────────────────────────────────

    async def verify_numbers(
        self,
        answer: str,
        context_results: list[dict[str, Any]],
        analytics_results: list[dict[str, Any]],
    ) -> list[str]:
        if not settings.AGENT_NUMBER_VERIFICATION_ENABLED:
            return []
        await self.tracer.emit("verify", "running")
        try:
            unverified = find_unverified_numbers(
                answer, context_results, analytics_results, self.artifacts
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - verification must never cost the user the answer
            logger.warning(
                "Number verification failed; continuing without it", error=str(exc)
            )
            unverified = []
        await self.tracer.emit("verify", "done", unverified=len(unverified))
        return unverified
