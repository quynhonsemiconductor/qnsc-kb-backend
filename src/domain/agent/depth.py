"""How hard an answer looks: quick, standard or thorough.

RAGFlow lets the asker choose between thinking modes. Here the choice only moves two limits
the agent already has: how many extra retrieval rounds it may run when a sub-question is not
covered, and how many model calls it may spend. Carried in a context variable so the choice
set at the request reaches the agent without threading a parameter through every layer; the
default is `standard`, the behaviour before this existed.
"""

from __future__ import annotations

from contextvars import ContextVar

DEPTHS = ("quick", "standard", "thorough")

current_depth: ContextVar[str] = ContextVar("retrieval_depth", default="standard")


def retrieval_rounds(base: int) -> int:
    depth = current_depth.get()
    if depth == "quick":
        return 0
    if depth == "thorough":
        return base + 2
    return base


def llm_calls(base: int) -> int:
    depth = current_depth.get()
    if depth == "quick":
        return max(3, base // 2)
    if depth == "thorough":
        return base + 4
    return base
