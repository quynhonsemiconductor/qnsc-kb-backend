"""The pinned MCP tool budget (MCP-PLAN §11: 15-30 task-shaped tools per app).

Nothing else fails when a tool is added, so this is the one place the number is stated.
Adding a tool means retiring one, or updating MCP-PLAN with a deliberate decision.
"""

from __future__ import annotations

import json

from src.domain import mcp_tools

READ_TOOL_BUDGET = 21
WRITE_TOOL_BUDGET = 6
TOTAL_TOOL_BUDGET = READ_TOOL_BUDGET + WRITE_TOOL_BUDGET

#: A definition is sent to the model on `tools/list`; past this it costs more context than
#: the tool is worth. It is a ceiling, not an exact size.
MAX_DEFINITION_CHARS = 3000


def test_the_catalogue_is_pinned_to_the_budget():
    reads = [tool for tool in mcp_tools.TOOLS if not tool.write]
    writes = [tool for tool in mcp_tools.TOOLS if tool.write]

    assert len(mcp_tools.TOOLS) == TOTAL_TOOL_BUDGET, (
        f"The MCP tool budget is {TOTAL_TOOL_BUDGET} ({READ_TOOL_BUDGET} read + "
        f"{WRITE_TOOL_BUDGET} write; MCP-PLAN §11). Adding a tool means retiring one."
    )
    assert len(reads) == READ_TOOL_BUDGET
    assert len(writes) == WRITE_TOOL_BUDGET
    assert 15 <= TOTAL_TOOL_BUDGET <= 30, "MCP-PLAN §11: 15-30 tools per app"


def test_every_definition_stays_within_the_context_budget():
    for tool in mcp_tools.TOOLS:
        size = len(json.dumps(tool.definition(), ensure_ascii=False))
        assert size <= MAX_DEFINITION_CHARS, f"{tool.name} definition is {size} chars"
        assert tool.definition()["annotations"]["readOnlyHint"] is (not tool.write)
