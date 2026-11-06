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


#: The one read tool whose REST endpoint is a POST: ask records a conversation/log, so it
#: is deliberately not offered to a strictly read-only Access principal.
NON_GET_READ_TOOL = "ask_knowledge_base"

SAMPLE = {
    "query": "q",
    "question": "what is x?",
    "article_id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
    "conversation_id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
    "entity_id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
    "version": 1,
    "from_version": 1,
    "to_version": 2,
    "heading": "Setup",
    "index": 1,
}


def sample_call(tool: mcp_tools.Tool) -> mcp_tools.Call:
    return tool.build(
        {key: value for key, value in SAMPLE.items() if key in tool.properties}
    )


def test_every_read_tool_is_get_backed_so_access_can_see_only_get(monkeypatch):
    monkeypatch.setattr(mcp_tools.settings, "KNOWLEDGE_GRAPH_ENABLED", True)
    reads = [tool for tool in mcp_tools.TOOLS if not tool.write]
    non_get = {tool.name for tool in reads if sample_call(tool).method != "GET"}
    assert non_get == {NON_GET_READ_TOOL}, non_get
    assert mcp_tools.TOOLS_BY_NAME[NON_GET_READ_TOOL].get_only is False

    get_backed = {tool.name for tool in reads if sample_call(tool).method == "GET"}
    access_visible = {tool.name for tool in mcp_tools.visible_tools(False, access=True)}
    assert access_visible == get_backed
    assert NON_GET_READ_TOOL not in access_visible
