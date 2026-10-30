"""The team configuration an AI client copies into its CLAUDE.md, and how it learns it changed.

The server owns the text and its version; the client compares the version it holds with the
one named in the MCP instructions, and asks for the text only when they differ.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from src.api.routers import mcp as mcp_router
from src.domain import mcp_tools
from src.domain.mcp_config import (
    APPLY,
    can_edit_config,
    config_payload,
    digest,
    instructions_note,
    render_block,
)
from src.domain.mcp_tools import Call


def _row(version=3, content="Always check the KB first."):
    return SimpleNamespace(
        version=version, content_md=content, updated_at=datetime(2026, 10, 30)
    )


def test_the_block_is_fenced_by_markers_that_carry_the_version():
    block = render_block(3, "Rule one.\nRule two.")

    lines = block.splitlines()
    assert lines[0].startswith("<!-- kb-config:start version=3 sha=")
    assert lines[1:3] == ["Rule one.", "Rule two."]
    assert lines[-1] == "<!-- kb-config:end -->"


def test_the_digest_in_the_marker_changes_with_the_text():
    assert render_block(1, "a") != render_block(1, "b")
    assert digest("a")[:12] in render_block(1, "a")


def test_nothing_configured_is_reported_as_version_zero_with_no_block():
    assert config_payload(None) == {
        "configured": False,
        "version": 0,
        "content_md": "",
        "block": "",
    }


def test_a_configured_payload_carries_the_text_the_version_and_the_block():
    payload = config_payload(_row())

    assert payload["configured"] and payload["version"] == 3
    assert payload["content_md"] == "Always check the KB first."
    assert payload["block"] == render_block(3, "Always check the KB first.")


def test_the_tool_is_visible_to_a_read_only_token():
    names = [tool.name for tool in mcp_tools.visible_tools(can_write=False)]

    assert "get_project_config" in names


def test_the_tool_asks_for_the_config_and_passes_the_version_it_holds():
    tool = mcp_tools.TOOLS_BY_NAME["get_project_config"]

    assert tool.build({}) == Call("GET", "/mcp-config", params=None)
    assert tool.build({"known_version": 4}) == Call(
        "GET", "/mcp-config", params={"known_version": 4}
    )
    assert not tool.write


def test_an_up_to_date_client_is_told_to_change_nothing():
    shaped = mcp_tools.TOOLS_BY_NAME["get_project_config"].shape(
        {"up_to_date": True, "configured": True, "version": 3}
    )

    assert shaped["up_to_date"] is True and "block" not in shaped


def test_a_stale_client_gets_the_block_and_how_to_apply_it_without_the_raw_text_twice():
    shaped = mcp_tools.TOOLS_BY_NAME["get_project_config"].shape(config_payload(_row()))

    assert shaped == {
        "version": 3,
        "block": render_block(3, "Always check the KB first."),
        "how_to_apply": APPLY,
    }


def test_an_unconfigured_server_has_nothing_to_apply():
    shaped = mcp_tools.TOOLS_BY_NAME["get_project_config"].shape(config_payload(None))

    assert shaped["configured"] is False and "block" not in shaped


def test_the_instructions_name_the_version_to_look_for():
    note = instructions_note(7)

    assert "version 7" in note and "<!-- kb-config:start version=7 " in note
    assert "get_project_config" in note


class _Ctx:
    def __init__(self, status, payload):
        self.status, self.payload, self.calls = status, payload, []

    async def call(self, call):
        self.calls.append(call)
        return self.status, self.payload


def _note(ctx):
    return asyncio.run(mcp_router._config_note(ctx))


def test_a_connecting_client_is_told_the_live_version():
    ctx = _Ctx(200, config_payload(_row(version=5)))

    assert _note(ctx) == instructions_note(5)
    assert ctx.calls == [Call("GET", "/mcp-config")]


def test_no_configuration_adds_nothing_to_the_instructions():
    assert _note(_Ctx(200, config_payload(None))) == ""


def test_a_failing_lookup_never_breaks_connecting():
    assert _note(_Ctx(500, {"detail": "boom"})) == ""

    class _Broken:
        async def call(self, call):
            raise RuntimeError("db down")

    assert _note(_Broken()) == ""


def test_only_an_admin_or_ceo_can_edit_the_configuration():
    assert can_edit_config(SimpleNamespace(role="Admin", roles=[]))
    assert can_edit_config(SimpleNamespace(role="CEO", roles=[]))
    assert not can_edit_config(SimpleNamespace(role="Manager", roles=[]))
    assert not can_edit_config(SimpleNamespace(role="Employee", roles=[]))
