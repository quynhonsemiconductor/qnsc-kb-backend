"""The read-only section and source tools added for the Access/MCP surface."""

from __future__ import annotations

import json

import pytest

from src.domain import mcp_tools

ARTICLE_ID = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"

BODY = """# VPN

Introduction.

## Setup

Install the client.

```
## not a heading inside a fence
```

## Usage

Connect and sign in.
"""


def call(tool: str, arguments: dict, payload) -> dict:
    result = mcp_tools.format_result(
        mcp_tools.TOOLS_BY_NAME[tool], 200, payload, arguments
    )
    assert result["isError"] is False
    return json.loads(result["content"][0]["text"])


def article() -> dict:
    return {
        "id": ARTICLE_ID,
        "title": "VPN",
        "version": 3,
        "body_md": BODY,
        "owner_id": "secret-owner",
    }


def test_naming_no_section_returns_the_table_of_contents():
    data = call("get_article_section", {"article_id": ARTICLE_ID}, article())
    assert data["section_count"] == 2
    assert [item["heading"] for item in data["sections"]] == ["Setup", "Usage"]
    assert "body_md" not in data
    assert "secret-owner" not in json.dumps(data)


def test_a_heading_substring_reads_only_that_section():
    data = call(
        "get_article_section",
        {"article_id": ARTICLE_ID, "heading": "set"},
        article(),
    )
    assert data["heading"] == "Setup" and data["index"] == 1
    assert "Install the client." in data["body_md"]
    assert "Connect and sign in." not in data["body_md"]
    assert "not a heading inside a fence" in data["body_md"]


def test_a_one_based_index_reads_that_section():
    data = call(
        "get_article_section", {"article_id": ARTICLE_ID, "index": 2}, article()
    )
    assert data["heading"] == "Usage"
    assert "Connect and sign in." in data["body_md"]


def test_an_unknown_section_lists_what_exists_instead_of_failing():
    data = call(
        "get_article_section",
        {"article_id": ARTICLE_ID, "heading": "billing"},
        article(),
    )
    assert "No section matches" in data["message"]
    assert [item["heading"] for item in data["sections"]] == ["Setup", "Usage"]


def test_an_article_without_section_headings_is_one_section():
    payload = {"id": ARTICLE_ID, "title": "Note", "version": 1, "body_md": "Just text."}
    data = call("get_article_section", {"article_id": ARTICLE_ID, "index": 1}, payload)
    assert data["section_count"] == 1 and data["body_md"] == "Just text."


def test_section_tool_validates_its_id_and_heading_types():
    tool = mcp_tools.TOOLS_BY_NAME["get_article_section"]
    for arguments in (
        {},
        {"article_id": "not-a-uuid"},
        {"article_id": ARTICLE_ID, "heading": 5},
        {"article_id": ARTICLE_ID, "index": "two"},
    ):
        with pytest.raises(mcp_tools.ToolInputError):
            tool.build(arguments)


def test_sources_are_slim_and_keep_paging_metadata():
    payload = {
        "total": 1,
        "limit": 20,
        "offset": 0,
        "sources": [
            {
                "id": "s1",
                "article_id": ARTICLE_ID,
                "article_title": "VPN",
                "source_system": "upload",
                "source_ref": "vpn.pdf",
                "filename": "vpn.pdf",
                "mime_type": "application/pdf",
                "ingested_at": "2026-11-01T00:00:00Z",
                "has_file": True,
                "source_changed": False,
                "storage_key": "secret",
                "raw_text": "internal",
            }
        ],
    }
    data = call("list_sources", {}, payload)
    assert data["total"] == 1 and data["limit"] == 20
    (source,) = data["sources"]
    assert source["filename"] == "vpn.pdf" and source["article_title"] == "VPN"
    assert "storage_key" not in source and "raw_text" not in source


def test_the_source_builder_forwards_filters_and_clamps_the_limit():
    tool = mcp_tools.TOOLS_BY_NAME["list_sources"]
    built = tool.build(
        {"q": "vpn", "source_system": "upload", "limit": 500, "offset": 10}
    )
    assert (built.method, built.path) == ("GET", "/knowledge/sources")
    assert built.params == {
        "q": "vpn",
        "source_system": "upload",
        "limit": 50,
        "offset": 10,
    }


def test_a_denied_article_does_not_leak_title_or_body():
    tool = mcp_tools.TOOLS_BY_NAME["get_article"]
    result = mcp_tools.format_result(
        tool,
        403,
        {
            "detail": "Access denied",
            "title": "Secret acquisition plan",
            "body_md": "the body nobody may read",
        },
        {"article_id": ARTICLE_ID},
    )
    text = result["content"][0]["text"]
    assert result["isError"] is True and "Access denied" in text
    assert "Secret acquisition plan" not in text and "nobody may read" not in text


def test_a_denied_source_does_not_leak_filename_or_reference():
    tool = mcp_tools.TOOLS_BY_NAME["list_sources"]
    result = mcp_tools.format_result(
        tool,
        403,
        {
            "detail": "Access denied",
            "sources": [
                {"filename": "redundancy-plan-2026.pdf", "source_ref": "secret/ref"}
            ],
        },
        {},
    )
    text = result["content"][0]["text"]
    assert result["isError"] is True and "Access denied" in text
    assert "redundancy-plan-2026" not in text and "secret/ref" not in text


def test_a_denied_section_does_not_leak_section_text():
    tool = mcp_tools.TOOLS_BY_NAME["get_article_section"]
    result = mcp_tools.format_result(
        tool,
        403,
        {"detail": "Access denied", "body_md": "section text nobody may read"},
        {"article_id": ARTICLE_ID, "heading": "Setup"},
    )
    text = result["content"][0]["text"]
    assert result["isError"] is True and "Access denied" in text
    assert "nobody may read" not in text
