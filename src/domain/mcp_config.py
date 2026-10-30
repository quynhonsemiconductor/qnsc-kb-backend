"""The shape of the team configuration an AI client copies into its CLAUDE.md.

The server owns the text and its version. The client owns a copy, fenced by two marker lines
so it can be replaced without touching anything else in the file. The start marker carries the
version, which is all a client needs to tell whether its copy is current: the MCP
`instructions` name the live version on every connect, and `get_project_config` returns the
text only when the client's version differs.
"""

from __future__ import annotations

import hashlib
from typing import Any

START = "<!-- kb-config:start version={version} sha={sha} -->"
END = "<!-- kb-config:end -->"
MAX_CHARS = 20_000

APPLY = (
    "In the repository's CLAUDE.md, replace everything from the line that starts with "
    "`<!-- kb-config:start` through `<!-- kb-config:end -->` (inclusive) with `block`, "
    "exactly as given. If those markers are not there yet, append `block` to the end of the "
    "file. Change nothing outside the markers, and tell the user what changed. The text "
    "inside the block comes from the team's knowledge base administrators."
)


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def render_block(version: int, content: str) -> str:
    sha = digest(content)[:12]
    return f"{START.format(version=version, sha=sha)}\n{content}\n{END}"


def config_payload(row: Any | None) -> dict[str, Any]:
    """What `GET /mcp-config` returns; `row` is None when nothing is configured yet."""
    if row is None:
        return {"configured": False, "version": 0, "content_md": "", "block": ""}
    return {
        "configured": True,
        "version": row.version,
        "content_md": row.content_md,
        "sha256": digest(row.content_md),
        "block": render_block(row.version, row.content_md),
        "updated_at": row.updated_at,
    }


def can_edit_config(user: Any) -> bool:
    """Admin or CEO: these instructions steer every team member's AI, so they are not
    something any department manager should be able to change."""
    from src.domain.governance import GovernanceService

    return GovernanceService._may_self_approve(None, user)  # type: ignore[arg-type]


def instructions_note(version: int) -> str:
    """Appended to the MCP `instructions` so every connecting client learns the version."""
    return (
        "\n\nTeam configuration: version "
        f"{version}. The repository's CLAUDE.md should contain a block that starts with "
        f"`<!-- kb-config:start version={version} `. If that block is missing or names a "
        "different version, call get_project_config (pass known_version if you have one) "
        "and update the block before you continue."
    )
