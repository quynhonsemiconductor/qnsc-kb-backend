# MCP server

The knowledge base speaks the Model Context Protocol, so Claude Desktop, Claude Code,
Cursor and other MCP clients can search, ask and read it, and propose changes.

- Endpoint: `POST /api/v1/mcp` (Streamable HTTP, stateless, JSON responses).
- Auth: a personal access token, `Authorization: Bearer kbpat_...`.
- Protocol versions: 2025-06-18, 2025-03-26, 2024-11-05.
- Switch: `MCP_ENABLED` (default `true`).

## Getting a token

Account menu → Access tokens → Create. The token is shown once. Choose `read` (search,
ask, read) or `write` (also submit/propose articles and edits). A token acts as its owner:
every call is authorised by that user's own roles, departments and ACLs at the time of the
call, so a token can never see more than its owner. Tokens expire (default 90 days, at most
`MCP_TOKEN_MAX_DAYS`), can be revoked at any time, and stop working if the owner is
deactivated. Only a SHA-256 hash is stored.

## Connecting a client

Claude Code:

```bash
claude mcp add --transport http qnsc-kb https://<api-host>/api/v1/mcp \
  --header "Authorization: Bearer kbpat_..."
```

Clients that only speak stdio can use `mcp-remote`:

```json
{
  "mcpServers": {
    "qnsc-kb": {
      "command": "npx",
      "args": ["mcp-remote", "https://<api-host>/api/v1/mcp",
               "--header", "Authorization: Bearer kbpat_..."]
    }
  }
}
```

## What it offers

Read tools: `search_knowledge`, `ask_knowledge_base`, `get_article`, `list_articles`,
`get_related_articles`, `list_article_versions`, `get_article_version`,
`list_departments`, `list_tags`, `get_glossary`, `find_entities`, `get_entity_neighbors`.

Write tools (write-scope tokens only): `create_article`, `update_article`,
`request_article_edit`, `rate_answer`. Articles go through the normal review workflow.

Resources: `kb://article/{id}` (Markdown). Prompts: `summarize_article`,
`answer_with_sources`, `compare_articles`, `find_knowledge_gaps`.

Deliberately not offered: deleting or restoring articles, approving drafts, reindexing,
connectors, user and role administration.

## How it is built

Each tool maps onto an existing REST endpoint (`src/domain/mcp_tools.py`). The transport
(`src/api/routers/mcp.py`) mints the token owner a five-minute access JWT and calls the
application in-process, so the endpoint enforces exactly what it enforces for the web app:
RBAC, ACLs, row-level security, the per-user AI rate limit and audit. There is no second
implementation of "who may see this". A test checks that every tool targets a route that
exists. Requests are also limited per token (`MCP_RATE_LIMIT_PER_MINUTE`).

## Operating notes

- The route is under `/api/v1`, so any ingress rule that already forwards the API forwards
  MCP. Streaming is not used, so no proxy buffering settings are needed.
- Browser requests with a foreign `Origin` are refused (DNS-rebinding defence); MCP
  clients send none.
- Changing the owner's password does not revoke tokens; revoke them from the token list.
