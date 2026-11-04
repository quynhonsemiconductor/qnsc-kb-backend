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

## Your first minutes

You do not need to learn the tool names: the server tells the client how to use itself
(`instructions`, tool descriptions, prompts), so just ask in plain language.

- "What is our policy on remote work?" runs `ask_knowledge_base` and returns an answer
  with citations (article, page).
- "Find where the boot sequence is documented and quote it" runs `search_knowledge`,
  then `get_article`.
- "What documents do we have about onboarding?" runs `list_articles`.
- In Claude Code, type `/` and pick a prompt from `qnsc-kb`. Start with
  `explore_knowledge_base`: it tours what is in the knowledge base and suggests
  questions.
- `/mcp` in Claude Code shows the server, its tools and whether it is connected.

If a document you expect is missing, you do not have access to it: results always follow
your own permissions.

## What it offers

Read tools: `search_knowledge`, `ask_knowledge_base`, `get_article`, `list_articles`,
`get_related_articles`, `list_article_versions`, `get_article_version`,
`get_article_changes`, `list_conversations`, `get_conversation`, `list_lessons`, `list_my_drafts`, `list_departments`,
`list_tags`, `get_glossary`, `find_entities`, `get_entity_neighbors`.

Write tools (write-scope tokens only): `create_article`, `update_article`,
`request_article_edit`, `rate_answer`, `record_lesson`, `forget_lesson`. Articles go through the
normal review workflow; lessons do not (below).

`find_entities` and `get_entity_neighbors` appear only when `KNOWLEDGE_GRAPH_ENABLED`
is on; without it the tools are hidden.

## Working with it over many sessions

- **Long documents come in pages.** `get_article` returns about 30 000 characters at a time;
  when the result has `next_offset`, call again with `offset=<next_offset>` until it has none.
  `body_chars` is the full length.
- **"What changed?" has its own tool.** `get_article_changes` compares two versions of an article
  and returns a short summary, then per section the lines removed and added, plus changes to
  title, department, tags and sensitivity; by default the latest against the one before it. Pass
  `from_version` (the `version` returned by `get_article` when it was read) to learn what changed
  since then. Whitespace-only edits are ignored and long changes are clipped (`truncated`). The
  caller needs the same right to read the article as for `list_article_versions`.
- **Past questions persist.** Every `ask_knowledge_base` runs inside a conversation stored on
  the server. `list_conversations` (optionally filtered by title) and `get_conversation`
  give an assistant back the user's own earlier questions and answers after a restart, and
  the `conversation_id` can be passed to `ask_knowledge_base` to carry on. They only ever show
  the token owner's own conversations.
- **What the team should remember goes in as a lesson, and does not wait for a reviewer.**
  Reviewed documents (uploads, `create_article`, `update_article`) keep their review workflow.
  What an assistant learns in a session is a different kind of knowledge: `record_lesson`
  publishes it **at once** for the author's department, as an article of type `LESSON`, so
  `search_knowledge` and `ask_knowledge_base` find it straight away (a colleague's assistant
  found one 9 s after it was saved). A lesson needs a department: pass `dept` unless the user
  has one set. `basis` is required (`said_by_user`, `confirmed_by_user`, `tested_in_session`): a lesson
  records what the person said, confirmed or tested with the assistant, not conclusions the assistant drew alone from
  reading documents, and not something the person did not agree to. What keeps this safe:
  - **Reviewed and unreviewed stay distinguishable.** A search hit has `type: LESSON`; an answer
    that relies on a lesson says it is an unreviewed team lesson, and when a reviewed document
    and a lesson disagree it presents both.
  - **The assistant can choose.** `search_knowledge` takes `scope` (`all`, `documents`,
    `lessons`) and `ask_knowledge_base` takes `sources` (`all`, `documents`); the web assistant
    sends the same `sources`. Left out, `AI_INCLUDE_LESSONS_DEFAULT` decides for answers
    (default on) and search covers both.
  - **A lesson is never duplicated or silently replaced.** `record_lesson` first compares it with
    what is stored: new -> `add`; adds to or corrects an earlier LESSON -> `update`, merged into
    it in place (a new version and an audit row, the newer fact wins); concerns a reviewed
    DOCUMENT -> `update` but nothing saved, use `request_article_edit`; already known -> `skip`.
  - **Anyone may retract their own.** `list_lessons` (optionally `mine`) shows what was saved,
    `forget_lesson` takes one out of search at once; the author may always do it, so may whoever
    may delete the department's articles. It is a POST of one lesson, not a delete tool.
  - **A quota** (`LESSONS_MAX_PER_USER_PER_DAY`, 50 saves or changes a day) stops a runaway assistant. A lesson also cannot grow past 20 000 characters; past that a new lesson is saved on its own.
  `list_my_drafts` is for the other kind of submission: documents sent with `create_article` or
  `update_article` that are still waiting for review.
