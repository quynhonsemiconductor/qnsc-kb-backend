"""What the MCP server offers: tools, resources and prompts.

Every tool is a thin mapping onto an existing REST endpoint, and the transport runs it
through the application itself (see `src/api/routers/mcp.py`), acting as the token's owner.
Nothing here queries the database or decides who may see what: RBAC, department scope,
ACLs, row-level security, rate limits and audit are the ones the web app already uses, so
the two can never disagree.

Deliberately absent: deleting or restoring articles, approving drafts, reindexing,
connectors, user and role administration. An AI client that can read and propose
changes is useful; one that can approve its own proposals is not.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.config import settings
from src.domain.mcp_config import APPLY


class ToolInputError(ValueError):
    """The arguments do not fit the tool; reported to the model as a tool error."""


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    params: dict[str, Any] | None = None
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    build: Callable[[dict[str, Any]], Call]
    shape: Callable[[Any], Any] | None = None
    #: Like `shape`, but also given the arguments of the call (paging needs them).
    shape_args: Callable[[Any, dict[str, Any]], Any] | None = None
    write: bool = False
    graph: bool = False

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.required),
                "additionalProperties": False,
            },
            "annotations": {
                "title": self.title,
                "readOnlyHint": not self.write,
                "destructiveHint": False,
                "idempotentHint": not self.write,
                "openWorldHint": False,
            },
        }


# -- argument helpers ---------------------------------------------------------------


def _text(
    args: dict[str, Any], key: str, *, required: bool = False, max_len: int = 4000
) -> str | None:
    value = args.get(key)
    if value is None or value == "":
        if required:
            raise ToolInputError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"'{key}' must be a string")
    if len(value) > max_len:
        raise ToolInputError(f"'{key}' is longer than {max_len} characters")
    return value


def _id(args: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    value = _text(args, key, required=required, max_len=64)
    if value is None:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise ToolInputError(f"'{key}' must be a UUID") from None


def _int(args: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"'{key}' must be an integer")
    return max(low, min(value, high))


def _optional_bool(args: dict[str, Any], key: str) -> bool | None:
    value = args.get(key)
    if value is not None and not isinstance(value, bool):
        raise ToolInputError(f"'{key}' must be true or false")
    return True if value else None


def _tags(args: dict[str, Any]) -> list[str]:
    value = args.get("tags") or []
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise ToolInputError("'tags' must be a list of strings")
    return value[:20]


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _pick(item: Any, keys: tuple[str, ...]) -> Any:
    return (
        {k: item.get(k) for k in keys if k in item} if isinstance(item, dict) else item
    )


# -- result shaping -----------------------------------------------------------------

ARTICLE_SUMMARY = (
    "id",
    "title",
    "dept",
    "type",
    "sensitivity",
    "status",
    "language",
    "version",
    "index_status",
    "created_at",
    "next_review",
    "tags",
)
ARTICLE_FULL = ARTICLE_SUMMARY + ("body_md", "source_available", "owner", "departments")


def _shape_article_list(data: Any) -> Any:
    if isinstance(data, list):
        return [_pick(item, ARTICLE_SUMMARY) for item in data]
    return data


def _shape_article(data: Any) -> Any:
    return _pick(data, ARTICLE_FULL)


def _page_end(body: str, start: int, window: int) -> int:
    """Where this page of a long body ends: at a line break when one is near the end."""
    end = min(len(body), start + window)
    if end >= len(body):
        return end
    cut = body.rfind("\n", start + (window * 4) // 5, end)
    return cut + 1 if cut != -1 else end


def _shape_article_page(data: Any, args: dict[str, Any]) -> Any:
    """One window of the article body, so a long document can be read to its end.

    A body longer than the window used to be cut off with a note and the rest was unreachable:
    `search_knowledge` returns passages, not "the next 30 000 characters". `offset` and the
    `next_offset` in the result make the whole text readable page by page.
    """
    article = _pick(data, ARTICLE_FULL)
    body = article.get("body_md") if isinstance(article, dict) else None
    if not isinstance(body, str):
        return article
    start = _int(args, "offset", 0, 0, 10**9)
    window = max(settings.MCP_MAX_RESULT_CHARS // 2, 1000)
    end = _page_end(body, start, window)
    article = {**article, "body_md": body[start:end]}
    article["body_chars"] = len(body)
    article["offset"] = start
    if end < len(body):
        article["next_offset"] = end
        article["truncated"] = (
            f"{len(body) - end} more characters; call get_article again with "
            f"offset={end} to continue."
        )
    return article


CONVERSATION_MESSAGE_CHARS = 3000


def _shape_conversations(data: Any, args: dict[str, Any]) -> Any:
    if not isinstance(data, list):
        return data
    needle = (_text(args, "q", max_len=200) or "").lower()
    rows = [
        _pick(item, ("id", "title", "updated_at"))
        for item in data
        if not needle or needle in str(item.get("title") or "").lower()
    ]
    return rows[: _int(args, "limit", 20, 1, 50)]


def _shape_conversation(data: Any, args: dict[str, Any]) -> Any:
    """The last `limit` turns of a conversation, answers as written for the user."""
    if not isinstance(data, list):
        return data
    turns = []
    for message in data[-_int(args, "limit", 20, 1, 50) :]:
        if not isinstance(message, dict):
            continue
        text = message.get("answer_grounded") or message.get("content")
        turn = _pick(message, ("role", "created_at", "usage_log_id"))
        turn["text"] = _clip(text, CONVERSATION_MESSAGE_CHARS)
        citations = [_shape_citation(c) for c in (message.get("citations") or [])[:10]]
        if citations:
            turn["citations"] = citations
        turns.append(_drop_none(turn))
    return turns


def _shape_versions(data: Any) -> Any:
    if isinstance(data, list):
        return [_pick(item, ("version", "created_at")) for item in data]
    return data


def _shape_departments(data: Any) -> Any:
    if isinstance(data, list):
        return [_pick(item, ("id", "name", "active")) for item in data]
    return data


def _shape_config(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    if data.get("up_to_date"):
        return {
            "up_to_date": True,
            "version": data.get("version"),
            "message": "Your copy is current. Change nothing.",
        }
    if not data.get("configured"):
        return {
            "configured": False,
            "message": "No team configuration has been set yet. Nothing to apply.",
        }
    return {
        "version": data.get("version"),
        "block": data.get("block"),
        "how_to_apply": APPLY,
    }


HIT_TEXT_CHARS = 1500
CITATION_EXCERPT_CHARS = 600


def _clip(text: Any, limit: int) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit].rstrip() + " [...]"
    return text


def _shape_search(data: Any) -> Any:
    """One passage per hit. The REST payload also carries the parent section and every
    child chunk (the same text again, several times over), which only costs the model
    context; `get_article` is the way to read more."""
    if not isinstance(data, list):
        return data
    hits = []
    for hit in data:
        if not isinstance(hit, dict):
            hits.append(hit)
            continue
        slim = _pick(
            hit,
            ("article_id", "title", "dept", "type", "heading", "page_number", "score"),
        )
        slim["text"] = _clip(hit.get("chunk_text"), HIT_TEXT_CHARS)
        slim["source_url"] = hit.get("source_url")
        evidence = hit.get("evidence")
        if isinstance(evidence, dict):
            slim["evidence"] = _pick(evidence, ("kind", "label", "page_number"))
        hits.append(_drop_none(slim))
    return hits


def _shape_citation(citation: Any) -> Any:
    if not isinstance(citation, dict):
        return citation
    slim = _pick(
        citation,
        ("source_id", "article_id", "title", "heading", "page_number", "source_url"),
    )
    slim["excerpt"] = _clip(citation.get("excerpt"), CITATION_EXCERPT_CHARS)
    return _drop_none(slim)


def _shape_answer(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    slim = _pick(data, ("answer", "conversation_id", "log_id"))
    slim["citations"] = [_shape_citation(c) for c in data.get("citations") or []]
    # `answer` already contains both parts; keep them separately only when they differ.
    for key in ("answer_grounded", "answer_extended"):
        if data.get(key) and data.get(key) != data.get("answer"):
            slim[key] = data[key]
    return slim


# -- tools --------------------------------------------------------------------------

_ID = {"type": "string", "description": "Article id (UUID)", "format": "uuid"}
_LANG = {"type": "string", "enum": ["vi", "en"], "description": "Answer language"}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 50}


def _choice(args: dict[str, Any], name: str, allowed: tuple[str, ...]) -> str | None:
    value = _text(args, name, max_len=20)
    if value is not None and value not in allowed:
        raise ToolInputError(f"'{name}' must be one of: {', '.join(allowed)}")
    return value


def _search(args: dict[str, Any]) -> Call:
    return Call(
        "GET",
        "/search",
        params=_drop_none(
            {
                "q": _text(args, "query", required=True),
                "dept": _text(args, "dept", max_len=100),
                "type": _text(args, "type", max_len=50),
                "tag": _text(args, "tag", max_len=50),
                "language": _text(args, "language", max_len=20),
                "scope": _choice(args, "scope", ("all", "documents", "lessons")),
                "limit": _int(args, "limit", 5, 1, 20),
            }
        ),
    )


def _ask(args: dict[str, Any]) -> Call:
    language = _text(args, "language", max_len=2) or "vi"
    if language not in ("vi", "en"):
        raise ToolInputError("'language' must be 'vi' or 'en'")
    question = _text(args, "question", required=True)
    if len(question or "") < 2:
        raise ToolInputError("'question' is too short")
    return Call(
        "POST",
        "/ai/ask",
        body=_drop_none(
            {
                "question": question,
                "conversation_id": _id(args, "conversation_id", required=False),
                "article_id": _id(args, "article_id", required=False),
                "language": language,
                "sources": _choice(args, "sources", ("all", "documents")),
            }
        ),
    )


def _list_articles(args: dict[str, Any]) -> Call:
    return Call(
        "GET",
        "/articles/",
        params=_drop_none(
            {
                "q": _text(args, "q", max_len=200),
                "dept": _text(args, "dept", max_len=100),
                "topic": _text(args, "topic", max_len=80),
                "status": _text(args, "status", max_len=30),
                "limit": _int(args, "limit", 20, 1, 50),
                "offset": _int(args, "offset", 0, 0, 100_000),
            }
        ),
    )


def _create_article(args: dict[str, Any]) -> Call:
    visibility = _text(args, "visibility", max_len=12) or "department"
    if visibility not in ("public", "department"):
        raise ToolInputError("'visibility' must be 'public' or 'department'")
    return Call(
        "POST",
        "/articles/",
        body=_drop_none(
            {
                "title": _text(args, "title", required=True, max_len=255),
                "body_md": _text(args, "body_md", required=True, max_len=2_000_000),
                "dept": _text(args, "dept", max_len=100),
                "language": _text(args, "language", max_len=20) or "vi",
                "tags": _tags(args),
                "visibility": visibility,
            }
        ),
    )


def _get_article(args: dict[str, Any]) -> Call:
    _int(args, "offset", 0, 0, 10**9)  # refused before the call, not after it
    return Call("GET", f"/articles/{_id(args, 'article_id')}")


def _conversations(args: dict[str, Any]) -> Call:
    _text(args, "q", max_len=200)
    _int(args, "limit", 20, 1, 50)
    return Call("GET", "/ai/conversations")


def _conversation(args: dict[str, Any]) -> Call:
    _int(args, "limit", 20, 1, 50)
    return Call("GET", f"/ai/conversations/{_id(args, 'conversation_id')}/messages")


def _list_lessons(args: dict[str, Any]) -> Call:
    return Call(
        "GET",
        "/knowledge/lessons",
        params=_drop_none(
            {
                "q": _text(args, "q", max_len=200),
                "mine": _optional_bool(args, "mine"),
                "limit": _int(args, "limit", 20, 1, 50),
            }
        ),
    )


def _forget_lesson(args: dict[str, Any]) -> Call:
    # A POST, not a DELETE: this server offers nothing that deletes. It takes ONE lesson out of search, and only
    # the person who saved it (or someone who may delete the department's articles) is allowed to.
    return Call("POST", f"/knowledge/lessons/{_id(args, 'lesson_id')}/retract")


def _my_drafts(args: dict[str, Any]) -> Call:
    return Call(
        "GET", "/knowledge/my-drafts", params={"limit": _int(args, "limit", 20, 1, 50)}
    )


LESSON_TAG = "session-lesson"


def _record_lesson(args: dict[str, Any]) -> Call:
    """A lesson learned in an AI session.

    The server compares it with what is stored and either publishes it at once as a new
    lesson, or merges it into the existing lesson it adds to, or says which reviewed document
    to ask an editor to change, or that it is already known (src/domain/lessons.py). There is
    no review step; the body shape and the `session-lesson` tag are applied there.
    """
    return Call(
        "POST",
        "/knowledge/lessons",
        body=_drop_none(
            {
                "title": _text(args, "title", required=True, max_len=255),
                "lesson": _text(args, "lesson", required=True, max_len=20_000),
                "context": _text(args, "context", max_len=20_000),
                "dept": _text(args, "dept", max_len=100),
                "language": _text(args, "language", max_len=20) or "vi",
                "tags": _tags(args),
            }
        ),
    )


def _update_article(args: dict[str, Any]) -> Call:
    body = _drop_none(
        {
            "title": _text(args, "title", max_len=255),
            "body_md": _text(args, "body_md", max_len=2_000_000),
            "tags": _tags(args) if "tags" in args else None,
            "version": (
                _int(args, "version", 1, 1, 10**9) if "version" in args else None
            ),
        }
    )
    if not {"title", "body_md", "tags"} & body.keys():
        raise ToolInputError("give at least one of title, body_md or tags to change")
    return Call("PUT", f"/articles/{_id(args, 'article_id')}", body=body)


def _edit_request(args: dict[str, Any]) -> Call:
    text = _text(args, "request_text", required=True, max_len=5000)
    if len(text or "") < 5:
        raise ToolInputError("'request_text' must be at least 5 characters")
    return Call(
        "POST",
        f"/articles/{_id(args, 'article_id')}/edit-requests",
        body={"request_text": text},
    )


def _feedback(args: dict[str, Any]) -> Call:
    rating = args.get("rating")
    if isinstance(rating, bool) or rating not in (-1, 1):
        raise ToolInputError("'rating' must be 1 (helpful) or -1 (not helpful)")
    return Call(
        "POST",
        "/ai/feedback",
        body=_drop_none(
            {
                "ai_usage_log_id": _id(args, "log_id"),
                "rating": rating,
                "comment": _text(args, "comment", max_len=2000),
            }
        ),
    )


def _version(args: dict[str, Any]) -> Call:
    number = _int(args, "version", 1, 1, 10**9)
    return Call("GET", f"/articles/{_id(args, 'article_id')}/versions/{number}")


def _related(args: dict[str, Any]) -> Call:
    return Call(
        "GET",
        f"/articles/{_id(args, 'article_id')}/related",
        params={"limit": _int(args, "limit", 6, 1, 20)},
    )


def _entities(args: dict[str, Any]) -> Call:
    return Call(
        "GET",
        "/graph/entities",
        params=_drop_none(
            {
                "q": _text(args, "q", max_len=200),
                "type": _text(args, "type", max_len=40),
                "limit": _int(args, "limit", 20, 1, 200),
            }
        ),
    )


TOOLS: tuple[Tool, ...] = (
    Tool(
        "search_knowledge",
        "Search the knowledge base",
        "Hybrid (keyword + semantic) search over the company knowledge base. Returns the "
        "best matching passages, each with its article_id, page and score. Use it to find "
        "where something is documented or to gather raw evidence; call get_article with "
        "the article_id to read the whole document. For a ready-made answer with "
        "citations use ask_knowledge_base instead. Only results the user may see are "
        "returned. Vietnamese and English both work; query in the language of the "
        "documents.",
        {
            "query": {"type": "string", "description": "What to look for"},
            "dept": {"type": "string", "description": "Restrict to a department name"},
            "type": {
                "type": "string",
                "description": "POLICY, SOP, DECISION, FAQ, RCA, HOWTO, PLAYBOOK or REFERENCE",
            },
            "tag": {"type": "string"},
            "language": {"type": "string", "description": "vi or en"},
            "scope": {
                "type": "string",
                "enum": ["all", "documents", "lessons"],
                "description": "documents: reviewed uploads only; lessons: what assistants "
                "saved with record_lesson; all (default): both. A lesson's type is LESSON.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("query",),
        _search,
        _shape_search,
    ),
    Tool(
        "ask_knowledge_base",
        "Ask the knowledge base",
        "Ask a question and get an answer written by the KB assistant, grounded in "
        "documents the user may access, with numbered citations ([C1], [C2]...) that map "
        "to the 'citations' list (article_id, page, excerpt). Prefer this for 'what is our "
        "policy on...' or 'how do we...' questions; use search_knowledge when you want "
        "the raw passages. It reads the whole knowledge base, so questions about "
        "'all documents' work too. Pass conversation_id from an earlier answer to ask a "
        "follow-up, and article_id to confine the question to one article. Relay the "
        "citations to the user; if the answer says nothing relevant was found, say so.",
        {
            "question": {"type": "string"},
            "language": _LANG,
            "conversation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Continue this conversation",
            },
            "article_id": {**_ID, "description": "Limit the question to one article"},
            "sources": {
                "type": "string",
                "enum": ["all", "documents"],
                "description": "all: reviewed documents and the lessons assistants saved "
                "(an answer says when it relies on a lesson); documents: reviewed uploads only. "
                "Default: the deployment's setting.",
            },
        },
        ("question",),
        _ask,
        _shape_answer,
    ),
    Tool(
        "get_article",
        "Read an article",
        "The text (Markdown) and metadata of one article: title, department, version, "
        "status, tags. A long article comes in pages: when the result has next_offset, "
        "call again with offset=next_offset to read on (body_chars is the full length). "
        "Pass the 'version' you read to update_article.",
        {
            "article_id": _ID,
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Character to start reading from (next_offset of the "
                "previous page)",
            },
        },
        ("article_id",),
        _get_article,
        shape_args=_shape_article_page,
    ),
    Tool(
        "list_conversations",
        "List your past conversations",
        "The questions this user has asked the knowledge base before, newest first, each "
        "with a conversation_id. Use it to pick up an earlier line of questioning or to "
        "check whether something was already asked; pass q to filter by title. Only the "
        "user's own conversations are ever listed.",
        {
            "q": {"type": "string", "description": "Match against titles"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        (),
        _conversations,
        shape_args=_shape_conversations,
    ),
    Tool(
        "get_conversation",
        "Read a past conversation",
        "The turns of one earlier conversation (the last `limit`, default 20): who said "
        "what, with the citations of each answer. Take the conversation_id from "
        "list_conversations or from an earlier ask_knowledge_base answer, and pass it to "
        "ask_knowledge_base to continue that conversation.",
        {
            "conversation_id": {"type": "string", "format": "uuid"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        ("conversation_id",),
        _conversation,
        shape_args=_shape_conversation,
    ),
    Tool(
        "list_lessons",
        "List team lessons",
        "The lessons assistants saved with record_lesson that this user may read, newest change "
        "first. Pass mine=true for only the ones this user saved, q to filter by title or text. "
        "Use it to see what the team already recorded, or to find a lesson to retract.",
        {
            "q": {"type": "string", "description": "Match against title or text"},
            "mine": {"type": "boolean", "description": "Only lessons this user saved"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        (),
        _list_lessons,
    ),
    Tool(
        "forget_lesson",
        "Retract a lesson",
        "Take a lesson out of search and answers because it is wrong or obsolete. Allowed for "
        "the person who saved it and for whoever may delete the department's articles. A "
        "lesson that is merely out of date is better corrected by calling record_lesson with "
        "the new fact, which merges into it. Requires a token with write scope.",
        {"lesson_id": {**_ID, "description": "Lesson id (from list_lessons)"}},
        ("lesson_id",),
        _forget_lesson,
        write=True,
    ),
    Tool(
        "list_my_drafts",
        "List what you submitted that is still waiting for review",
        "Documents this user submitted with create_article or update_article that are not "
        "published yet: waiting for a reviewer, or rejected with the reviewer's note. Lessons "
        "are not here: record_lesson publishes at once (see list_lessons). Only the user's own "
        "submissions are listed.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
        (),
        _my_drafts,
    ),
    Tool(
        "list_articles",
        "List articles",
        "Browse articles the user can see (newest first), with optional filters. Use it to "
        "answer 'what documents do we have about...' or to find an article by title; "
        "returns metadata only, no body.",
        {
            "q": {"type": "string", "description": "Match against titles"},
            "dept": {"type": "string"},
            "topic": {"type": "string"},
            "status": {"type": "string"},
            "limit": _LIMIT,
            "offset": {"type": "integer", "minimum": 0},
        },
        (),
        _list_articles,
        _shape_article_list,
    ),
    Tool(
        "get_related_articles",
        "Find related articles",
        "Articles related to the given one.",
        {"article_id": _ID, "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        ("article_id",),
        _related,
        _shape_article_list,
    ),
    Tool(
        "list_article_versions",
        "List article versions",
        "The version history of an article.",
        {"article_id": _ID},
        ("article_id",),
        lambda a: Call("GET", f"/articles/{_id(a, 'article_id')}/versions"),
        _shape_versions,
    ),
    Tool(
        "get_article_version",
        "Read an old article version",
        "The stored snapshot of one earlier version of an article.",
        {"article_id": _ID, "version": {"type": "integer", "minimum": 1}},
        ("article_id", "version"),
        _version,
    ),
    Tool(
        "list_departments",
        "List departments",
        "Departments of the user's company, for filtering and for choosing where to file "
        "a new article.",
        {},
        (),
        lambda a: Call("GET", "/auth/departments"),
        _shape_departments,
    ),
    Tool(
        "get_project_config",
        "Get the team configuration",
        "The team's standing instructions for AI assistants, kept by the administrators and "
        "versioned. Call it when the server's instructions name a configuration version "
        "that your CLAUDE.md block does not have, passing known_version if you have one: "
        "it answers 'up to date' or returns the new block and how to apply it.",
        {
            "known_version": {
                "type": "integer",
                "description": "The version in your current block, if any",
            }
        },
        (),
        lambda a: Call(
            "GET",
            "/mcp-config",
            params=(
                {"known_version": _int(a, "known_version", -1, -1, 10**9)}
                if _int(a, "known_version", -1, -1, 10**9) >= 0
                else None
            ),
        ),
        _shape_config,
    ),
    Tool(
        "list_tags",
        "List tags",
        "Tags in use, with counts.",
        {},
        (),
        lambda a: Call("GET", "/meta/tags"),
    ),
    Tool(
        "get_glossary",
        "Read the glossary",
        "Company glossary terms and their definitions.",
        {},
        (),
        lambda a: Call("GET", "/meta/glossary"),
    ),
    Tool(
        "find_entities",
        "Search the knowledge graph",
        "Find people, systems, products or other entities extracted from the documents. "
        "Only available when the knowledge graph is enabled on this server.",
        {
            "q": {"type": "string"},
            "type": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        (),
        _entities,
        graph=True,
    ),
    Tool(
        "get_entity_neighbors",
        "Explore the knowledge graph",
        "Entities connected to one entity, and the articles that connect them. Take the "
        "entity_id from find_entities.",
        {"entity_id": {"type": "string", "format": "uuid"}},
        ("entity_id",),
        lambda a: Call("GET", f"/graph/entities/{_id(a, 'entity_id')}/neighbors"),
        graph=True,
    ),
    Tool(
        "create_article",
        "Submit a new article",
        "Submit a new article. It goes through the normal review workflow, so it may be a "
        "draft awaiting approval rather than published; the result says which. Requires a "
        "token with write scope and the user's own permission to create articles.",
        {
            "title": {"type": "string", "maxLength": 255},
            "body_md": {"type": "string", "description": "Article body in Markdown"},
            "dept": {"type": "string"},
            "language": {"type": "string", "description": "vi or en"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            "visibility": {"type": "string", "enum": ["department", "public"]},
        },
        ("title", "body_md"),
        _create_article,
        write=True,
    ),
    Tool(
        "record_lesson",
        "Record a lesson learned in this session",
        "Save something worth keeping from this session -- a decision and its reason, a fix "
        "that worked, a rule you were told -- so the next person's AI can find it. Write the "
        "lesson so it makes sense without this conversation. A lesson is published at once for "
        "the user's department, with no reviewer: search_knowledge and ask_knowledge_base find "
        "it straight away, labelled as an unreviewed team lesson. The server first checks what "
        "is already stored: the result says decision 'add' (saved), 'update' (it was merged into "
        "an earlier lesson, or a reviewed document already covers it: then nothing was saved, "
        "use request_article_edit if the document is wrong) or 'skip' (already known; nothing "
        "was saved). Requires a token with write scope.",
        {
            "title": {"type": "string", "maxLength": 255},
            "lesson": {
                "type": "string",
                "description": "What was learned, in a few sentences, standalone",
            },
            "context": {
                "type": "string",
                "description": "Optional: where it applies and why it holds",
            },
            "dept": {"type": "string"},
            "language": {"type": "string", "description": "vi or en"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 19},
        },
        ("title", "lesson"),
        _record_lesson,
        write=True,
    ),
    Tool(
        "update_article",
        "Propose changes to an article",
        "Change an article's title, body or tags. Goes through the normal review "
        "workflow. Pass the version you read so that someone else's newer edit is not "
        "overwritten.",
        {
            "article_id": _ID,
            "title": {"type": "string"},
            "body_md": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "version": {
                "type": "integer",
                "minimum": 1,
                "description": "The version you read",
            },
        },
        ("article_id",),
        _update_article,
        write=True,
    ),
    Tool(
        "request_article_edit",
        "Report a correction",
        "Tell the article's editors something is wrong or outdated. Lighter than "
        "update_article: nothing is changed, the editors are notified.",
        {"article_id": _ID, "request_text": {"type": "string", "minLength": 5}},
        ("article_id", "request_text"),
        _edit_request,
        write=True,
    ),
    Tool(
        "rate_answer",
        "Rate an answer",
        "Record whether an ask_knowledge_base answer helped (1) or not (-1). Use the "
        "log_id from that answer.",
        {
            "log_id": {"type": "string", "format": "uuid"},
            "rating": {"type": "integer", "enum": [-1, 1]},
            "comment": {"type": "string"},
        },
        ("log_id", "rating"),
        _feedback,
        write=True,
    ),
)

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def available(tool: Tool, can_write: bool) -> bool:
    """Whether this server and this token offer the tool at all."""
    if tool.write and not can_write:
        return False
    return settings.KNOWLEDGE_GRAPH_ENABLED or not tool.graph


def visible_tools(can_write: bool) -> list[Tool]:
    return [tool for tool in TOOLS if available(tool, can_write)]


# -- result formatting --------------------------------------------------------------


def cap_text(text: str, limit: int | None = None) -> str:
    limit = limit or settings.MCP_MAX_RESULT_CHARS
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        f"{text[:limit]}\n\n[truncated: {omitted} more characters. "
        "Ask for a narrower result.]"
    )


def _cap_article_body(data: Any, limit: int) -> Any:
    if isinstance(data, dict) and isinstance(data.get("body_md"), str):
        body = data["body_md"]
        if len(body) > limit:
            data = {
                **data,
                "body_md": body[:limit]
                + f"\n\n[truncated: {len(body) - limit} more characters]",
            }
    return data


#: What a model should do about an error it cannot otherwise act on. Matched on the text of
#: the REST error, so the REST message stays the single source of what went wrong.
ERROR_HINTS = (
    (
        "A department is required",
        "Pass `dept` (call list_departments for the names): this user has no default "
        "department to file under.",
    ),
    (
        "Department does not exist or is inactive",
        "Call list_departments and pass one of its names exactly.",
    ),
)


def _hint_for(detail: str) -> str:
    for marker, hint in ERROR_HINTS:
        if marker in detail:
            return f" {hint}"
    return ""


def format_result(
    tool: Tool,
    status_code: int,
    payload: Any,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An MCP `tools/call` result. HTTP errors become tool errors the model can act on."""
    if status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False, default=str)
        return error_result(
            f"Error {status_code}: {cap_text(detail, 2000)}{_hint_for(detail)}"
        )
    if tool.shape_args is not None:
        data = tool.shape_args(payload, arguments or {})
    else:
        data = tool.shape(payload) if tool.shape else payload
    data = _cap_article_body(data, settings.MCP_MAX_RESULT_CHARS // 2)
    text = cap_text(json.dumps(data, ensure_ascii=False, default=str, indent=2))
    return {"content": [{"type": "text", "text": text}], "isError": False}


def error_result(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


# -- resources ----------------------------------------------------------------------

RESOURCE_SCHEME = "kb://article/"


def resource_templates() -> list[dict[str, Any]]:
    return [
        {
            "uriTemplate": f"{RESOURCE_SCHEME}{{id}}",
            "name": "article",
            "title": "Knowledge base article",
            "description": "The Markdown text of an article the user can read.",
            "mimeType": "text/markdown",
        }
    ]


def article_uri(article_id: str) -> str:
    return f"{RESOURCE_SCHEME}{article_id}"


def article_id_from_uri(uri: Any) -> str:
    if not isinstance(uri, str) or not uri.startswith(RESOURCE_SCHEME):
        raise ToolInputError(f"unknown resource: {uri!r}")
    try:
        return str(uuid.UUID(uri[len(RESOURCE_SCHEME) :]))
    except ValueError:
        raise ToolInputError(f"unknown resource: {uri!r}") from None


def article_markdown(article: dict[str, Any]) -> str:
    head = [f"# {article.get('title', '')}", ""]
    meta = [
        f"{key}: {article[key]}"
        for key in ("dept", "type", "status", "version", "language")
        if article.get(key) is not None
    ]
    if meta:
        head += ["> " + " · ".join(meta), ""]
    return cap_text("\n".join(head) + (article.get("body_md") or ""))


# -- prompts ------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    name: str
    title: str
    description: str
    arguments: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    render: Callable[[dict[str, str]], str] = lambda a: ""

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "arguments": list(self.arguments),
        }


def _arg(name: str, description: str, required: bool = True) -> dict[str, Any]:
    return {"name": name, "description": description, "required": required}


PROMPTS: tuple[Prompt, ...] = (
    Prompt(
        "explore_knowledge_base",
        "What is in the knowledge base?",
        "A first look for someone new: which departments and kinds of documents exist, "
        "and questions worth asking.",
        (),
        lambda a: (
            "Give me a quick tour of the company knowledge base. Call list_departments, "
            "then list_articles (a couple of pages if needed) and, if tags exist, "
            "list_tags. Summarize what areas and document types it covers, which "
            "departments own what, and how recent it is. Finish with five concrete "
            "questions I could ask with ask_knowledge_base, based on titles you actually "
            "saw. Do not invent topics."
        ),
    ),
    Prompt(
        "summarize_article",
        "Summarize an article",
        "A short summary of one article, with the key decisions, owners and dates.",
        (_arg("article_id", "Article id (UUID)"),),
        lambda a: (
            f"Read the knowledge base article {a['article_id']} with get_article and "
            "summarize it in at most 8 bullet points. Name decisions, owners, deadlines "
            "and numbers explicitly. Do not add anything the article does not say."
        ),
    ),
    Prompt(
        "answer_with_sources",
        "Answer from the knowledge base",
        "Answer a question using only the knowledge base, citing the articles used.",
        (_arg("question", "The question to answer"),),
        lambda a: (
            "Answer this question using only the company knowledge base: "
            f"{a['question']}\n\n"
            "Call ask_knowledge_base first. If its answer is thin, use search_knowledge "
            "and get_article to check the sources yourself. Cite every article you relied "
            "on by title. If the knowledge base does not contain the answer, say so "
            "plainly instead of guessing."
        ),
    ),
    Prompt(
        "compare_articles",
        "Compare two articles",
        "Where two articles agree, differ or contradict each other.",
        (
            _arg("article_id_a", "First article id"),
            _arg("article_id_b", "Second article id"),
        ),
        lambda a: (
            f"Fetch articles {a['article_id_a']} and {a['article_id_b']} with "
            "get_article. List what they agree on, where they differ, and any "
            "contradiction that would confuse a reader. Quote the conflicting sentences."
        ),
    ),
    Prompt(
        "find_knowledge_gaps",
        "Find gaps on a topic",
        "What a topic's documentation is missing or has let go stale.",
        (_arg("topic", "Topic to review"),),
        lambda a: (
            "Use search_knowledge and list_articles to review what the knowledge base "
            f"says about: {a['topic']}. Report what is well covered, what is missing, "
            "and which articles look outdated (old version, past next_review). Suggest "
            "articles worth writing, but do not create them."
        ),
    ),
)

PROMPTS_BY_NAME = {p.name: p for p in PROMPTS}


def render_prompt(name: Any, arguments: dict[str, Any] | None) -> dict[str, Any]:
    prompt = PROMPTS_BY_NAME.get(name) if isinstance(name, str) else None
    if prompt is None:
        raise ToolInputError(f"unknown prompt: {name!r}")
    given = arguments or {}
    values: dict[str, str] = {}
    for spec in prompt.arguments:
        value = given.get(spec["name"])
        if spec["required"] and (not isinstance(value, str) or not value.strip()):
            raise ToolInputError(f"missing argument: {spec['name']}")
        if value is not None:
            if not isinstance(value, str) or len(value) > 4000:
                raise ToolInputError(f"invalid argument: {spec['name']}")
            values[spec["name"]] = value.strip()
    for key in ("article_id", "article_id_a", "article_id_b"):
        if key in values:
            _id(values, key)
    return {
        "description": prompt.description,
        "messages": [
            {"role": "user", "content": {"type": "text", "text": prompt.render(values)}}
        ],
    }
