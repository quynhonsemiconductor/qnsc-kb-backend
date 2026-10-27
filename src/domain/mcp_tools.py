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
    write: bool = False

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


def _shape_versions(data: Any) -> Any:
    if isinstance(data, list):
        return [_pick(item, ("version", "created_at")) for item in data]
    return data


def _shape_departments(data: Any) -> Any:
    if isinstance(data, list):
        return [_pick(item, ("id", "name", "active")) for item in data]
    return data


def _shape_answer(data: Any) -> Any:
    return _pick(
        data,
        (
            "answer",
            "citations",
            "conversation_id",
            "log_id",
            "answer_grounded",
            "answer_extended",
        ),
    )


# -- tools --------------------------------------------------------------------------

_ID = {"type": "string", "description": "Article id (UUID)", "format": "uuid"}
_LANG = {"type": "string", "enum": ["vi", "en"], "description": "Answer language"}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 50}


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
        "Hybrid search over the company knowledge base. Returns the most relevant passages "
        "with their article ids. Use this to find where something is documented, then "
        "get_article for the full text. Only results the user may see are returned.",
        {
            "query": {"type": "string", "description": "What to look for"},
            "dept": {"type": "string", "description": "Restrict to a department name"},
            "type": {
                "type": "string",
                "description": "POLICY, SOP, DECISION, FAQ, RCA, HOWTO, PLAYBOOK or REFERENCE",
            },
            "tag": {"type": "string"},
            "language": {"type": "string", "description": "vi or en"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("query",),
        _search,
    ),
    Tool(
        "ask_knowledge_base",
        "Ask the knowledge base",
        "Ask a question and get a cited answer written by the KB assistant, grounded in "
        "documents the user may access. Prefer this for 'what is our policy on...' "
        "questions; use search_knowledge when you want the raw passages. Pass "
        "conversation_id from an earlier answer to ask a follow-up.",
        {
            "question": {"type": "string"},
            "language": _LANG,
            "conversation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Continue this conversation",
            },
            "article_id": {**_ID, "description": "Limit the question to one article"},
        },
        ("question",),
        _ask,
        _shape_answer,
    ),
    Tool(
        "get_article",
        "Read an article",
        "The full text (Markdown) and metadata of one article.",
        {"article_id": _ID},
        ("article_id",),
        lambda a: Call("GET", f"/articles/{_id(a, 'article_id')}"),
        _shape_article,
    ),
    Tool(
        "list_articles",
        "List articles",
        "Browse articles the user can see, with optional filters.",
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
        "Find people, systems, products or other entities extracted from the documents.",
        {
            "q": {"type": "string"},
            "type": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        (),
        _entities,
    ),
    Tool(
        "get_entity_neighbors",
        "Explore the knowledge graph",
        "Entities connected to one entity, and the articles that connect them.",
        {"entity_id": {"type": "string", "format": "uuid"}},
        ("entity_id",),
        lambda a: Call("GET", f"/graph/entities/{_id(a, 'entity_id')}/neighbors"),
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


def visible_tools(can_write: bool) -> list[Tool]:
    return [tool for tool in TOOLS if can_write or not tool.write]


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


def format_result(tool: Tool, status_code: int, payload: Any) -> dict[str, Any]:
    """An MCP `tools/call` result. HTTP errors become tool errors the model can act on."""
    if status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False, default=str)
        return error_result(f"Error {status_code}: {cap_text(detail, 2000)}")
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
