"""Four defects found by auditing the live stack, each pinned against regression.

All four were measured, not inferred:

1. Keyword search ANDed every term, so a question longer than its own subject could not
   match the passage containing that subject. Against the live corpus:

       "CCOpt"                                    2 hits
       "Innovus CCOpt"                            2 hits
       "Innovus CCOpt hoat dong"                  0 hits   <- conjunction dies here
       "Innovus CCOpt hoat dong nhu the nao ..."  0 hits

   An English technical passage can never contain the Vietnamese syllables of the question
   asking about it, so the keyword leg was dead for every phrased Vietnamese question.

2. Comment and bookmark responses serialised eager-loaded ORM `User` rows, so every reply
   carried `password_hash`. Reproduced with `jsonable_encoder`.

3. `X-Request-ID` was set on every response but not exposed through CORS, so browser
   JavaScript read `null` — making the request-failure lookup unusable from the UI, which
   was the entire point of persisting it.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.dialects import postgresql

from src.models.interaction import Comment
from src.models.user import Department, User


class _Result:
    rowcount = 0

    def scalars(self):
        return self

    def all(self):
        return []


class _RecordingDB:
    def __init__(self):
        self.sql: list[str] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        try:
            self.sql.append(str(statement.compile(dialect=postgresql.dialect())))
        except Exception:
            self.sql.append(str(statement))
        return _Result()

    async def scalar(self, statement):
        return 0


def _user() -> User:
    user = User(
        id=uuid.uuid4(), role="Staff", company_domain="qnsc.vn", dept="Engineering"
    )
    user.departments = [Department(id=uuid.uuid4(), name="Engineering", active=True)]
    return user


def _keyword_sql(query: str) -> str:
    from src.repositories.chunk import ChunkRepository

    db = _RecordingDB()
    asyncio.run(
        ChunkRepository(db).hybrid_search(
            user=_user(),
            query=query,
            query_embedding=None,
            limit=5,
            filters={},
        )
    )
    matches = [item for item in db.sql if "to_tsvector" in item]
    assert matches, "no keyword statement was issued"
    return matches[0]


# ----------------------------------------------------------- 1. keyword search recall


def test_keyword_terms_are_or_joined_not_and_joined():
    """The regression: ANDing meant the passage had to contain EVERY word of the question."""
    sql = _keyword_sql("Innovus CCOpt hoat dong nhu the nao trong CTS")

    assert "||" in sql, "terms must be OR-joined; `&` semantics kill multi-word recall"


def test_every_content_term_gets_its_own_tsquery():
    """One tsquery per term is what allows the OR; a single plainto_tsquery cannot."""
    sql = _keyword_sql("Innovus CCOpt hoat dong")

    # innovus, ccopt, hoat, dong all survive normalisation as content words.
    assert sql.count("plainto_tsquery") >= 4


def test_query_terms_are_bound_parameters_not_inlined_sql():
    """Interpolating a term would put user text inside a tsquery expression."""
    sql = _keyword_sql("Innovus CCOpt hoat dong")

    assert "ccopt" not in sql.lower()
    assert "innovus" not in sql.lower()


def test_ranking_is_preserved_so_a_broad_or_does_not_flatten_relevance():
    """OR admits weak matches; ts_rank_cd plus the relevance floor is what sorts them."""
    sql = _keyword_sql("Innovus CCOpt hoat dong")

    assert "ts_rank_cd" in sql


def test_a_single_term_query_still_works():
    """The case that already worked must not regress."""
    sql = _keyword_sql("CCOpt")

    assert "plainto_tsquery" in sql
    assert "to_tsvector" in sql


def test_a_query_with_no_usable_terms_still_produces_a_valid_tsquery():
    """An empty tsquery matches nothing; the fallback must keep the statement sane."""
    sql = _keyword_sql("a")

    assert "plainto_tsquery" in sql


# ------------------------------------------------------- 2. password hash never leaves


def test_the_raw_orm_comment_would_have_leaked_the_password_hash():
    """Pins WHY the response model is required, not merely that one is set."""
    user = User(
        id=uuid.uuid4(),
        email="owner@example.com",
        name="Owner",
        password_hash="$2b$12$SECRET",
        role="Staff",
        company_domain="qnsc.vn",
    )
    comment = Comment(
        id=uuid.uuid4(), article_id=uuid.uuid4(), user_id=user.id, text="hi"
    )
    comment.user = user

    # This is what FastAPI does WITHOUT a response_model.
    assert "password_hash" in str(jsonable_encoder(comment))


def test_the_comment_response_model_exposes_only_safe_user_fields():
    from src.api.routers.interactions import CommentResponse

    user = User(
        id=uuid.uuid4(),
        email="owner@example.com",
        name="Owner",
        password_hash="$2b$12$SECRET",
        role="Staff",
        company_domain="qnsc.vn",
    )
    comment = Comment(
        id=uuid.uuid4(), article_id=uuid.uuid4(), user_id=user.id, text="hi"
    )
    comment.user = user
    comment.created_at = datetime.utcnow()

    dumped = CommentResponse.model_validate(comment).model_dump()

    assert sorted(dumped["user"]) == ["email", "id", "name"]
    assert "password_hash" not in str(dumped)


def test_both_comment_routes_declare_a_response_model():
    """A route without one silently falls back to serialising the whole ORM graph.

    Asserted against the module SOURCE, not `app.routes`: the app object is built at import
    time, so a route table read from an already-imported module still shows the old
    decorators and a removed `response_model` goes undetected. Mutation-checked — the route
    table version passed against the broken code.
    """
    import inspect

    from src.api.routers import interactions

    source = inspect.getsource(interactions)
    decorators = [
        line.strip()
        for line in source.splitlines()
        if '"/articles/{id}/comments"' in line and "@router." in line
    ]

    assert len(decorators) == 2, f"expected the GET and the POST, found {decorators}"
    for decorator in decorators:
        assert "response_model=" in decorator, decorator


def test_bookmarks_are_projected_rather_than_returned_raw():
    """list_bookmarks hands back ORM Articles with `owner` eager-loaded.

    Asserted on the RETURN statement, not merely that `_article_card` appears somewhere in
    the function: the `import` line alone satisfied a substring check, so the test passed
    against code that had dropped the projection entirely. Mutation-checked.
    """
    import inspect

    from src.api.routers import interactions

    source = inspect.getsource(interactions.list_bookmarks)

    assert "return [_article_card(" in source, (
        "bookmarks must return the projected card, not the raw ORM rows"
    )


# ------------------------------------------------- 3. the request id must reach the UI


def test_the_request_id_header_is_exposed_to_browser_javascript():
    """Measured: `response.headers.get("X-Request-ID")` returned null in the browser.

    `allow_headers` covers the REQUEST direction only. Without `expose_headers` the browser
    hides the response header from JavaScript, so the id an operator needs for
    /governance/request-failures?request_id=... could never be reported by a user.
    """
    from src.api.main import app

    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            exposed = middleware.kwargs.get("expose_headers") or []
            assert "X-Request-ID" in exposed
            return
    raise AssertionError("CORS middleware is not configured")


def test_retry_after_is_exposed_so_clients_can_honour_rate_limits():
    """429 responses carry Retry-After; a client that cannot read it cannot back off."""
    from src.api.main import app

    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            assert "Retry-After" in (middleware.kwargs.get("expose_headers") or [])
            return
    raise AssertionError("CORS middleware is not configured")
