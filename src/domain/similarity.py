"""Near-duplicate detection for uploads and drafts.

This is a REVIEW HINT, not a correctness gate: byte-identical uploads are already
rejected upstream by the `source_hash` check, and `articles.py` deliberately downgrades
an "exact" text match to "very_high" because extracted text can collapse to boilerplate.

Two things about the previous implementation are worth keeping in mind, because both
were load-bearing and neither was obvious.

It was quadratic, uncapped and on the event loop. `SequenceMatcher` is O(n*m); measured
on a developer machine, a single 20,000-character pair costs 8.25 s and the cost rises
roughly 8x per doubling, while `MAX_SOURCE_TEXT_CHARS` admits 2,000,000. That ran once
per existing article, inline in an `async def` with no `await` between iterations, so it
blocked every other request on the worker — SSE answer streams included — for as long as
it took.

It was also WRONG, which matters more. `difflib` enables an "autojunk" heuristic on any
sequence longer than 200 elements, dropping every element that appears in more than 1% of
it. On character sequences that is every letter in the language, so the match index is
gutted and the score collapses. The same 95%-identical pair scores:

    length     autojunk on (before)     autojunk off (now)
       152                    0.970                  0.970
     1,003                    0.212                  0.972
     4,003                    0.181                  0.971
    12,002                    0.673                  0.967

At 1,003 characters a 95%-identical document scored 0.212 — BELOW the 0.25 threshold, so
it was not reported as similar at all. Anything past a couple of hundred characters, which
is every real document, was scored unusably and unstably.

So the comparison now runs with `autojunk=False` on a bounded prefix, off the event loop,
and only against candidates that cheap token overlap already ranks as plausible. Expect
near-duplicates to start being flagged that previously passed silently: that is the
correction, not a regression.
"""

import asyncio
import re
from difflib import SequenceMatcher
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.core.config import settings
from src.models.article import Article
from src.models.user import User
from src.domain.permissions import PermissionService
from src.repositories.article import ArticleRepository

#: Score at or above which a candidate is worth showing a reviewer.
MATCH_THRESHOLD = 0.25
#: Matches returned, highest first.
MATCH_LIMIT = 5


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def token_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_similarity(left: str, right: str) -> float:
    """Character-level similarity over a bounded prefix, with autojunk disabled.

    Both arguments are truncated to ``SIMILARITY_COMPARE_CHARS``. See the module
    docstring: a longer prefix costs super-linearly more and measurably does not move
    the score, and ``autojunk`` silently destroys it on any real document.
    """
    if not left or not right:
        return 0.0
    limit = settings.SIMILARITY_COMPARE_CHARS
    return SequenceMatcher(None, left[:limit], right[:limit], autojunk=False).ratio()


def _rank(content: str, candidates: list[tuple[str, str, str, str]]) -> list[dict]:
    """Score and rank candidates. Pure CPU — called in a worker thread.

    ``candidates`` is ``(article_id, title, lifecycle_status, body_md)``, already
    permission-filtered. Plain strings rather than ORM objects on purpose: this runs off
    the event loop, where touching a lazy relationship on the async session would be a
    bug that only shows up under load.
    """
    normalized = normalize(content)

    scored: list[tuple[float, str, tuple[str, str, str]]] = []
    for article_id, title, lifecycle_status, body in candidates:
        candidate = normalize(body)
        scored.append(
            (
                token_similarity(normalized, candidate),
                candidate,
                (article_id, title, lifecycle_status),
            )
        )

    # Spend the O(n*m) comparison only where cheap token overlap says it could matter.
    # Everything else keeps its token score, which is what `max()` would have chosen
    # anyway for all but pathological inputs.
    order = sorted(range(len(scored)), key=lambda i: scored[i][0], reverse=True)
    compare = set(order[: settings.SIMILARITY_MAX_SEQUENCE_COMPARISONS])

    matches: list[dict] = []
    for index, (token_score, candidate, meta) in enumerate(scored):
        article_id, title, lifecycle_status = meta
        score = token_score
        if index in compare:
            score = max(score, sequence_similarity(normalized, candidate))
        if score >= MATCH_THRESHOLD:
            matches.append(
                {
                    "article_id": article_id,
                    "title": title,
                    "score": round(score, 4),
                    "lifecycle_status": lifecycle_status,
                }
            )
    return sorted(matches, key=lambda item: item["score"], reverse=True)[:MATCH_LIMIT]


async def find_similar_documents(
    db: AsyncSession, user: User, content: str
) -> list[dict]:
    stmt = (
        select(Article)
        .where(*ArticleRepository._authorized_article_filters(user))
        .options(
            selectinload(Article.sources),
            selectinload(Article.access_groups),
            selectinload(Article.departments),
            selectinload(Article.user_permissions),
        )
    )
    # Permission filtering stays on the event loop: the relationships it reads are
    # eagerly loaded above, so it costs no I/O, and resolving them from a worker thread
    # against an async session would not be safe.
    candidates = [
        (str(article.id), article.title, article.lifecycle_status, article.body_md)
        for article in (await db.execute(stmt)).scalars().all()
        if PermissionService.can_view_article(user, article)
    ]
    if not candidates:
        return []
    return await asyncio.to_thread(_rank, content, candidates)


def classify_similarity(matches: list[dict]) -> str:
    score = matches[0]["score"] if matches else 0.0
    if score >= 0.999:
        return "exact"
    if score >= 0.85:
        return "very_high"
    if score >= MATCH_THRESHOLD:
        return "partial"
    return "none"
