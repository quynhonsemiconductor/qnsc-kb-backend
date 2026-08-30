"""A one-article knowledge base must still fill the context budget.

Asked "what is Verilog" against a knowledge base holding a single Verilog lecture, the
assistant answered that the knowledge base contains no direct definition — and then
listed a handful of disconnected syntax bullets. The document was 16,500 characters and
the prompt budget was 8 parents / 14,000 characters, but the model only ever saw 3.

`RAG_MAX_PARENTS_PER_ARTICLE = 3` is a DIVERSITY guard: it stops one long document
crowding out every other source. Applied flatly it also starves the answer when there is
nothing to diversify across. With one article it is not a guard at all, just a ceiling
two thirds below the budget, and any answer living outside those 3 parents is reported
as absent from a document that contains it.

The cap now scales to how many articles are actually available. The global parent,
character and token budgets are untouched and still bound everything.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.domain.ai_service import _per_article_cap, _select_context


def _parent(article: str, chunk: str, score: float = 0.9, text: str = "content "):
    return {
        "article_id": article,
        "chunk_id": chunk,
        "parent_chunk_id": chunk,
        "parent_text": text * 40,
        "score": score,
    }


# ── the cap itself ────────────────────────────────────────────────────────────


def test_a_single_article_may_fill_the_whole_budget():
    """The reported case."""
    parents = [_parent("only", f"c{i}") for i in range(8)]
    assert _per_article_cap(parents) == settings.RAG_MAX_CONTEXT_PARENTS


def test_many_articles_keep_the_original_cap():
    """Diversity is available, so the guard must behave exactly as before."""
    parents = [_parent(f"a{i}", f"c{i}") for i in range(8)]
    assert _per_article_cap(parents) == settings.RAG_MAX_PARENTS_PER_ARTICLE


def test_the_cap_never_drops_below_the_configured_floor():
    """More articles than budget must not shrink the per-article allowance to zero."""
    parents = [_parent(f"a{i}", f"c{i}") for i in range(50)]
    assert _per_article_cap(parents) >= settings.RAG_MAX_PARENTS_PER_ARTICLE


@pytest.mark.parametrize("articles", [1, 2, 3, 4, 8])
def test_the_even_share_never_exceeds_the_global_parent_budget(articles):
    parents = [
        _parent(f"a{i}", f"c{i}-{j}") for i in range(articles) for j in range(8)
    ]
    assert _per_article_cap(parents) <= settings.RAG_MAX_CONTEXT_PARENTS


# ── end to end through _select_context ────────────────────────────────────────


def test_one_article_now_contributes_more_than_three_parents():
    """The regression, measured where it actually bit."""
    parents = [_parent("only", f"c{i}") for i in range(8)]
    selected = _select_context(parents)
    assert len(selected) > settings.RAG_MAX_PARENTS_PER_ARTICLE


def test_a_diverse_corpus_is_still_spread_across_articles():
    """The guard must not have been removed: no article may take the whole budget when
    other articles are available."""
    parents = [_parent("dominant", f"d{i}", score=0.99) for i in range(8)]
    parents += [_parent(f"other{i}", f"o{i}", score=0.9) for i in range(4)]

    selected = _select_context(parents)
    from_dominant = sum(1 for item in selected if item["article_id"] == "dominant")

    assert from_dominant <= _per_article_cap(parents)
    assert len({item["article_id"] for item in selected}) > 1


def test_the_global_parent_budget_still_bounds_a_single_article():
    parents = [_parent("only", f"c{i}") for i in range(40)]
    assert len(_select_context(parents)) <= settings.RAG_MAX_CONTEXT_PARENTS


def test_low_scoring_parents_are_still_excluded():
    """Relaxing the cap must not let irrelevant context in."""
    below = settings.RAG_MIN_CONTEXT_SCORE - 0.01
    parents = [_parent("only", f"c{i}", score=below) for i in range(8)]
    assert _select_context(parents) == []
