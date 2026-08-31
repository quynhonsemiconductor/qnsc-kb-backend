"""Official-style SQuAD / MLQA answer metrics: Exact Match and token-level F1.

WHY THIS IS SEPARATE FROM `evaluator.py`. The metrics there are set-overlap
proxies for a live dashboard: `answer_correctness` is recall-only (it never
penalises a long answer that happens to contain the expected terms) and it drops
tokens of 3 characters or fewer, which in Vietnamese deletes real content words
("ba", "mẹ", "vua", "hồ"). Neither property is acceptable for a benchmark gate,
so MLQA/ViQuAD scoring lives here and the dashboard metrics are left alone.

NORMALISATION follows the SQuAD v1.1 script (lowercase, strip punctuation, strip
articles, collapse whitespace) with the MLQA multilingual amendment: the article
list is per-language, and Vietnamese has no articles to strip. Diacritics are
PRESERVED -- folding them would make "hoà" and "hoa" score as a match, and the
answer span really is one or the other.

Tokenisation is whitespace-plus-punctuation over Unicode word characters, which
matches the official MLQA scorer's behaviour for Vietnamese (it treats syllables
as tokens; Vietnamese words are multi-syllable, but both prediction and reference
are tokenised the same way, so the comparison stays fair).

A question with several reference answers scores as the MAX over references,
which is what both official scripts do. For unanswerable questions (ViQuAD 2.0),
the reference is the empty string: an abstaining prediction scores 1.0 and any
non-empty prediction scores 0.0.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

# SQuAD strips English articles. MLQA extends this per language; only the two
# languages this system targets are listed, and Vietnamese contributes none --
# "một" is a numeral/classifier, not an article, and removing it changes meaning.
_ARTICLES = {
    "en": {"a", "an", "the"},
    "vi": set(),
}

_PUNCT = set(string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_punctuation(text: str) -> str:
    # Unicode punctuation too, not just ASCII: Vietnamese sources carry curly
    # quotes and en dashes that would otherwise fuse into adjacent tokens.
    return "".join(
        character
        for character in text
        if character not in _PUNCT and not unicodedata.category(character).startswith("P")
    )


def normalize_answer(text: str, lang: str = "en") -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    without_punctuation = _strip_punctuation(lowered)
    articles = _ARTICLES.get(lang, _ARTICLES["en"])
    tokens = [token for token in _WHITESPACE_RE.split(without_punctuation) if token]
    if articles:
        tokens = [token for token in tokens if token not in articles]
    return " ".join(tokens)


def get_tokens(text: str, lang: str = "en") -> list[str]:
    normalized = normalize_answer(text, lang)
    return normalized.split() if normalized else []


def exact_match(prediction: str, reference: str, lang: str = "en") -> float:
    return float(normalize_answer(prediction, lang) == normalize_answer(reference, lang))


def token_f1(prediction: str, reference: str, lang: str = "en") -> float:
    """Harmonic mean of token precision and recall, SQuAD-style."""
    predicted_tokens = get_tokens(prediction, lang)
    reference_tokens = get_tokens(reference, lang)

    # Both empty means an abstention that was supposed to abstain. Exactly one
    # empty means a disagreement no overlap can rescue.
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_over_references(
    metric, prediction: str, references: list[str], lang: str
) -> float:
    if not references:
        # Unanswerable: the only correct prediction is no prediction.
        return float(not normalize_answer(prediction, lang))
    return max(metric(prediction, reference, lang) for reference in references)


def score_answer(
    prediction: str, references: list[str], lang: str = "en"
) -> tuple[float, float]:
    """Return ``(exact_match, token_f1)`` maximised over reference answers."""
    return (
        _best_over_references(exact_match, prediction, references, lang),
        _best_over_references(token_f1, prediction, references, lang),
    )


def aggregate(scores: list[tuple[float, float]]) -> dict[str, float]:
    """Mean EM and F1 as percentages, matching how both benchmarks report."""
    if not scores:
        return {"exact_match": 0.0, "f1": 0.0, "count": 0}
    return {
        "exact_match": 100.0 * sum(score[0] for score in scores) / len(scores),
        "f1": 100.0 * sum(score[1] for score in scores) / len(scores),
        "count": len(scores),
    }
