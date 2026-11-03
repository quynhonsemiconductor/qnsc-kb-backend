"""Near-duplicate scoring must be correct, bounded, and off the event loop.

The previous implementation failed all three, and the correctness failure is the one
that matters most because it was silent. `difflib` enables an "autojunk" heuristic on any
sequence longer than 200 elements, discarding every element present in more than 1% of
it. Applied to characters that is every letter in the language, so the match index is
gutted and the score collapses — a 95%-identical pair scored 0.212 at ~1,000 characters,
below the 0.25 threshold, and was therefore reported as not similar at all.

It was also O(n*m) over whole documents (8.25 s for one 20,000-character pair, ~8x per
doubling, `MAX_SOURCE_TEXT_CHARS` admitting 2,000,000) and ran once per article inline in
an async function, blocking the worker's event loop for the duration.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from src.core.config import settings
from src.domain import similarity as similarity_module
from src.domain.similarity import (
    MATCH_THRESHOLD,
    _rank,
    classify_similarity,
    normalize,
    sequence_similarity,
    token_similarity,
)

WORDS = [
    "verilog",
    "module",
    "always",
    "begin",
    "end",
    "wire",
    "reg",
    "clock",
    "reset",
    "assign",
    "testbench",
    "simulation",
    "synthesis",
    "fpga",
    "latch",
    "sequential",
]


def _document(chars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    size = 0
    while size < chars:
        word = rng.choice(WORDS)
        out.append(word)
        size += len(word) + 1
    return " ".join(out)


def _edited(text: str, fraction: float, seed: int = 1) -> str:
    rng = random.Random(seed)
    words = text.split()
    for index in rng.sample(range(len(words)), int(len(words) * fraction)):
        words[index] = rng.choice(WORDS)
    return " ".join(words)


def _candidates(bodies: list[str]) -> list[tuple[str, str, str, str]]:
    return [
        (f"id-{i}", f"Article {i}", "active", body) for i, body in enumerate(bodies)
    ]


# ── correctness ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("length", [300, 1_000, 4_000, 12_000])
def test_a_near_duplicate_scores_high_at_every_document_length(length):
    """The regression that mattered: this scored 0.212 at ~1,000 characters before,
    which is below the threshold, so a 95%-identical upload was silently not flagged."""
    original = _document(length)
    near_duplicate = _edited(original, 0.05)

    score = sequence_similarity(normalize(original), normalize(near_duplicate))

    assert score > 0.85, f"{length} chars scored {score:.3f}"


def test_an_identical_document_scores_one():
    text = normalize(_document(5_000))
    assert sequence_similarity(text, text) == pytest.approx(1.0)


def test_unrelated_documents_stay_below_the_threshold():
    """The fix must not buy recall with false positives."""
    left = normalize(_document(5_000, seed=1))
    right = normalize(" ".join(f"unrelated{n}" for n in range(900)))
    assert (
        max(sequence_similarity(left, right), token_similarity(left, right))
        < MATCH_THRESHOLD
    )


def test_empty_input_scores_zero_rather_than_raising():
    assert sequence_similarity("", "anything") == 0.0
    assert sequence_similarity("anything", "") == 0.0
    assert token_similarity("", "anything") == 0.0


def test_a_near_duplicate_now_reaches_the_confirmation_band():
    """classify_similarity drives requires_update_confirmation, so the collapsed score
    did not merely mis-rank — it changed what the reviewer was asked to do."""
    original = _document(4_000)
    matches = _rank(original, _candidates([_edited(original, 0.05)]))
    assert matches, "a 95%-identical document must be reported"
    assert classify_similarity(matches) in {"very_high", "exact"}


# ── ranking behaviour ─────────────────────────────────────────────────────────


def test_matches_are_ranked_and_capped():
    original = _document(2_000)
    bodies = [
        _edited(original, fraction) for fraction in (0.02, 0.05, 0.10, 0.20, 0.30)
    ]
    bodies += [_document(2_000, seed=99) for _ in range(3)]

    matches = _rank(original, _candidates(bodies))

    assert len(matches) <= 5
    assert [m["score"] for m in matches] == sorted(
        (m["score"] for m in matches), reverse=True
    )
    assert all(m["score"] >= MATCH_THRESHOLD for m in matches)


def test_the_result_shape_is_unchanged():
    original = _document(1_000)
    match = _rank(original, _candidates([_edited(original, 0.02)]))[0]
    assert set(match) == {"article_id", "title", "score", "lifecycle_status"}
    assert match["article_id"] == "id-0"
    assert match["lifecycle_status"] == "active"


def test_no_candidates_yields_no_matches():
    assert _rank(_document(500), []) == []


# ── cost ──────────────────────────────────────────────────────────────────────


def test_a_large_corpus_runs_a_bounded_number_of_comparisons(monkeypatch):
    """Previously this shape was minutes of event-loop-blocking work: 200 articles, each
    compared over its full length. The O(n*m) comparison is now spent only on the
    candidates cheap token overlap ranks highest.

    Counted rather than TIMED. This assertion used to be `elapsed < 30`, which passed on
    CI and failed on a developer machine measuring 48 s for the same code — a wall-clock
    bound tests the hardware, not the algorithm, and its own comment already worried
    about exactly that. The invariant worth pinning is how many times the expensive call
    runs, which is deterministic.
    """
    calls = 0
    real = similarity_module.sequence_similarity

    def counting(left: str, right: str) -> float:
        nonlocal calls
        calls += 1
        return real(left, right)

    monkeypatch.setattr(similarity_module, "sequence_similarity", counting)

    original = _document(2_000)
    bodies = [_edited(original, 0.30, seed=n) for n in range(200)]
    _rank(original, _candidates(bodies))

    assert calls <= settings.SIMILARITY_MAX_SEQUENCE_COMPARISONS, (
        f"{calls} sequence comparisons for 200 candidates; the cap is "
        f"{settings.SIMILARITY_MAX_SEQUENCE_COMPARISONS}"
    )


def test_the_comparison_reads_only_a_bounded_prefix():
    """A difference past the cap cannot change the score — which is what makes the cost
    bound hold regardless of MAX_SOURCE_TEXT_CHARS."""
    head = normalize(_document(settings.SIMILARITY_COMPARE_CHARS * 2))[
        : settings.SIMILARITY_COMPARE_CHARS
    ]
    assert sequence_similarity(head + "a" * 5_000, head + "z" * 5_000) == pytest.approx(
        1.0
    )


def test_scoring_does_not_block_the_event_loop():
    """`find_similar_documents` offloads to a thread. Proven by keeping a 10 ms heartbeat
    running while a deliberately expensive ranking is in flight: inline, the loop would
    starve and the heartbeat would stall."""
    original = _document(20_000)
    candidates = _candidates([_edited(original, 0.30, seed=n) for n in range(60)])

    async def scenario():
        ticks = 0
        stop = False

        async def heartbeat():
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await asyncio.to_thread(_rank, original, candidates)
        stop = True
        beat.cancel()
        return ticks

    assert asyncio.run(scenario()) > 5


# ── the token-score floor ─────────────────────────────────────────────────────


def _counting(monkeypatch) -> list[int]:
    calls = [0]
    real = similarity_module.sequence_similarity

    def counting(left: str, right: str) -> float:
        calls[0] += 1
        return real(left, right)

    monkeypatch.setattr(similarity_module, "sequence_similarity", counting)
    return calls


def _unrelated(chars: int) -> str:
    rng = random.Random(7)
    words: list[str] = []
    while sum(len(w) + 1 for w in words) < chars:
        words.append(f"zq{rng.randrange(400)}")
    return " ".join(words)


def test_a_candidate_with_almost_no_shared_tokens_is_not_compared_character_by_character(
    monkeypatch,
):
    calls = _counting(monkeypatch)
    original = _document(2_000)
    assert len(original.split()) >= settings.SIMILARITY_FLOOR_MIN_TOKENS
    _rank(original, _candidates([_unrelated(2_000) for _ in range(20)]))
    assert calls[0] == 0


def test_a_floor_of_zero_compares_every_top_candidate_as_before(monkeypatch):
    monkeypatch.setattr(settings, "SIMILARITY_SEQUENCE_MIN_TOKEN_SCORE", 0.0)
    calls = _counting(monkeypatch)
    original = _document(2_000)
    _rank(original, _candidates([_unrelated(2_000) for _ in range(20)]))
    assert calls[0] == 20


def test_a_near_duplicate_is_still_compared_and_still_reported(monkeypatch):
    calls = _counting(monkeypatch)
    original = _document(2_000)
    matches = _rank(
        original,
        _candidates([_unrelated(2_000), _edited(original, 0.05), _unrelated(2_000)]),
    )
    assert [m["article_id"] for m in matches] == ["id-1"]
    assert matches[0]["score"] >= 0.9
    assert calls[0] == 1


def test_text_without_spaces_is_still_compared_because_its_token_score_says_nothing(
    monkeypatch,
):
    """One long run of characters is one token: an edited copy has token overlap 0 and must still be found."""
    calls = _counting(monkeypatch)
    rng = random.Random(3)
    alphabet = [chr(code) for code in range(0x4E00, 0x4E00 + 400)]
    original = "".join(rng.choice(alphabet) for _ in range(1_500))
    edited = list(original)
    for index in rng.sample(range(len(edited)), 60):
        edited[index] = rng.choice(alphabet)
    matches = _rank(original, _candidates(["".join(edited)]))
    assert calls[0] == 1
    assert len(matches) == 1 and matches[0]["score"] >= 0.9
