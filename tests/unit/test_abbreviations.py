"""Names a question spells out are found in the company's own vocabulary (src/rag/abbreviations.py)."""

import asyncio
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.domain.search_service import SearchService
from src.rag import abbreviations
from src.rag.abbreviations import VocabularyCache, find_aliases, name_core

VOCAB = {
    "CLKDIV": 1,
    "CLK_EN": 10,
    "INTMAP": 11,
    "INTMAP_001": 1,
    "SYSDBG": 10,
    "WSTRB": 3,
    "PRDATA": 8,
    "PSLVERR": 15,
    "INTEN": 1,
    "INTR_ENABLE": 3,
    "UART0": 11,
    "SPECIFICATION": 19,
    "IEEE": 1,
    "CORE1_": 1,
    "QNSC": 22,
}


@pytest.mark.parametrize(
    "question,expected",
    [
        ("SPI host clock divider field", ["CLKDIV"]),
        ("How do I set the clock divider of the SPI host?", ["CLKDIV"]),
        ("which signal does the clock enable for a block", ["CLK_EN"]),
        ("What does the interrupt map table contain?", ["INTMAP"]),
        ("system debug block overview", ["SYSDBG"]),
        ("the write strobe of the bus", ["WSTRB"]),
        ("peripheral read data signal", ["PRDATA"]),
        ("peripheral slave error response", ["PSLVERR"]),
    ],
)
def test_a_name_the_question_spells_out_is_found(question, expected):
    assert find_aliases(question, VOCAB) == expected


def test_each_word_gives_its_beginning_or_its_consonants():
    assert set(find_aliases("interrupt enable", VOCAB)) == {"INTR_ENABLE", "INTEN"}
    assert find_aliases("clock divider", VOCAB) == ["CLKDIV"], "clk <- clock"


def test_a_name_that_only_looks_alike_is_not_proposed():
    assert find_aliases("interrupt enable", {"IEEE": 1}) == []
    assert find_aliases("clock rate", {"CORE1_": 1}) == []
    assert find_aliases("clock divider", {"QNSC": 22, "SPECIFICATION": 19}) == []


def test_with_two_words_an_initial_alone_needs_a_longer_part_beside_it():
    assert find_aliases("peripheral address", {"PADDR": 5}) == ["PADDR"]
    assert find_aliases("write strobe", {"WSTRB": 5}) == ["WSTRB"]
    assert find_aliases("big dogs", {"BDOGS": 1}) == ["BDOGS"], "b + dogs"
    assert find_aliases("big dog", {"BDOG": 1}) == [], "four letters: too short"


def test_three_words_may_give_initials():
    assert find_aliases("peripheral read data", {"PRDATA": 8}) == ["PRDATA"]


def test_filler_words_never_take_part():
    assert find_aliases("the clock of divider", VOCAB) == []
    assert find_aliases("what is the interrupt", {"WHATISTHE": 1}) == []


@pytest.mark.parametrize(
    "question",
    [
        "INTR_ENABLE offset",
        "What is the CLKDIV clock divider?",
        "Trong QNSC_ROM_MAS, interrupt map nằm ở đâu?",
        "PSLVERR interrupt map",
    ],
)
def test_a_question_that_already_names_something_is_left_alone(question):
    assert find_aliases(question, VOCAB) == []


def test_the_name_of_the_block_asked_about_does_not_stop_the_search_for_the_register():
    assert find_aliases("I2C clock divider", VOCAB) == ["CLKDIV"]
    assert find_aliases("UART0 clock enable", VOCAB) == ["CLK_EN"]
    assert find_aliases("SPI host clock divider field", VOCAB) == ["CLKDIV"]


def test_a_name_shorter_than_five_letters_is_never_proposed():
    assert find_aliases("write strobe", {"STRB": 3, "WSTRB": 3}) == ["WSTRB"]
    assert find_aliases("wrapper apb", {"WRAP": 1}) == []


def test_a_word_of_the_question_is_not_proposed_back():
    assert find_aliases("the specification document", VOCAB) == []
    assert find_aliases("uart base", {"UART0": 11}) == []


def test_spellings_of_one_name_give_the_most_used_one():
    assert find_aliases("interrupt map", VOCAB) == ["INTMAP"]


def test_the_names_the_corpus_leans_on_come_first_and_the_list_is_cut():
    vocab = {"CLKDIV": 1, "CLKDVR": 9, "CLCKDV": 4}

    assert find_aliases("clock divider", vocab, limit=2) == ["CLKDVR", "CLCKDV"]


def test_nothing_in_nothing_out():
    assert find_aliases("", VOCAB) == []
    assert find_aliases("clock divider", {}) == []
    assert find_aliases(None, VOCAB) == []  # type: ignore[arg-type]


def test_a_name_core_has_no_underscore_or_trailing_number():
    assert name_core("CLK_EN") == "clken" and name_core("UART0") == "uart"
    assert name_core("APB_S11") == "apbs"


# -- the cache ------------------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


async def _turns(count=3):
    for _ in range(count):
        await asyncio.sleep(0)


def test_the_cache_answers_none_at_once_and_loads_in_the_background():
    async def scenario():
        cache = VocabularyCache()
        calls = []

        async def loader(company):
            calls.append(company)
            return {"CLKDIV": 1}

        first = cache.get("acme", loader)
        await _turns()
        return first, cache.get("acme", loader), calls

    first, second, calls = _run(scenario())

    assert first is None, "the first question never waits for the load"
    assert second == {"CLKDIV": 1} and calls == ["acme"]


def test_one_load_runs_at_a_time_per_company():
    async def scenario():
        cache = VocabularyCache()
        calls = []
        release = asyncio.Event()

        async def loader(company):
            calls.append(company)
            await release.wait()
            return {}

        for _ in range(5):
            cache.get("acme", loader)
        cache.get("other", loader)
        await _turns()
        release.set()
        await _turns()
        return calls

    assert sorted(_run(scenario())) == ["acme", "other"]


def test_an_old_vocabulary_is_still_served_while_it_is_refreshed():
    async def scenario():
        cache = VocabularyCache(ttl_seconds=-1)
        answers = iter([{"OLD": 1}, {"NEW": 1}])

        async def loader(company):
            return next(answers)

        cache.get("acme", loader)
        await _turns()
        stale = cache.get("acme", loader)
        await _turns()
        return stale, cache.get("acme", loader)

    stale, fresh = _run(scenario())

    assert stale == {"OLD": 1} and fresh == {"NEW": 1}


def test_a_failed_load_is_not_retried_straight_away_and_costs_no_search():
    async def scenario():
        cache = VocabularyCache(retry_seconds=60)
        calls = []

        async def loader(company):
            calls.append(1)
            raise RuntimeError("database down")

        answers = []
        for _ in range(4):
            answers.append(cache.get("acme", loader))
            await _turns()
        return answers, calls

    answers, calls = _run(scenario())

    assert answers == [None, None, None, None] and len(calls) == 1


def test_outside_an_event_loop_nothing_is_started():
    cache = VocabularyCache()

    async def loader(company):  # pragma: no cover - must not run
        raise AssertionError("no loop, no load")

    assert cache.get("acme", loader) is None
    assert cache._running == set()


def test_a_company_can_be_made_to_reload():
    async def scenario():
        cache = VocabularyCache()

        async def loader(company):
            return {"A": 1}

        cache.get("acme", loader)
        await _turns()
        cache.forget("acme")
        return cache.get("acme", loader)

    assert _run(scenario()) is None


# -- in a search ----------------------------------------------------------------------------------


class _Cache:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.asked = []

    def get(self, company, loader):
        self.asked.append((company, loader))
        return self.vocabulary


def test_the_setting_off_means_no_names_and_no_cache_use(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_ABBREVIATION_EXPANSION_ENABLED", False)
    cache = _Cache({"CLKDIV": 1})
    monkeypatch.setattr("src.domain.search_service.vocabulary_cache", cache)

    user = SimpleNamespace(company_domain="acme.test")

    assert SearchService._alias_names(user, "clock divider", "") == []
    assert cache.asked == []


def test_with_the_setting_on_the_vocabulary_of_the_askers_company_is_used(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_ABBREVIATION_EXPANSION_ENABLED", True)
    cache = _Cache({"CLKDIV": 1})
    monkeypatch.setattr("src.domain.search_service.vocabulary_cache", cache)
    user = SimpleNamespace(company_domain="acme.test")

    names = SearchService._alias_names(user, "SPI clock divider", "")

    assert names == ["CLKDIV"]
    assert cache.asked[0][0] == "acme.test"
    assert cache.asked[0][1] is abbreviations.load_company_vocabulary


def test_the_translation_of_a_question_is_read_for_names_too(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_ABBREVIATION_EXPANSION_ENABLED", True)
    monkeypatch.setattr(
        "src.domain.search_service.vocabulary_cache", _Cache({"CLKDIV": 1})
    )
    user = SimpleNamespace(company_domain="acme.test")

    names = SearchService._alias_names(
        user, "bộ chia xung nhịp của SPI", "SPI clock divider"
    )

    assert names == ["CLKDIV"]


def test_no_vocabulary_yet_or_no_company_means_no_names(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_ABBREVIATION_EXPANSION_ENABLED", True)
    monkeypatch.setattr("src.domain.search_service.vocabulary_cache", _Cache(None))

    with_company = SimpleNamespace(company_domain="acme.test")
    assert SearchService._alias_names(with_company, "clock divider", "") == []
    assert SearchService._alias_names(SimpleNamespace(), "clock divider", "") == []


# -- the extended search only adds, it does not replace -----------------------------------------------


def _hit(name):
    return {"chunk_id": name, "title": name}


def test_the_first_ordinary_results_keep_their_place_and_new_finds_follow_them():
    ordinary = [_hit(x) for x in "ABCDE"]
    extended = [_hit(x) for x in "XYAZBC"]

    merged = abbreviations.merge_alias_results(ordinary, extended)

    assert [h["chunk_id"] for h in merged] == list("ABXYZCDE")


def test_a_wrong_name_costs_the_answer_at_most_the_places_inserted():
    ordinary = [_hit("answer")] + [_hit(f"o{i}") for i in range(5)]
    extended = [_hit(f"n{i}") for i in range(10)] + [_hit("answer")]

    merged = [
        h["chunk_id"] for h in abbreviations.merge_alias_results(ordinary, extended)
    ]

    assert merged.index("answer") == 0
    ordinary = [_hit("o0"), _hit("o1"), _hit("answer")]
    merged = [
        h["chunk_id"] for h in abbreviations.merge_alias_results(ordinary, extended)
    ]
    assert merged.index("answer") <= 2 + abbreviations.INSERT_EXTENDED


def test_when_the_extended_search_finds_nothing_new_the_order_is_the_ordinary_one():
    ordinary = [_hit(x) for x in "ABCD"]

    merged = abbreviations.merge_alias_results(ordinary, [_hit("C"), _hit("A")])

    assert [h["chunk_id"] for h in merged] == list("ABCD")


def test_one_side_empty_is_the_other_side():
    assert abbreviations.merge_alias_results([], [_hit("A")]) == [_hit("A")]
    assert abbreviations.merge_alias_results([_hit("A")], []) == [_hit("A")]


def _search_with(monkeypatch, found, passes):
    async def fake_pass(self, user, query, filters, limit, apply_aliases):
        passes.append(apply_aliases)
        if not apply_aliases:
            self._found_aliases = found
            return [_hit("A"), _hit("B"), _hit("C")]
        return [_hit("N"), _hit("A")]

    monkeypatch.setattr(SearchService, "_search_pass", fake_pass)
    service = SearchService.__new__(SearchService)
    return _run(service.search(SimpleNamespace(), "q", None, 3))


def test_a_question_with_no_name_is_searched_once(monkeypatch):
    passes = []

    results = _search_with(monkeypatch, [], passes)

    assert passes == [False] and [h["chunk_id"] for h in results] == ["A", "B", "C"]


def test_a_question_with_a_name_is_searched_twice_and_merged(monkeypatch):
    passes = []

    results = _search_with(monkeypatch, ["CLKDIV"], passes)

    assert passes == [False, True]
    assert [h["chunk_id"] for h in results] == ["A", "B", "N"], "cut to the limit"
