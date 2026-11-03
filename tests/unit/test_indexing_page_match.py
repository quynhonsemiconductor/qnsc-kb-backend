"""Page attribution must stay correct now that pages are tokenised once per document.

_match_source_page is called for every parent AND every child chunk. It used to
re-tokenise, re-set and re-join every extracted page on each of those calls, which is
O(chunks x pages) passes over the whole document — ~104 ms per call across 120 pages,
so roughly 80 seconds of pure re-tokenisation for an 800-chunk file. The tokens are now
prepared once. These tests pin the BEHAVIOUR that refactor had to preserve.
"""
from __future__ import annotations

from src.domain.indexing import (
    _match_source_page,
    _normalized_tokens,
    _prepare_source_pages,
)


def _pages() -> list[tuple[int, str]]:
    return [
        (1, "Quy dinh chung ve an toan lao dong trong nha may san xuat"),
        (2, "Chinh sach nghi phep hang nam va thu tuc de nghi nghi phep"),
        (3, ""),
        (4, "      "),
        (5, "Bang luong va phu cap cho ky su van hanh thiet bi"),
    ]


def test_a_chunk_is_attributed_to_the_page_it_came_from():
    prepared = _prepare_source_pages(_pages())

    assert _match_source_page("thu tuc de nghi nghi phep hang nam", prepared) == 2
    assert _match_source_page("phu cap cho ky su van hanh thiet bi", prepared) == 5


def test_pages_without_text_are_skipped_not_scored():
    """An empty page must never win, and must not be prepared at all."""
    prepared = _prepare_source_pages(_pages())

    assert [page_number for page_number, _tokens, _text in prepared] == [1, 2, 5]
    assert _match_source_page("khong lien quan gi ca xyz", prepared) is None


def test_unrelated_text_stays_unattributed():
    prepared = _prepare_source_pages(_pages())

    assert _match_source_page("zzz qqq wwwww", prepared) is None
    assert _match_source_page("", prepared) is None


def test_no_pages_at_all_is_not_an_error():
    assert _match_source_page("bat ky noi dung nao", _prepare_source_pages([])) is None
    assert _match_source_page("bat ky noi dung nao", _prepare_source_pages(None)) is None


def test_a_long_exact_run_outranks_scattered_token_overlap():
    """The exact-run bonus is what keeps a verbatim passage on its own page."""
    run = "dieu khoan bao mat thong tin khach hang phai duoc tuan thu nghiem ngat"
    pages = [
        # Page 1 shares many individual tokens but not the run.
        (1, "thong tin khach hang bao mat dieu khoan tuan thu nghiem ngat rieng le"),
        (2, f"phan mo dau {run} phan ket luan"),
    ]
    prepared = _prepare_source_pages(pages)

    assert len(" ".join(_normalized_tokens(run)[:24])) >= 24, "the case must exercise the bonus"
    assert _match_source_page(run, prepared) == 2


def test_preparation_is_reusable_across_many_chunks():
    """The whole point: one prepared index, many lookups, same answers every time."""
    prepared = _prepare_source_pages(_pages())
    chunk = "thu tuc de nghi nghi phep hang nam"

    assert {_match_source_page(chunk, prepared) for _ in range(50)} == {2}
