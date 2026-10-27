"""A text page with a white backdrop and a hairline is not a picture.

`page.get_drawings()` is non-empty for nearly every page a word processor produced: a white
rectangle behind the text block, a faint rule under the heading. Any drawing used to make the
page "Visual evidence on page N" -- 76 of the 127 pages of a plain text book. The shapes
below are the real ones from that book's page 77.

`page_has_vector_artwork` is pure, so no PDF engine is needed.
"""

from __future__ import annotations

from src.domain.evidence_extraction import page_has_vector_artwork

W, H = 595.0, 842.0
WHITE = (1.0, 1.0, 1.0)
PAGE = {"page_width": W, "page_height": H}


def _artwork(drawings):
    return page_has_vector_artwork(drawings, **PAGE)


def _d(rect, fill=None, color=None):
    return {"rect": rect, "fill": fill, "color": color}


def test_the_text_page_that_was_wrongly_cited_has_no_artwork():
    drawings = [
        _d((70.5, 70.5, 524.2, 771.5), fill=WHITE),
        _d((71.0, 97.3, 523.8, 97.8), fill=(0.918, 0.926, 0.938)),
    ]
    assert _artwork(drawings) is False


def test_a_page_with_no_drawings_has_none():
    assert _artwork([]) is False


def test_a_figure_page_whose_frames_are_white_has_none():
    drawings = [_d((124.1, 70.5, 470.6, 265.0), fill=WHITE)]
    assert _artwork(drawings) is False


def test_a_coloured_chart_area_is_artwork():
    drawings = [_d((100.0, 200.0, 500.0, 500.0), fill=(0.2, 0.4, 0.8))]
    assert _artwork(drawings) is True


def test_a_stroked_outline_is_artwork_even_without_fill():
    drawings = [_d((100.0, 200.0, 500.0, 500.0), color=(0.0, 0.0, 0.0))]
    assert _artwork(drawings) is True


def test_a_background_wash_is_not_artwork():
    drawings = [_d((0.0, 0.0, W, H), fill=(0.9, 0.9, 0.7))]
    assert _artwork(drawings) is False


def test_rules_and_underlines_are_not_artwork():
    drawings = [
        _d((50.0, 100.0 + i * 20, 545.0, 100.5 + i * 20), color=(0, 0, 0))
        for i in range(20)
    ]
    assert _artwork(drawings) is False


def test_many_small_bars_add_up_to_a_chart():
    drawings = [
        _d(
            (100.0 + i * 40, 600.0 - 40 * (i + 1), 125.0 + i * 40, 600.0),
            fill=(0.1, 0.5, 0.2),
        )
        for i in range(8)
    ]
    assert _artwork(drawings) is True


def test_a_drawing_without_a_rect_is_ignored():
    assert _artwork([{"fill": (0, 0, 0)}]) is False


def test_a_zero_sized_page_is_not_artwork():
    assert (
        page_has_vector_artwork(
            [_d((0, 0, 10, 10), fill=(0, 0, 0))], page_width=0, page_height=0
        )
        is False
    )


def test_drawing_outside_the_page_is_clipped():
    assert _artwork([_d((700.0, 900.0, 900.0, 1100.0), fill=(0, 0, 0))]) is False
