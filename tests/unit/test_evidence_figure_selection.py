"""Visual evidence must be the cited figure, not the page it happened to sit on.

`page_visual` used to be emitted for ANY page holding an image or a drawing, with the
whole page as its bounding box. A citation therefore showed the figure a reviewer asked
about surrounded by unrelated content from the same page, and `figure` -- which the
schema has always allowed -- was never produced at all.

Cropping to the image makes "is this image content?" a question that has to be answered,
because a logo is an image too. The thresholds come from the PDFs in this workspace:
718 image placements, median 1.63% of their page, 383 of them under 2%, because template
artwork repeats on every page. One deck places the same image on all 149 of its pages.

`select_page_figures` is pure so that this judgement can be tested without a PDF engine
(the test venv has no pymupdf), and so the rule is readable on its own.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from src.domain.evidence_extraction import (
    EXTRACTION_VERSION,
    figure_caption_text,
    select_page_figures,
)

# A4-ish, in points.
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
PAGE = {"page_width": PAGE_WIDTH, "page_height": PAGE_HEIGHT, "page_count": 20}


def _rect(fraction: float, *, top: float = 100.0, left: float = 100.0):
    """A square-ish rect covering `fraction` of the page, large enough on both sides."""
    area = PAGE_WIDTH * PAGE_HEIGHT * fraction
    side = area**0.5
    return (left, top, left + side, top + side)


# --- what counts as a figure ------------------------------------------------


def test_a_large_image_becomes_a_figure():
    kept = select_page_figures([(7, _rect(0.30))], **PAGE)

    assert [xref for xref, _rect in kept] == [7]


def test_a_decorative_icon_is_rejected_on_area():
    """The median placement in this corpus is 1.63% of its page: template artwork."""
    assert select_page_figures([(7, _rect(0.016))], **PAGE) == []


def test_the_area_threshold_is_the_configured_one(monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_FIGURE_MIN_PAGE_FRACTION", 0.10)

    assert select_page_figures([(7, _rect(0.05))], **PAGE) == []
    assert len(select_page_figures([(7, _rect(0.20))], **PAGE)) == 1


def test_a_thin_full_width_rule_is_rejected_despite_its_area():
    """A full-width divider clears the area bar while being 25pt tall.

    This is why area alone is not enough: stretched across a wide page, a band too
    short to hold anything still covers 3% of it.
    """
    height = 25.0
    rule = (0.0, 400.0, PAGE_WIDTH, 400.0 + height)
    fraction = (PAGE_WIDTH * height) / (PAGE_WIDTH * PAGE_HEIGHT)
    assert fraction > settings.EVIDENCE_FIGURE_MIN_PAGE_FRACTION, "precondition"
    assert height < settings.EVIDENCE_FIGURE_MIN_POINTS, "precondition"

    assert select_page_figures([(7, rule)], **PAGE) == []


def test_a_zero_area_placement_is_ignored():
    assert select_page_figures([(7, (10.0, 10.0, 10.0, 300.0))], **PAGE) == []


def test_an_inverted_rect_is_normalised_rather_than_dropped():
    """PyMuPDF reports y downward, but a transformed placement can still arrive with its
    corners swapped, and a negative width would silently fail every size check."""
    kept = select_page_figures([(7, (400.0, 500.0, 100.0, 100.0))], **PAGE)

    assert len(kept) == 1
    x0, y0, x1, y1 = kept[0][1]
    assert (x0, y0, x1, y1) == (100.0, 100.0, 400.0, 500.0)


# --- the template logo, which is the sharpest signal ------------------------


def test_an_image_repeated_across_the_document_is_template_not_evidence():
    """The case that motivated this: one image on all 149 pages of a deck."""
    kept = select_page_figures(
        [(14, _rect(0.30))], **PAGE, xref_page_counts={14: 20}
    )

    assert kept == []


def test_a_figure_appearing_on_a_couple_of_pages_is_still_a_figure():
    """A diagram reused in two sections is not a letterhead."""
    kept = select_page_figures(
        [(14, _rect(0.30))], **PAGE, xref_page_counts={14: 2}
    )

    assert len(kept) == 1


def test_the_template_rule_needs_a_document_long_enough_to_judge():
    """On a 3-page document, "on 25% of pages" is one page and means nothing."""
    short = {"page_width": PAGE_WIDTH, "page_height": PAGE_HEIGHT, "page_count": 3}

    kept = select_page_figures([(14, _rect(0.30))], **short, xref_page_counts={14: 2})

    assert len(kept) == 1


def test_a_single_page_document_never_applies_the_template_rule():
    one = {"page_width": PAGE_WIDTH, "page_height": PAGE_HEIGHT, "page_count": 1}

    kept = select_page_figures([(14, _rect(0.30))], **one, xref_page_counts={14: 1})

    assert len(kept) == 1


# --- repeated placements of one image ---------------------------------------


def test_repeats_of_one_xref_are_unioned_into_a_single_figure():
    """Two rects for one xref is a tiled or double-drawn placement of the same artwork,
    not two figures. Emitting both would show a reviewer the same picture twice."""
    kept = select_page_figures(
        [
            (7, (100.0, 100.0, 300.0, 300.0)),
            (7, (250.0, 200.0, 500.0, 450.0)),
        ],
        **PAGE,
    )

    assert len(kept) == 1
    assert kept[0][1] == (100.0, 100.0, 500.0, 450.0)


def test_two_small_placements_can_union_into_something_citable():
    """A figure split across two placements is one figure, and its union is what the
    reviewer should see."""
    half = _rect(0.012, top=100.0)
    other = _rect(0.012, top=100.0 + (half[3] - half[1]))

    assert select_page_figures([(7, half)], **PAGE) == [], "each half is too small alone"
    assert len(select_page_figures([(7, half), (7, other)], **PAGE)) == 1


def test_distinct_images_stay_distinct():
    kept = select_page_figures(
        [(7, _rect(0.30, top=50.0)), (9, _rect(0.30, top=450.0))], **PAGE
    )

    assert [xref for xref, _r in kept] == [7, 9]


# --- ordering ---------------------------------------------------------------


def test_figures_are_returned_in_reading_order():
    """So "Figure 1" is the top-most figure rather than whichever xref the PDF listed
    first -- the numbering a reviewer sees has to match the page."""
    lower = _rect(0.20, top=600.0, left=50.0)
    upper = _rect(0.20, top=80.0, left=50.0)

    kept = select_page_figures([(9, lower), (7, upper)], **PAGE)

    assert [xref for xref, _r in kept] == [7, 9]


def test_figures_on_the_same_row_order_left_to_right():
    left = _rect(0.10, top=100.0, left=40.0)
    right = _rect(0.10, top=100.0, left=350.0)

    kept = select_page_figures([(9, right), (7, left)], **PAGE)

    assert [xref for xref, _r in kept] == [7, 9]


# --- degenerate input -------------------------------------------------------


@pytest.mark.parametrize(
    "page_width,page_height",
    [(0.0, 792.0), (612.0, 0.0), (-1.0, 792.0)],
)
def test_a_page_with_no_area_yields_nothing(page_width, page_height):
    assert (
        select_page_figures(
            [(7, _rect(0.30))],
            page_width=page_width,
            page_height=page_height,
            page_count=10,
        )
        == []
    )


def test_no_placements_yields_nothing():
    assert select_page_figures([], **PAGE) == []


# --- captions, which are what make a cropped figure findable ---------------


def test_a_caption_below_the_figure_is_collected():
    """Cropping removes the words, so without this a figure is unsearchable."""
    figure = (100.0, 100.0, 400.0, 300.0)
    lines = [
        ((100.0, 310.0, 400.0, 325.0), "Figure 2: quarterly revenue by region"),
        ((100.0, 700.0, 400.0, 715.0), "Unrelated paragraph far below"),
    ]

    caption = figure_caption_text(lines, figure, page_height=PAGE_HEIGHT)

    assert caption == "Figure 2: quarterly revenue by region"


def test_a_heading_above_the_figure_is_collected():
    figure = (100.0, 300.0, 400.0, 500.0)
    lines = [((100.0, 270.0, 400.0, 288.0), "Revenue breakdown")]

    assert (
        figure_caption_text(lines, figure, page_height=PAGE_HEIGHT)
        == "Revenue breakdown"
    )


def test_labels_inside_the_figure_are_collected():
    """A diagram's own labels describe it better than anything around it."""
    figure = (100.0, 100.0, 400.0, 300.0)
    lines = [((150.0, 150.0, 250.0, 165.0), "Input layer")]

    assert figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == "Input layer"


def test_text_in_a_different_column_is_not_claimed_as_a_caption():
    """Horizontal overlap is required, or a two-column page gives every figure the
    neighbouring column's prose."""
    figure = (50.0, 100.0, 280.0, 300.0)
    lines = [((330.0, 150.0, 560.0, 165.0), "Other column text")]

    assert figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == ""


def test_caption_lines_are_ordered_top_to_bottom():
    figure = (100.0, 300.0, 400.0, 400.0)
    lines = [
        ((100.0, 410.0, 400.0, 425.0), "second"),
        ((100.0, 275.0, 400.0, 290.0), "first"),
    ]

    assert figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == "first second"


def test_blank_and_whitespace_lines_are_skipped():
    figure = (100.0, 100.0, 400.0, 300.0)
    lines = [
        ((100.0, 310.0, 400.0, 325.0), "   "),
        ((100.0, 330.0, 400.0, 345.0), ""),
        ((100.0, 305.0, 400.0, 318.0), "Real  caption\ntext"),
    ]

    assert (
        figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == "Real caption text"
    )


def test_the_caption_band_is_the_configured_fraction(monkeypatch):
    figure = (100.0, 100.0, 400.0, 300.0)
    lines = [((100.0, 500.0, 400.0, 515.0), "Far below")]

    monkeypatch.setattr(settings, "EVIDENCE_FIGURE_CAPTION_MARGIN", 0.01)
    assert figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == ""

    monkeypatch.setattr(settings, "EVIDENCE_FIGURE_CAPTION_MARGIN", 0.40)
    assert figure_caption_text(lines, figure, page_height=PAGE_HEIGHT) == "Far below"


# --- the stored rows -------------------------------------------------------


def test_the_extraction_version_moved_so_old_rows_are_identifiable():
    """v1 rows hold a page-sized box. They are replaced by a reindex, not upgraded in
    place, so the version has to distinguish them."""
    assert EXTRACTION_VERSION == "evidence-v2"
