"""A scanned page is a picture of a sheet, not a figure on it.

A scanner produces a PDF page that is one image covering the sheet. The figure selector
only rejects icons and template artwork, so that image passed every test and each scanned
page became "Figure 1 on page N" -- a whole page of text presented as a diagram. The text is
still read by OCR; the page just must not be cited as a figure.

`page_is_image_scan` is pure so it is tested without a PDF engine (the test venv has no
pymupdf), like `select_page_figures`.
"""

from __future__ import annotations

from src.core.config import settings
from src.domain.evidence_extraction import page_is_image_scan, select_page_figures

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
PAGE = {"page_width": PAGE_WIDTH, "page_height": PAGE_HEIGHT}


def _scan(**kwargs):
    return page_is_image_scan(kwargs.pop("placements"), **PAGE, **kwargs)


def test_an_image_filling_the_page_is_a_scan():
    assert _scan(placements=[(5, (0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT))])


def test_a_scan_with_a_scanner_margin_is_still_a_scan():
    """Scanners leave a border, so an exact 100% would miss most real scans."""
    assert _scan(placements=[(5, (12.0, 14.0, PAGE_WIDTH - 12.0, PAGE_HEIGHT - 14.0))])


def test_a_large_figure_is_not_a_scan():
    """A figure taking half the page is the very thing that should stay a figure."""
    assert not _scan(placements=[(5, (50.0, 80.0, 560.0, 480.0))])


def test_a_page_with_no_images_is_not_a_scan():
    assert not _scan(placements=[])


def test_a_page_scanned_in_strips_is_a_scan():
    """No single strip is 85% of the page, together they are the page."""
    third = PAGE_HEIGHT / 3
    strips = [
        (1, (0.0, 0.0, PAGE_WIDTH, third)),
        (2, (0.0, third, PAGE_WIDTH, 2 * third)),
        (3, (0.0, 2 * third, PAGE_WIDTH, PAGE_HEIGHT)),
    ]

    assert _scan(placements=strips)


def test_the_same_placement_listed_twice_is_not_counted_twice():
    """PyMuPDF can report one placement more than once; a half-page figure must not
    double into a scan."""
    half = (0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT * 0.5)

    assert not _scan(placements=[(5, half), (5, half)])


def test_an_image_hanging_off_the_page_counts_only_where_it_lands():
    """A placement bigger than the page is clipped to it; the overhang is not coverage."""
    assert not _scan(placements=[(5, (-2000.0, -2000.0, 100.0, 100.0))])


def test_a_degenerate_page_is_not_a_scan():
    assert not page_is_image_scan(
        [(5, (0.0, 0.0, 10.0, 10.0))], page_width=0.0, page_height=0.0
    )


def test_the_threshold_is_the_configured_one(monkeypatch):
    seventy_percent = [(5, (0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT * 0.7))]
    assert not _scan(placements=seventy_percent)

    monkeypatch.setattr(settings, "EVIDENCE_SCAN_PAGE_COVERAGE", 0.6)
    assert _scan(placements=seventy_percent)


def test_a_scan_would_otherwise_have_passed_as_a_figure():
    """Pins the reason this rule exists: the figure selector alone keeps a full-page image."""
    placement = [(5, (0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT))]

    kept = select_page_figures(placement, page_count=20, **PAGE)

    assert len(kept) == 1
    assert _scan(placements=placement)
