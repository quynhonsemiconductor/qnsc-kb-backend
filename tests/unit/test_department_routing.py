from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.api.routers.auth import DepartmentInput
from src.domain.department_routing import route_document_candidates, suggest_departments


def test_department_creation_requires_a_short_description():
    with pytest.raises(ValidationError):
        DepartmentInput(name="Engineering")

    assert DepartmentInput(
        name="Engineering",
        description="Product engineering standards and release operations.",
    ).description.startswith("Product engineering")


def test_department_suggestion_uses_department_description():
    departments = [
        SimpleNamespace(
            id="engineering-id",
            name="Engineering",
            description="Product engineering standards, software releases, and technical operations.",
        ),
        SimpleNamespace(
            id="people-id",
            name="People",
            description="Hiring, employee benefits, and workplace policies.",
        ),
    ]

    selected, suggestions, proposed = suggest_departments(
        "Release process",
        "This document explains software release standards and engineering operations.",
        departments,
    )

    assert selected == ["engineering-id"]
    assert suggestions[0]["name"] == "Engineering"
    assert proposed is None


def test_department_suggestion_proposes_a_new_department_when_no_description_matches():
    selected, suggestions, proposed = suggest_departments(
        "Laboratory safety",
        "Chemical storage, safety equipment, and laboratory procedures.",
        [
            SimpleNamespace(
                id="people-id",
                name="People",
                description="Hiring and employee benefits.",
            )
        ],
    )

    assert selected == []
    assert suggestions == []
    assert proposed and proposed["name"] == "Laboratory safety"


def test_routing_splits_only_when_the_owning_department_changes():
    departments = [
        SimpleNamespace(
            id="engineering-id",
            name="Engineering",
            description="Software release standards and technical operations.",
        ),
        SimpleNamespace(
            id="people-id",
            name="People",
            description="Hiring, employee benefits, and workplace policies.",
        ),
    ]
    markdown = """# Company handbook

## Release process
Engineering release standards and technical operations.

### Rollback
Engineering release operations during rollback.

## Benefits
Employee benefits and workplace policies.
"""

    candidates = route_document_candidates("Company handbook", markdown, departments)

    assert len(candidates) == 2
    assert candidates[0]["department_ids"] == ["engineering-id"]
    assert "### Rollback" in candidates[0]["body_md"]
    assert candidates[1]["department_ids"] == ["people-id"]


def _two_departments() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id="engineering-id",
            name="Engineering",
            description="Software release standards and technical operations.",
        ),
        SimpleNamespace(
            id="people-id",
            name="People",
            description="Hiring, employee benefits, and workplace policies.",
        ),
    ]


def test_a_heading_with_nothing_under_it_is_not_a_candidate():
    """``## Overview`` directly followed by ``## Scope`` used to emit ``## Overview``.

    A section boundary is unconditional on size, so a heading with no body became a
    candidate whose entire content was its own heading. That is not a reviewable article.
    It is also not rare: a source over RESTRUCTURE_SECTION_CHARS is formatted in parts by
    separate LLM calls that never see one another, then concatenated, so adjacent headings
    appear at every seam.
    """
    markdown = """# Company handbook

## Overview

## Release process
Engineering release standards and technical operations.
"""

    candidates = route_document_candidates("Company handbook", markdown, _two_departments())

    assert len(candidates) == 1
    # Absorbed rather than dropped: the heading is still in the reviewer's text.
    assert "## Overview" in candidates[0]["body_md"]
    assert candidates[0]["department_ids"] == ["engineering-id"]


def test_a_trailing_heading_is_absorbed_by_the_section_before_it():
    markdown = """## Release process
Engineering release standards and technical operations.

## Appendix
"""

    candidates = route_document_candidates("Company handbook", markdown, _two_departments())

    assert len(candidates) == 1
    assert candidates[0]["body_md"].rstrip().endswith("## Appendix")


def test_front_matter_stays_with_the_first_section_it_introduces():
    """A version line must not become a candidate no department owns.

    The preamble guard used to ask whether the buffer held any non-H1 content, which got
    this backwards: a bare ``# Handbook`` was held correctly, but adding ``Version 1.0``
    made it a standalone unrouted candidate.
    """
    markdown = """# Company handbook
Version 2.0

## Release process
Engineering release standards and technical operations.

## Benefits
Employee benefits and workplace policies.
"""

    candidates = route_document_candidates("Company handbook", markdown, _two_departments())

    # Front matter merges forward, and the genuine owner change still splits.
    assert len(candidates) == 2
    assert "Version 2.0" in candidates[0]["body_md"]
    assert candidates[0]["department_ids"] == ["engineering-id"]
    assert candidates[1]["department_ids"] == ["people-id"]


def test_short_sections_with_real_content_still_split_on_owner_change():
    """The rule is 'no content of its own', NOT a character floor.

    Both sections here are barely 50 characters, and both must survive as separate
    candidates: they say something real and they belong to different departments.
    """
    markdown = """## Release
Engineering release operations.

## Benefits
Employee benefits policies.
"""

    candidates = route_document_candidates("Company handbook", markdown, _two_departments())

    assert len(candidates) == 2
    assert [item["department_ids"] for item in candidates] == [
        ["engineering-id"],
        ["people-id"],
    ]
