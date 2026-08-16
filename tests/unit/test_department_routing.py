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
