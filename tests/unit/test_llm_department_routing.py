"""The LLM decides the department; keyword ranking is what happens when it cannot.

Routing scored a document against each department's name and description by literal
token overlap. `Department.description` defaults to "", so on a tenant where nobody has
written them every score is zero: nothing routes, and every document instead proposes
creating a new department named after its own filename. Measured against the corpus that
prompted this, a set of EDA lecture PDFs, that is exactly what happened -- "Lecture-5-STA"
proposed a department called "Lecture-5-STA".

An LLM now decides when one is configured. The keyword ranking is unchanged and still
runs for every section: it supplies the reviewer's alternatives and the new-department
proposal, and it is the whole answer whenever the model declines, errors, or is off.

The cases below are mostly about the model behaving badly, because that is the part that
must not reach a user. A routing suggestion is something a reviewer sees and edits; it
must never be able to fail a document import.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import settings
from src.domain.department_routing import (
    route_document_candidates,
    route_document_candidates_llm,
)


class _Department:
    def __init__(self, identifier: str, name: str, description: str = ""):
        self.id = identifier
        self.name = name
        self.description = description


#: Deliberately description-less: the state that made keyword routing useless.
UNDESCRIBED = [_Department("phys", "Physical Design"), _Department("dv", "Verification")]

MARKDOWN = (
    "## Static timing\nSetup and hold across all paths, then signoff.\n\n"
    "## RTL entry\nVerilog modules and their testbenches.\n"
)


@pytest.fixture
def llm(monkeypatch):
    """Install a fake provider. Returns a setter for what the model replies."""
    from src.domain import llm_client

    state: dict = {"reply": '{"assignments": []}', "raises": None, "calls": 0}

    async def fake_complete(messages, **kwargs):
        state["calls"] += 1
        state["messages"] = messages
        state["kwargs"] = kwargs
        if state["raises"] is not None:
            raise state["raises"]
        return state["reply"], 10, "fake-model", "fake"

    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: object())
    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(settings, "DEPARTMENT_ROUTING_LLM_ENABLED", True)
    return state


def _route(departments=UNDESCRIBED, markdown=MARKDOWN, title="Lecture-5-STA"):
    return asyncio.run(route_document_candidates_llm(title, markdown, departments))


def test_the_llm_routes_what_keyword_overlap_cannot(llm):
    """The reported situation: no description anywhere, so no keyword can ever match."""
    llm["reply"] = '{"assignments": [{"section": 1, "department": 1}, {"section": 2, "department": 2}]}'

    routed = _route()

    assert [item["department_ids"] for item in routed] == [["phys"], ["dv"]]
    # Nothing left to propose creating once a real owner has been named.
    assert all(item["proposed_department"] is None for item in routed)
    # And the deterministic path still finds nothing, which is the point.
    assert not any(
        item["department_ids"]
        for item in route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)
    )


def test_the_chosen_department_is_offered_first_to_the_reviewer(llm):
    llm["reply"] = '{"assignments": [{"section": 1, "department": 2}]}'
    first = _route()[0]
    assert first["department_suggestions"][0]["department_id"] == "dv"
    assert first["department_suggestions"][0]["name"] == "Verification"


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "",
        "{}",
        '{"assignments": null}',
        '{"assignments": [{"section": 1}]}',
        '{"assignments": "everything to physical design"}',
        '{"assignments": [{"section": "1", "department": "1"}]}',
    ],
)
def test_an_unusable_reply_falls_back_instead_of_raising(llm, reply):
    llm["reply"] = reply
    routed = _route()
    assert routed == route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)


def test_a_department_number_that_does_not_exist_is_dropped(llm):
    """Not clamped into range. A wrong department is worse than no suggestion: a
    reviewer reads a filled-in field as something the system worked out."""
    llm["reply"] = '{"assignments": [{"section": 1, "department": 99}, {"section": 2, "department": -1}]}'
    assert not any(item["department_ids"] for item in _route())


def test_department_zero_means_no_owner(llm):
    """A real answer, not a failure -- it leaves the section to the fallback."""
    llm["reply"] = '{"assignments": [{"section": 1, "department": 0}, {"section": 2, "department": 0}]}'
    assert not any(item["department_ids"] for item in _route())


def test_true_is_not_accepted_as_department_one(llm):
    """bool subclasses int, so an unguarded isinstance check would route to the first
    department on the list whenever the model answered with a boolean."""
    llm["reply"] = '{"assignments": [{"section": true, "department": true}]}'
    assert not any(item["department_ids"] for item in _route())


def test_fenced_json_is_still_read(llm):
    """Models wrap JSON in code fences whatever the prompt says."""
    llm["reply"] = '```json\n{"assignments": [{"section": 1, "department": 1}]}\n```'
    assert _route()[0]["department_ids"] == ["phys"]


def test_a_provider_that_fails_does_not_fail_the_import(llm):
    llm["raises"] = RuntimeError("provider is down")
    routed = _route()
    assert routed == route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)


def test_a_provider_that_times_out_does_not_fail_the_import(llm):
    llm["raises"] = asyncio.TimeoutError()
    assert _route() == route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)


def test_no_provider_configured_is_the_old_behaviour_exactly(monkeypatch):
    from src.domain import llm_client

    monkeypatch.setattr(llm_client, "resolve_provider", lambda *a, **k: None)
    assert _route() == route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)


def test_the_switch_turns_it_off_without_asking_the_provider(llm, monkeypatch):
    monkeypatch.setattr(settings, "DEPARTMENT_ROUTING_LLM_ENABLED", False)
    llm["reply"] = '{"assignments": [{"section": 1, "department": 1}]}'
    assert _route() == route_document_candidates("Lecture-5-STA", MARKDOWN, UNDESCRIBED)
    assert llm["calls"] == 0, "the provider must not be called when the switch is off"


def test_no_departments_means_no_call(llm):
    _route(departments=[])
    assert llm["calls"] == 0


def test_a_document_with_too_many_sections_is_not_sent(llm):
    """Past this it is re-outlining rather than routing, and the reply grows unbounded."""
    markdown = "".join(f"## Part {i}\nSome text about part {i}.\n\n" for i in range(20))
    _route(markdown=markdown)
    assert llm["calls"] == 0


def test_one_call_per_document_not_per_section(llm):
    llm["reply"] = '{"assignments": [{"section": 1, "department": 1}]}'
    _route()
    assert llm["calls"] == 1


def test_hidden_reasoning_is_disabled_for_this_call(llm):
    """The budget is tens of tokens of JSON; reasoning would consume all of it before
    the answer appeared. The same defect was fixed once already for RAG answers."""
    _route()
    assert llm["kwargs"]["thinking"] is False


def test_adjacent_sections_the_llm_gives_one_owner_are_merged(llm):
    """The merge rule is shared with the keyword path rather than reimplemented."""
    llm["reply"] = '{"assignments": [{"section": 1, "department": 1}, {"section": 2, "department": 1}]}'
    routed = _route()
    assert len(routed) == 1
    # Asserted explicitly: without it the collapse-to-one-document branch also returns a
    # single item, so this would pass against an implementation that ignored the LLM.
    assert routed[0]["department_ids"] == ["phys"]
    assert routed[0]["position"] == 1
    assert "Static timing" in routed[0]["body_md"] and "RTL entry" in routed[0]["body_md"]


def test_the_prompt_carries_the_descriptions_the_model_needs(llm):
    departments = [_Department("phys", "Physical Design", "floorplan routing signoff")]
    _route(departments=departments)
    prompt = llm["messages"][1]["content"]
    assert "1. Physical Design" in prompt
    assert "floorplan routing signoff" in prompt
    # Numbers, not UUIDs: a short integer cannot be half-hallucinated into a real id.
    assert "phys" not in prompt
