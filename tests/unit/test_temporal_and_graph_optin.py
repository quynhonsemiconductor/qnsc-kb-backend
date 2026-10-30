"""Time-focused retrieval and the per-tenant switch for the entity-graph leg."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from src.domain import temporal
from src.domain.agent.core import Budget, Tracer
from src.domain.agent.plan import AgentPlan, sanitize_plan
from src.domain.agent.retrieve import agentic_retrieve
from src.domain.temporal import (
    IN_FORCE,
    OUT_OF_FORCE,
    UNDATED,
    Validity,
    apply_time_focus,
    bucket,
    parse_as_of,
    parse_validity,
    rank_by_time,
)
from src.repositories.feature_flags import FeatureFlagRepository


def _run(coro):
    return asyncio.run(coro)


MARCH_2025 = date(2025, 3, 1)


# --- validity -------------------------------------------------------------------------


def test_dates_are_read_from_the_structured_metadata():
    v = parse_validity({"issue_date": "2024-01-15", "expiry_date": "2025-12-31"})

    assert v == Validity(date(2024, 1, 15), date(2025, 12, 31))


@pytest.mark.parametrize(
    "metadata",
    [None, {}, [], "x", {"issue_date": "not a date"}, {"issue_date": None}],
)
def test_missing_or_unreadable_dates_mean_undated(metadata):
    assert parse_validity(metadata) == Validity()


def test_the_planners_date_must_be_a_plausible_iso_date():
    assert parse_as_of("2025-03-01") == date(2025, 3, 1)
    for bad in (None, "", "March 2025", "1850-01-01", "2999-01-01", 5):
        assert parse_as_of(bad) is None


# --- bucketing ------------------------------------------------------------------------


def test_an_article_issued_before_and_not_yet_expired_is_in_force():
    assert bucket(Validity(date(2024, 1, 1), date(2026, 1, 1)), MARCH_2025) == IN_FORCE
    assert bucket(Validity(date(2024, 1, 1), None), MARCH_2025) == IN_FORCE


def test_an_article_issued_later_or_expired_earlier_is_not_in_force():
    assert bucket(Validity(date(2025, 6, 1), None), MARCH_2025) == OUT_OF_FORCE
    assert bucket(Validity(date(2020, 1, 1), date(2024, 12, 31)), MARCH_2025) == (
        OUT_OF_FORCE
    )


def test_no_dates_and_an_expiry_alone_are_not_promoted():
    assert bucket(Validity(), MARCH_2025) == UNDATED
    assert bucket(Validity(None, date(2026, 1, 1)), MARCH_2025) == UNDATED


def test_the_boundaries_are_inclusive():
    assert bucket(Validity(MARCH_2025, MARCH_2025), MARCH_2025) == IN_FORCE


# --- ranking --------------------------------------------------------------------------


def _hit(article_id, score=0.5):
    return {"article_id": article_id, "chunk_id": f"c-{article_id}", "score": score}


_VALIDITY = {
    "old": Validity(date(2020, 1, 1), date(2024, 12, 31)),
    "current": Validity(date(2025, 1, 1), None),
    "future": Validity(date(2026, 1, 1), None),
    "plain": Validity(),
}


def test_articles_in_force_come_first_then_undated_then_the_rest():
    results = [_hit("future"), _hit("old"), _hit("plain"), _hit("current")]

    ranked = rank_by_time(results, MARCH_2025, _VALIDITY)

    assert [r["article_id"] for r in ranked] == ["current", "plain", "future", "old"]


def test_ranking_keeps_retrieval_order_within_a_bucket_and_drops_nothing():
    results = [_hit("current", 0.9), _hit("plain"), _hit("current", 0.4), _hit("old")]

    ranked = rank_by_time(results, MARCH_2025, _VALIDITY)

    assert len(ranked) == len(results)
    assert [r["score"] for r in ranked if r["article_id"] == "current"] == [0.9, 0.4]


def test_an_article_with_no_stored_dates_is_undated():
    ranked = rank_by_time([_hit("unknown"), _hit("current")], MARCH_2025, _VALIDITY)

    assert [r["article_id"] for r in ranked] == ["current", "unknown"]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Db:
    def __init__(self, rows=None, error=None):
        self.rows, self.error = rows or [], error

    async def execute(self, _stmt):
        if self.error:
            raise self.error
        return _Rows(self.rows)


def _db():
    return _Db(
        [
            ("old", {"issue_date": "2020-01-01", "expiry_date": "2024-12-31"}),
            ("current", {"issue_date": "2025-01-01"}),
        ]
    )


def test_time_focus_reorders_with_dates_read_from_the_database():
    results = [_hit("old"), _hit("current")]

    out = _run(apply_time_focus(_db(), results, MARCH_2025))

    assert [r["article_id"] for r in out] == ["current", "old"]


@pytest.mark.parametrize("as_of", [None])
def test_without_a_date_the_order_is_untouched(as_of):
    results = [_hit("old"), _hit("current")]

    assert _run(apply_time_focus(_db(), results, as_of)) is results


def test_without_a_session_or_with_one_result_the_order_is_untouched():
    one = [_hit("old")]
    two = [_hit("old"), _hit("current")]

    assert _run(apply_time_focus(_db(), one, MARCH_2025)) is one
    assert _run(apply_time_focus(None, two, MARCH_2025)) is two


def test_a_failed_date_lookup_leaves_the_order_as_retrieval_made_it():
    results = [_hit("old"), _hit("current")]

    out = _run(
        apply_time_focus(_Db(error=RuntimeError("db down")), results, MARCH_2025)
    )

    assert out is results


def test_no_article_dates_at_all_leaves_the_order_alone():
    results = [_hit("a"), _hit("b")]

    assert _run(
        apply_time_focus(_Db([("a", None), ("b", {})]), results, MARCH_2025)
    ) is (results)


# --- the planner ------------------------------------------------------------------------


def test_the_plan_keeps_a_valid_date_and_drops_nonsense():
    good = sanitize_plan({"sub_queries": ["q"], "as_of": "2025-03-01"}, "q", [])
    bad = sanitize_plan({"sub_queries": ["q"], "as_of": "last spring"}, "q", [])
    none = sanitize_plan({"sub_queries": ["q"]}, "q", [])

    assert good.as_of == "2025-03-01"
    assert bad.as_of is None and none.as_of is None
    assert good.public()["as_of"] == "2025-03-01"


def test_the_planner_prompt_tells_the_model_todays_date():
    from src.domain.agent import plan as plan_module

    rendered = plan_module._SYSTEM % {
        "max_sub": 3,
        "types": "SOP",
        "today": date.today().isoformat(),
    }

    assert date.today().isoformat() in rendered and '"as_of"' in rendered


# --- retrieval --------------------------------------------------------------------------


class _Search:
    def __init__(self, results):
        self._results = results
        self.chunk_repo = SimpleNamespace(db=_db())
        self.vector_search_degraded = False

    async def search(self, user, query, filters=None, limit=5):
        return list(self._results)


def _retrieve(plan, results):
    return _run(
        agentic_retrieve(
            _Search(results),
            SimpleNamespace(company_domain="qnsc.vn"),
            "q",
            plan,
            tracer=Tracer(),
            budget=Budget(6),
        )
    )


def test_a_question_about_a_date_puts_the_articles_in_force_then_first():
    results = [_hit("old", 0.95), _hit("current", 0.9)]
    plan = AgentPlan(sub_queries=["q"], as_of="2025-03-01")

    out = _retrieve(plan, results)

    assert [r["article_id"] for r in out.results] == ["current", "old"]


def test_a_question_with_no_date_keeps_the_ranking():
    results = [_hit("old", 0.95), _hit("current", 0.9)]

    out = _retrieve(AgentPlan(sub_queries=["q"]), results)

    assert [r["article_id"] for r in out.results] == ["old", "current"]


def test_the_setting_turns_time_focus_off(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "TEMPORAL_RETRIEVAL_ENABLED", False)
    results = [_hit("old", 0.95), _hit("current", 0.9)]

    out = _retrieve(AgentPlan(sub_queries=["q"], as_of="2025-03-01"), results)

    assert [r["article_id"] for r in out.results] == ["old", "current"]


# --- graph retrieval opt-in -------------------------------------------------------------


class _FlagRepo(FeatureFlagRepository):
    def __init__(self, flag):
        self._flag = flag

    async def get(self, key):
        return self._flag


_USER = SimpleNamespace(id="u1", role="Employee", dept="HR")


def _flag(**overrides):
    base = dict(enabled=True, rollout_percent=100, role=None, department=None)
    return SimpleNamespace(**{**base, **overrides})


def test_no_flag_means_the_graph_leg_stays_off():
    assert _run(_FlagRepo(None).is_opted_in("rag.graph_retrieval", _USER)) is False
    # while is_enabled keeps its "no flag, feature on" meaning for features that ship on
    assert _run(_FlagRepo(None).is_enabled("rag.graph_retrieval", _USER)) is True


def test_a_switched_on_flag_opts_in_and_a_switched_off_one_does_not():
    assert _run(_FlagRepo(_flag()).is_opted_in("k", _USER)) is True
    assert _run(_FlagRepo(_flag(enabled=False)).is_opted_in("k", _USER)) is False


def test_an_opted_in_flag_still_follows_role_department_and_rollout():
    assert _run(_FlagRepo(_flag(role="Admin")).is_opted_in("k", _USER)) is False
    assert _run(_FlagRepo(_flag(department="Finance")).is_opted_in("k", _USER)) is False
    assert _run(_FlagRepo(_flag(rollout_percent=0)).is_opted_in("k", _USER)) is False
