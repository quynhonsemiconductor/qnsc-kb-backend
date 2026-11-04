"""`get_article_changes`: what changed between two versions of an article (src/domain/article_changes.py)."""

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routers import articles
from src.domain import article_changes
from src.domain.article_changes import describe_changes, diff_versions

BEFORE = """# UART

## Overview

UART0 is the console.

## Registers

| name | offset |
|---|---|
| DLL | 0x00 |
| DLM | 0x04 |

## Reset

Reset value is 0.
"""


def _snapshot(body, **extra):
    return {
        "title": "UART",
        "dept": "Engineering",
        "tags": ["uart"],
        "body_md": body,
        **extra,
    }


def _version(number, body, day=1, **extra):
    return SimpleNamespace(
        version=number,
        snapshot=_snapshot(body, **extra),
        created_at=datetime(2026, 11, day, 9, 0, 0),
    )


def test_a_changed_line_is_reported_in_its_own_section():
    after = BEFORE.replace("| DLM | 0x04 |", "| DLM | 0x08 |")

    result = diff_versions(_snapshot(BEFORE), _snapshot(after))

    assert result["body"] == {
        "lines_added": 1,
        "lines_removed": 1,
        "sections": ["Registers"],
    }
    [hunk] = result["hunks"]
    assert hunk["section"] == "Registers" and hunk["kind"] == "changed"
    assert hunk["removed"] == ["| DLM | 0x04 |"] and hunk["added"] == ["| DLM | 0x08 |"]
    assert result["summary"][0].startswith("Text: 1 line(s) added, 1 removed")
    assert result["truncated"] is False


def test_an_added_section_and_a_removed_line_are_told_apart():
    after = (
        BEFORE.replace("UART0 is the console.\n", "")
        + "\n## Errors\n\nParity error sets bit 2.\n"
    )

    result = diff_versions(_snapshot(BEFORE), _snapshot(after))

    kinds = {h["section"]: h["kind"] for h in result["hunks"]}
    assert (
        kinds["Overview"] == "removed"
    ), "a deletion is attributed to the section it was in"
    assert kinds["Errors"] == "added"
    assert result["body"]["sections"] == ["Overview", "Errors"]


def test_trailing_whitespace_and_blank_lines_are_not_changes():
    after = BEFORE.replace("UART0 is the console.", "UART0 is the console.   ").replace(
        "\n\n## Reset", "\n\n\n## Reset"
    )

    result = diff_versions(_snapshot(BEFORE), _snapshot(after))

    assert result["hunks"] == [] and result["summary"] == ["The text is unchanged."]


def test_title_department_and_tags_that_moved_are_listed():
    result = diff_versions(
        _snapshot(BEFORE, tags=["uart", "old"], sensitivity="public"),
        _snapshot(
            BEFORE,
            tags=["uart", "new"],
            sensitivity="internal",
            title="UART v2",
            dept="Firmware",
        ),
    )

    by_field = {c["field"]: c for c in result["metadata_changes"]}
    assert by_field["tags"] == {"field": "tags", "added": ["new"], "removed": ["old"]}
    assert by_field["title"] == {"field": "title", "before": "UART", "after": "UART v2"}
    assert (
        by_field["dept"]["after"] == "Firmware"
        and by_field["sensitivity"]["after"] == "internal"
    )
    assert "Tags: added new; removed old." in result["summary"]
    assert "The text is unchanged." in result["summary"]


def test_a_field_an_older_snapshot_never_recorded_is_not_reported_as_changed():
    old = {"title": "UART", "body_md": "x", "tags": ["uart"]}
    new = {
        "title": "UART",
        "body_md": "x",
        "tags": ["uart"],
        "visibility": "department",
    }

    result = diff_versions(old, new)

    assert result["metadata_changes"] == []
    assert result["summary"] == ["The text is unchanged."]


def test_a_big_change_is_clipped_and_says_so():
    old = "\n".join(f"line {i}" for i in range(100))
    new = "\n".join(f"changed {i}" for i in range(100))
    result = diff_versions(_snapshot(old), _snapshot(new))
    [hunk] = result["hunks"]
    assert len(hunk["removed"]) == article_changes.MAX_LINES_PER_SIDE
    assert hunk["more_lines"] == 2 * (100 - article_changes.MAX_LINES_PER_SIDE)
    assert result["truncated"] is True and result["body"]["lines_removed"] == 100

    many_old = "\n".join(f"a{i}\nkeep{i}" for i in range(80))
    many_new = "\n".join(f"b{i}\nkeep{i}" for i in range(80))
    many = diff_versions(_snapshot(many_old), _snapshot(many_new))
    assert len(many["hunks"]) == article_changes.MAX_HUNKS
    assert (
        many["hunks_not_shown"] == 80 - article_changes.MAX_HUNKS
        and many["truncated"] is True
    )


def test_a_very_long_line_is_clipped():
    result = diff_versions(_snapshot("x"), _snapshot("y" * 5000))
    assert len(result["hunks"][0]["added"][0]) < 400


def test_text_too_long_to_compare_is_summarised_by_counts(monkeypatch):
    monkeypatch.setattr(article_changes, "MAX_COMPARED_LINES", 10)

    result = diff_versions(_snapshot("a\n" * 30), _snapshot("b\n" * 30))

    assert result["hunks"] == [] and result["truncated"] is True
    assert result["body"] == {"lines_before": 30, "lines_after": 30}


# -- choosing the two versions -------------------------------------------------------------------


HISTORY = [
    _version(1, "# A\n\none\n", day=1),
    _version(2, "# A\n\none\ntwo\n", day=2),
    _version(3, "# A\n\none\ntwo\nthree\n", day=3),
]


def test_by_default_the_latest_is_compared_with_the_one_before():
    result = describe_changes(list(reversed(HISTORY)), None, None)

    assert (result["from_version"], result["to_version"]) == (2, 3)
    assert result["versions_covered"] == [3] and result["latest_version"] == 3
    assert result["hunks"][0]["added"] == ["three"]
    assert result["from_at"].startswith("2026-11-02") and result["to_at"].startswith(
        "2026-11-03"
    )


def test_from_version_alone_means_changed_since_that_version():
    result = describe_changes(HISTORY, 1, None)

    assert (result["from_version"], result["to_version"]) == (1, 3)
    assert result["versions_covered"] == [2, 3]
    assert result["hunks"][0]["added"] == ["two", "three"]
    assert result["body"]["lines_added"] == 2


def test_to_version_stops_short_of_the_latest():
    result = describe_changes(HISTORY, None, 2)

    assert (result["from_version"], result["to_version"]) == (1, 2)
    assert result["latest_version"] == 3


def test_the_first_version_has_nothing_to_compare_with():
    result = describe_changes([HISTORY[0]], None, None)

    assert result["first_version"] is True and result["version"] == 1
    assert "first version" in result["summary"][0]


@pytest.mark.parametrize(
    "older,newer,message",
    [
        (9, None, "Version 9 does not exist"),
        (None, 9, "Version 9 does not exist"),
        (3, 2, "must be older"),
    ],
)
def test_a_version_that_is_missing_or_backwards_is_a_message_not_a_guess(
    older, newer, message
):
    with pytest.raises(ValueError, match=message):
        describe_changes(HISTORY, older, newer)


def test_no_versions_at_all_is_a_message():
    with pytest.raises(ValueError, match="no stored versions"):
        describe_changes([], None, None)


# -- the endpoint --------------------------------------------------------------------------------


@pytest.fixture
def history_for_endpoint(monkeypatch):
    from src.domain.articles import ArticleService

    holder = SimpleNamespace(history=list(reversed(HISTORY)), asked=[])

    async def get_history(self, user, article_id):
        holder.asked.append((user, article_id))
        return holder.history

    monkeypatch.setattr(ArticleService, "get_history", get_history)
    return holder


def _call(**kwargs):
    import asyncio

    args = dict(
        from_version=None,
        to_version=None,
        current_user=SimpleNamespace(id=1),
        db=object(),
    )
    args.update(kwargs)
    return asyncio.run(articles.article_changes(uuid.uuid4(), **args))


def test_the_endpoint_returns_the_title_and_the_comparison(history_for_endpoint):
    out = _call()

    assert out["title"] == "UART" and out["to_version"] == 3 and out["hunks"]


def test_the_endpoint_asks_the_history_with_the_callers_own_rights(
    history_for_endpoint,
):
    user = SimpleNamespace(id="me")

    _call(current_user=user)

    assert history_for_endpoint.asked[0][0] is user


def test_the_endpoint_turns_a_bad_version_into_a_422(history_for_endpoint):
    with pytest.raises(HTTPException) as caught:
        _call(from_version=9)

    assert (
        caught.value.status_code == 422
        and "Version 9 does not exist" in caught.value.detail
    )


def test_the_comparison_runs_off_the_event_loop(history_for_endpoint, monkeypatch):
    import threading

    seen = []
    real = articles.describe_changes

    def spy(*args):
        seen.append(threading.current_thread())
        return real(*args)

    monkeypatch.setattr(articles, "describe_changes", spy)

    _call()

    assert seen and seen[0] is not threading.main_thread()
