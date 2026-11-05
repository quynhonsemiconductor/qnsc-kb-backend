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
    from src.repositories.article import ArticleRepository

    holder = SimpleNamespace(
        history=list(reversed(HISTORY)), asked=[], loaded=[], index_loaded=0
    )

    async def get_version_index(self, user, article_id):
        holder.asked.append((user, article_id))
        holder.index_loaded += 1
        index = [
            SimpleNamespace(version=item.version, created_at=item.created_at)
            for item in holder.history
        ]
        return SimpleNamespace(title="UART"), index

    async def get_version_by_number(self, article_id, version_num, user=None):
        holder.loaded.append(version_num)
        return next(
            (item for item in holder.history if item.version == version_num), None
        )

    monkeypatch.setattr(ArticleService, "get_version_index", get_version_index)
    monkeypatch.setattr(
        ArticleRepository, "get_version_by_number", get_version_by_number
    )
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
    real = articles.compare_selected

    def spy(*args):
        seen.append(threading.current_thread())
        return real(*args)

    monkeypatch.setattr(articles, "compare_selected", spy)

    _call()

    assert seen and seen[0] is not threading.main_thread()


# -- the feed of recent changes ------------------------------------------------------------------


def _change(
    title, latest_body, baseline_body=None, *, versions=1, number=2, kind_day=3
):
    return SimpleNamespace(
        article=SimpleNamespace(
            id=uuid.uuid4(), title=title, dept="Engineering", type="SOP"
        ),
        latest=_version(number, latest_body, day=kind_day),
        baseline=(
            None if baseline_body is None else _version(number - 1, baseline_body)
        ),
        versions=versions,
        changed_at=datetime(2026, 11, kind_day, 9, 0, 0),
    )


def test_an_article_with_nothing_before_the_period_is_new():
    [entry] = article_changes.describe_recent([_change("A", "# A\n\ntext\n")])

    assert entry["kind"] == "new" and entry["title"] == "A"
    assert "summary" not in entry and "from_version" not in entry
    assert entry["changed_at"] == "2026-11-03T09:00:00"


def test_an_existing_article_is_updated_and_says_what_changed():
    entry = article_changes.describe_recent(
        [_change("A", "# A\n\none\ntwo\n", "# A\n\none\n", versions=2, number=3)]
    )[0]

    assert entry["kind"] == "updated" and entry["from_version"] == 2
    assert entry["version"] == 3 and entry["versions_in_period"] == 2
    assert entry["summary"][0].startswith("Text: 1 line(s) added, 0 removed")
    assert entry["sections"] == ["A"]


def test_the_feed_keeps_the_order_it_was_given_and_caps_the_summary():
    many = _change(
        "B",
        "# B\n",
        "# B\n",
        number=2,
    )
    many.latest.snapshot.update(
        title="B2", dept="Firmware", tags=["x"], sensitivity="internal", type="HOWTO"
    )
    entries = article_changes.describe_recent([_change("A", "x"), many])

    assert [e["title"] for e in entries] == ["A", "B"]
    assert len(entries[1]["summary"]) <= article_changes.FEED_SUMMARY_LINES


def test_a_long_article_is_not_compared_line_by_line_in_the_feed(monkeypatch):
    monkeypatch.setattr(article_changes, "FEED_MAX_COMPARED_LINES", 10)
    body = "\n".join(f"line {i}" for i in range(50))

    entry = article_changes.describe_recent(
        [_change("Big", body + "\nmore", body, number=2)]
    )[0]

    assert "too long to compare" in entry["summary"][0]


def test_diff_versions_takes_a_lower_ceiling_for_the_caller_that_needs_one():
    old, new = _snapshot("a\n" * 30), _snapshot("b\n" * 30)

    assert diff_versions(old, new, max_lines=10)["hunks"] == []
    assert diff_versions(old, new)["hunks"], "the default ceiling compares these"


@pytest.fixture
def recent_repo(monkeypatch):
    holder = SimpleNamespace(asked=[], found=[], more=False)

    class FakeRepo:
        def __init__(self, db):
            pass

        async def recent_changes(self, user, since, dept=None, scope=None, limit=20):
            holder.asked.append(
                SimpleNamespace(
                    user=user, since=since, dept=dept, scope=scope, limit=limit
                )
            )
            return holder.found, holder.more

    monkeypatch.setattr(articles, "ArticleRepository", FakeRepo)
    return holder


def _recent(**kwargs):
    import asyncio

    args = dict(
        days=7,
        since=None,
        dept=None,
        scope="all",
        limit=20,
        current_user=SimpleNamespace(id=1),
        db=object(),
    )
    args.update(kwargs)
    return asyncio.run(articles.recent_changes(**args))


def test_the_feed_endpoint_looks_back_the_days_asked_in_naive_utc(recent_repo):
    from datetime import timedelta, timezone

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    out = _recent(days=3)

    [asked] = recent_repo.asked
    assert asked.since.tzinfo is None, "version timestamps are stored naive"
    assert (
        timedelta(days=3, seconds=-5)
        < before - asked.since
        < timedelta(days=3, seconds=5)
    )
    assert out["count"] == 0 and out["changes"] == [] and out["truncated"] is False
    assert (
        out["since"] == asked.since.isoformat()
    ), "echoed as naive UTC, like changed_at"


def test_the_feed_endpoint_prefers_since_and_reads_an_offset_time(recent_repo):
    from datetime import timezone

    _recent(since=datetime(2026, 11, 1, 7, 0, tzinfo=timezone(timedelta_hours(7))))

    assert recent_repo.asked[0].since == datetime(2026, 11, 1, 0, 0)


def timedelta_hours(hours):
    from datetime import timedelta

    return timedelta(hours=hours)


def test_the_feed_endpoint_treats_a_bare_date_as_utc(recent_repo):
    _recent(since=datetime(2026, 11, 1))

    assert recent_repo.asked[0].since == datetime(2026, 11, 1)


def test_the_feed_endpoint_refuses_a_start_in_the_future(recent_repo):
    from datetime import timedelta, timezone

    with pytest.raises(HTTPException) as caught:
        _recent(since=datetime.now(timezone.utc) + timedelta(days=1))

    assert caught.value.status_code == 422 and not recent_repo.asked


def test_the_feed_endpoint_passes_the_filters_and_the_callers_own_rights(recent_repo):
    user = SimpleNamespace(id="me")

    _recent(dept="Firmware", scope="lessons", limit=5, current_user=user)

    asked = recent_repo.asked[0]
    assert asked.user is user and asked.dept == "Firmware"
    assert asked.scope == "lessons" and asked.limit == 5
    _recent(scope="all")
    assert recent_repo.asked[1].scope is None


def test_the_feed_endpoint_reports_what_the_repository_says_about_more(recent_repo):
    recent_repo.found = [_change("A", "x"), _change("B", "y")]

    full = _recent(limit=2)
    recent_repo.more = True
    cut = _recent(limit=2)

    assert [c["title"] for c in full["changes"]] == ["A", "B"]
    assert full["truncated"] is False, "exactly a full page is not a cut one"
    assert cut["truncated"] is True and cut["count"] == 2


def test_a_zero_ceiling_is_a_ceiling_not_unset():
    old, new = _snapshot("a"), _snapshot("b")

    assert diff_versions(old, new, max_lines=0)["hunks"] == []
    assert diff_versions(old, new, max_lines=None)["hunks"]


def test_the_repository_reads_one_row_more_than_the_page_to_know_about_more():
    import asyncio

    from src.repositories.article import ArticleRepository

    from tests.unit.test_query_authorization import _reader

    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

        def scalars(self):
            return Rows([])

    class Db:
        def __init__(self, rows):
            self.rows = rows
            self.statements = []

        async def execute(self, statement):
            self.statements.append(statement)
            return Rows(self.rows if len(self.statements) == 1 else [])

    three = [
        SimpleNamespace(
            article_id=i, latest=1, versions=1, changed_at=datetime(2026, 11, 2)
        )
        for i in range(3)
    ]
    db = Db(three)

    changes, more = asyncio.run(
        ArticleRepository(db).recent_changes(_reader(), datetime(2026, 11, 1), limit=2)
    )

    assert more is True
    assert "LIMIT" in str(db.statements[0].compile()).upper()
    assert db.statements[0]._limit_clause.value == 3, "limit + 1"


def test_only_the_snapshots_the_question_names_are_loaded(history_for_endpoint):
    _call()
    assert sorted(history_for_endpoint.loaded) == [
        2,
        3,
    ], "the latest and the one before"

    history_for_endpoint.loaded.clear()
    _call(from_version=1)
    assert sorted(history_for_endpoint.loaded) == [1, 3]

    history_for_endpoint.loaded.clear()
    _call(from_version=1, to_version=2)
    assert sorted(history_for_endpoint.loaded) == [1, 2]


def test_the_first_version_loads_one_snapshot_and_compares_nothing(
    history_for_endpoint,
):
    history_for_endpoint.history = [HISTORY[0]]

    out = _call()

    assert out["first_version"] is True and history_for_endpoint.loaded == [1]


def test_a_bad_version_is_refused_before_any_snapshot_is_loaded(history_for_endpoint):
    with pytest.raises(HTTPException):
        _call(from_version=9)

    assert history_for_endpoint.loaded == []


def test_the_version_index_selects_numbers_and_times_not_snapshots():
    import asyncio

    from src.repositories.article import ArticleRepository
    from tests.unit.test_query_authorization import _DB, _reader

    db = _DB()
    asyncio.run(ArticleRepository(db).version_index(uuid.uuid4(), _reader()))

    sql = str(db.statement.compile()).lower()
    select_list = sql.split(" from ")[0]
    assert "article_versions.version" in select_list
    assert (
        "snapshot" not in select_list
    ), "the text of every version is what this avoids"
    assert "articles.company_domain" in sql, "the same visibility rules as the history"


def test_select_versions_decides_from_numbers_alone():
    from src.domain.article_changes import select_versions

    numbers_only = [SimpleNamespace(version=n) for n in (3, 1, 2)]

    chosen = select_versions(numbers_only, None, None)
    assert (chosen.from_version, chosen.new_number, chosen.latest) == (2, 3, 3)
    assert select_versions(numbers_only[1:2], None, None).from_version is None
    assert select_versions(numbers_only, 1, 2).numbers == [1, 2, 3]
