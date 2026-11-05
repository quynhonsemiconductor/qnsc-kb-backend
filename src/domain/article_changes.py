"""What changed between two versions of an article, in a form a person (or an assistant) can read at a glance.

The version history stores whole snapshots, so "what changed?" meant fetching two of them and comparing thousands of
characters by eye. This compares them line by line and reports, per section of the document, the lines that were
removed and added, plus the metadata that moved (title, department, tags, sensitivity...), with a short summary on
top. Pure functions over the snapshot dicts: no database, no model call, the same answer every time.

Whitespace at the end of a line is not a change. Long lines and long lists of changes are clipped, and the result
says so (`truncated`), because the reader is usually a model with a limited context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

#: Snapshot fields compared besides the body. Ids and timestamps are not "what the document says".
METADATA_FIELDS = (
    "title",
    "type",
    "dept",
    "language",
    "sensitivity",
    "visibility",
    "external_id",
    "tags",
)
MAX_HUNKS = 30
MAX_LINES_PER_SIDE = 12
MAX_LINE_CHARS = 300
#: Two documents longer than this are summarised by counts only: the comparison is quadratic in the worst case.
MAX_COMPARED_LINES = 20_000
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")


def _lines(text: Any) -> list[str]:
    return [line.rstrip() for line in str(text or "").splitlines()]


def _heading_at(lines: list[str], index: int) -> str | None:
    """The nearest markdown heading at or above `index`, or None before the first one."""
    for position in range(min(index, len(lines) - 1), -1, -1):
        match = _HEADING.match(lines[position])
        if match and match.group(1):
            return match.group(1)
    return None


def _first_text(lines: list[str], start: int, end: int) -> int:
    """Index of the first non-blank line in lines[start:end], else `start`."""
    return next((k for k in range(start, end) if lines[k].strip()), start)


def _clip(line: str) -> str:
    return (
        line
        if len(line) <= MAX_LINE_CHARS
        else line[:MAX_LINE_CHARS].rstrip() + " [...]"
    )


def _side(lines: list[str]) -> tuple[list[str], int]:
    """The non-blank lines of one side of a hunk, clipped, and how many were left out."""
    kept = [line for line in lines if line.strip()]
    shown = [_clip(line) for line in kept[:MAX_LINES_PER_SIDE]]
    return shown, max(0, len(kept) - len(shown))


def _metadata_changes(before: dict, after: dict) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for field in METADATA_FIELDS:
        # A field one snapshot never recorded is unknown, not changed: older versions were written by code
        # that stored fewer fields, and "visibility: None -> department" would be noise.
        if field not in before or field not in after:
            continue
        old, new = before.get(field), after.get(field)
        if field == "tags":
            old, new = sorted(old or []), sorted(new or [])
            if old != new:
                changes.append(
                    {
                        "field": "tags",
                        "added": [tag for tag in new if tag not in old],
                        "removed": [tag for tag in old if tag not in new],
                    }
                )
        elif (old or None) != (new or None):
            changes.append({"field": field, "before": old, "after": new})
    return changes


def diff_versions(
    before: dict[str, Any],
    after: dict[str, Any],
    max_lines: int | None = None,
) -> dict[str, Any]:
    """Compare two version snapshots. `before` is the older one.

    `max_lines` lowers the length above which the text is summarised by counts only (default MAX_COMPARED_LINES): a
    caller comparing many articles in one request bounds its worst case that way.
    """
    old_lines, new_lines = _lines(before.get("body_md")), _lines(after.get("body_md"))
    metadata = _metadata_changes(before, after)
    result: dict[str, Any] = {"metadata_changes": metadata}

    ceiling = MAX_COMPARED_LINES if max_lines is None else max_lines
    if max(len(old_lines), len(new_lines)) > ceiling:
        result.update(
            body={"lines_before": len(old_lines), "lines_after": len(new_lines)},
            hunks=[],
            truncated=True,
            summary=[
                f"The text is too long to compare line by line ({len(old_lines)} -> {len(new_lines)} lines); "
                "open both versions to compare them."
            ],
        )
        return result

    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[dict[str, Any]] = []
    added_total = removed_total = 0
    sections: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed, removed_more = _side(old_lines[i1:i2])
        added, added_more = _side(new_lines[j1:j2])
        removed_count = sum(1 for line in old_lines[i1:i2] if line.strip())
        added_count = sum(1 for line in new_lines[j1:j2] if line.strip())
        if not removed_count and not added_count:
            continue  # only blank lines moved
        removed_total += removed_count
        added_total += added_count
        # A hunk belongs to the section it lands in: where the new text is, or for a pure deletion where the old was.
        # The position is that of the first line with text: a hunk that opens with a blank line (the gap before a
        # new heading) would otherwise be filed under the section above it.
        heading = (
            _heading_at(new_lines, _first_text(new_lines, j1, j2))
            if added_count
            else _heading_at(old_lines, _first_text(old_lines, i1, i2))
        ) or "(before the first heading)"
        if heading not in sections:
            sections.append(heading)
        hunk: dict[str, Any] = {
            "section": heading,
            "kind": (
                "added"
                if not removed_count
                else "removed" if not added_count else "changed"
            ),
            "removed": removed,
            "added": added,
        }
        if removed_more or added_more:
            hunk["more_lines"] = removed_more + added_more
        hunks.append(hunk)

    shown = hunks[:MAX_HUNKS]
    result["hunks"] = shown
    result["body"] = {
        "lines_added": added_total,
        "lines_removed": removed_total,
        "sections": sections,
    }
    result["truncated"] = len(hunks) > len(shown) or any(
        "more_lines" in h for h in shown
    )
    if len(hunks) > len(shown):
        result["hunks_not_shown"] = len(hunks) - len(shown)

    summary = []
    if added_total or removed_total:
        where = ", ".join(f'"{s}"' for s in sections[:5]) + (
            f" and {len(sections) - 5} more" if len(sections) > 5 else ""
        )
        summary.append(
            f"Text: {added_total} line(s) added, {removed_total} removed, in {len(sections)} "
            f"section(s): {where}."
        )
    else:
        summary.append("The text is unchanged.")
    for change in metadata:
        if change["field"] == "tags":
            parts = []
            if change["added"]:
                parts.append("added " + ", ".join(change["added"]))
            if change["removed"]:
                parts.append("removed " + ", ".join(change["removed"]))
            summary.append("Tags: " + "; ".join(parts) + ".")
        else:
            summary.append(
                f"{change['field'].replace('_', ' ').capitalize()}: "
                f"{change['before']!r} -> {change['after']!r}."
            )
    result["summary"] = summary
    return result


@dataclass(frozen=True)
class Selection:
    """Which two versions a question about changes is about, decided from the version numbers alone.

    `from_version` is None for the first version of an article (nothing earlier to compare with). Deciding this needs
    no snapshot, so a caller can load only the two it names.
    """

    numbers: list[int]
    latest: int
    new_number: int
    from_version: int | None


def select_versions(
    versions: list[Any], from_version: int | None, to_version: int | None
) -> Selection:
    """Pick the two versions. `versions` is the article's history, any order, each with `.version`.

    By default the latest against the one before it; `from_version` alone compares that version with the latest, which
    is how a reader who read version N asks what changed since. Raises ValueError with a message for the caller.
    """
    numbers = sorted({item.version for item in versions})
    if not numbers:
        raise ValueError("This article has no stored versions.")
    latest = numbers[-1]
    new_number = to_version if to_version is not None else latest
    if new_number not in numbers:
        raise ValueError(
            f"Version {new_number} does not exist; versions are {numbers[0]}..{latest}."
        )
    if from_version is None:
        earlier = [n for n in numbers if n < new_number]
        return Selection(numbers, latest, new_number, earlier[-1] if earlier else None)
    if from_version not in numbers:
        raise ValueError(
            f"Version {from_version} does not exist; versions are {numbers[0]}..{latest}."
        )
    if from_version >= new_number:
        raise ValueError("'from_version' must be older than 'to_version'.")
    return Selection(numbers, latest, new_number, from_version)


def first_version_result(selection: Selection, version: Any) -> dict[str, Any]:
    """The answer for an article's first version: there is nothing earlier to compare it with."""
    return {
        "first_version": True,
        "version": selection.new_number,
        "created_at": _stamp(version),
        "summary": [
            f"Version {selection.new_number} is the first version; there is nothing earlier to compare."
        ],
    }


def compare_selected(selection: Selection, old: Any, new: Any) -> dict[str, Any]:
    """Compare the two versions a selection named (each with `.snapshot` and `.created_at`)."""
    assert selection.from_version is not None
    result = diff_versions(old.snapshot or {}, new.snapshot or {})
    between = [
        n
        for n in selection.numbers
        if selection.from_version < n <= selection.new_number
    ]
    return {
        "from_version": selection.from_version,
        "to_version": selection.new_number,
        "from_at": _stamp(old),
        "to_at": _stamp(new),
        "versions_covered": between,
        "latest_version": selection.latest,
        **result,
    }


def describe_changes(
    versions: list[Any], from_version: int | None, to_version: int | None
) -> dict[str, Any]:
    """Pick the two versions and compare them; `versions` carry `.version`, `.snapshot` and `.created_at`.

    The endpoint does the same in two steps (`select_versions`, then the two snapshots it names) so that an article
    with a long history does not load every snapshot to compare two.
    """
    selection = select_versions(versions, from_version, to_version)
    by_number = {item.version: item for item in versions}
    if selection.from_version is None:
        return first_version_result(selection, by_number[selection.new_number])
    return compare_selected(
        selection, by_number[selection.from_version], by_number[selection.new_number]
    )


def _stamp(version: Any) -> str | None:
    created = getattr(version, "created_at", None)
    return created.isoformat() if created else None


#: Lines above which the feed does not compare an article line by line: it lists many articles per request.
FEED_MAX_COMPARED_LINES = 4_000
FEED_SUMMARY_LINES = 3


def describe_recent(changes: list[Any]) -> list[dict[str, Any]]:
    """The newest changes across articles as short entries, each with a line saying what changed.

    `changes` are `RecentChange`s (src/repositories/article.py): `.article`, `.latest`, `.baseline`, `.versions`,
    `.changed_at`. An article with no version from before the period is "new"; otherwise "updated", with the newest
    version compared to the one the period started from. Pure: no database, no model call.
    """
    entries: list[dict[str, Any]] = []
    for change in changes:
        article = change.article
        entry: dict[str, Any] = {
            "article_id": str(article.id),
            "title": article.title,
            "dept": article.dept,
            "type": article.type,
            "kind": "new" if change.baseline is None else "updated",
            "version": change.latest.version,
            "versions_in_period": change.versions,
            "changed_at": change.changed_at.isoformat(),
        }
        if change.baseline is not None:
            result = diff_versions(
                change.baseline.snapshot or {},
                change.latest.snapshot or {},
                max_lines=FEED_MAX_COMPARED_LINES,
            )
            entry["from_version"] = change.baseline.version
            entry["summary"] = result["summary"][:FEED_SUMMARY_LINES]
            entry["sections"] = (result.get("body") or {}).get("sections", [])[:5]
        entries.append(entry)
    return entries
