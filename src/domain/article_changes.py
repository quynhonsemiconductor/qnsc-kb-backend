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


def diff_versions(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two version snapshots. `before` is the older one."""
    old_lines, new_lines = _lines(before.get("body_md")), _lines(after.get("body_md"))
    metadata = _metadata_changes(before, after)
    result: dict[str, Any] = {"metadata_changes": metadata}

    if max(len(old_lines), len(new_lines)) > MAX_COMPARED_LINES:
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


def describe_changes(
    versions: list[Any], from_version: int | None, to_version: int | None
) -> dict[str, Any]:
    """Pick the two versions and compare them.

    `versions` is the article's history, any order, each with `.version`, `.snapshot` and `.created_at`. By default
    the latest against the one before it; `from_version` alone compares that version with the latest, which is how a
    reader who read version N asks what changed since. Raises ValueError with a message for the caller.
    """
    by_number = {item.version: item for item in versions}
    if not by_number:
        raise ValueError("This article has no stored versions.")
    numbers = sorted(by_number)
    latest = numbers[-1]
    new_number = to_version if to_version is not None else latest
    if new_number not in by_number:
        raise ValueError(
            f"Version {new_number} does not exist; versions are {numbers[0]}..{latest}."
        )
    if from_version is None:
        earlier = [n for n in numbers if n < new_number]
        if not earlier:
            return {
                "first_version": True,
                "version": new_number,
                "created_at": _stamp(by_number[new_number]),
                "summary": [
                    f"Version {new_number} is the first version; there is nothing earlier to compare."
                ],
            }
        from_version = earlier[-1]
    if from_version not in by_number:
        raise ValueError(
            f"Version {from_version} does not exist; versions are {numbers[0]}..{latest}."
        )
    if from_version >= new_number:
        raise ValueError("'from_version' must be older than 'to_version'.")
    old, new = by_number[from_version], by_number[new_number]
    result = diff_versions(old.snapshot or {}, new.snapshot or {})
    between = [n for n in numbers if from_version < n <= new_number]
    return {
        "from_version": from_version,
        "to_version": new_number,
        "from_at": _stamp(old),
        "to_at": _stamp(new),
        "versions_covered": between,
        "latest_version": latest,
        **result,
    }


def _stamp(version: Any) -> str | None:
    created = getattr(version, "created_at", None)
    return created.isoformat() if created else None
