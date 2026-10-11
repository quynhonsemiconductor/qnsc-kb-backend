"""Closed, deterministic analytics over authorized extracted table evidence."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RowFilter(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    equals: str = Field(max_length=1000)


class AnalyticsPlan(BaseModel):
    operation: Literal[
        "count", "sum", "average", "median", "min", "max", "difference", "percentage_change"
    ]
    evidence_id: str = Field(min_length=1, max_length=64)
    column: str | None = Field(default=None, max_length=255)
    filters: list[RowFilter] = Field(default_factory=list, max_length=5)
    left_row: int | None = Field(default=None, ge=0)
    right_row: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> "AnalyticsPlan":
        if self.operation != "count" and not self.column:
            raise ValueError("column is required for this operation")
        if self.operation in {"difference", "percentage_change"} and (
            self.left_row is None or self.right_row is None
        ):
            raise ValueError("two row indices are required for comparison")
        return self


def parse_decimal(value: object) -> Decimal:
    """Parse conservative EN/Vietnamese numeric forms, rejecting formulas and infinity."""
    text = str(value or "").strip()
    if not text or text.startswith(("=", "+", "@")):
        raise ValueError("not a finite number")
    text = re.sub(r"[\s%₫$€£]", "", text)
    if not re.fullmatch(r"-?[0-9][0-9.,]*", text):
        raise ValueError("not a finite number")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") == 1 and len(text.rsplit(",", 1)[1]) != 3:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("not a finite number") from exc
    if not number.is_finite():
        raise ValueError("not a finite number")
    return number


def _format(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f") if normalized != normalized.to_integral() else str(normalized.quantize(Decimal(1)))


def execute_analytics_plan(
    plan: AnalyticsPlan,
    table: dict,
    *,
    source_id: str,
    max_rows: int = 500,
) -> dict:
    columns = [str(item) for item in table.get("columns", [])]
    rows = table.get("rows", [])
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ValueError("table exceeds the analytics row limit")
    index_by_name = {name: index for index, name in enumerate(columns)}
    filtered: list[tuple[int, list]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        keep = True
        for rule in plan.filters:
            column_index = index_by_name.get(rule.column)
            if column_index is None:
                raise ValueError(f"unknown filter column: {rule.column}")
            keep = column_index < len(row) and str(row[column_index]) == rule.equals
            if not keep:
                break
        if keep:
            filtered.append((row_index, row))
    if plan.operation == "count":
        result = Decimal(len(filtered))
        operands = [
            {"label": f"Row {index + 1}", "row": index, "value": "included"}
            for index, _row in filtered
        ]
        formula = f"COUNT({len(filtered)} rows)"
    else:
        column_index = index_by_name.get(str(plan.column))
        if column_index is None:
            raise ValueError(f"unknown numeric column: {plan.column}")
        numeric: list[tuple[int, Decimal]] = []
        skipped = 0
        for row_index, row in filtered:
            try:
                numeric.append((row_index, parse_decimal(row[column_index])))
            except (IndexError, ValueError):
                skipped += 1
        if not numeric:
            raise ValueError("no numeric cells matched the plan")
        if plan.operation in {"difference", "percentage_change"}:
            by_row = dict(numeric)
            if plan.left_row not in by_row or plan.right_row not in by_row:
                raise ValueError("comparison rows are not numeric or do not match filters")
            left, right = by_row[plan.left_row], by_row[plan.right_row]
            if plan.operation == "difference":
                result = right - left
                formula = f"{_format(right)} - {_format(left)}"
            else:
                if left == 0:
                    raise ValueError("percentage change from zero is undefined")
                result = ((right - left) / abs(left)) * Decimal(100)
                formula = f"(({_format(right)} - {_format(left)}) / |{_format(left)}|) × 100"
            operands = [
                {
                    "label": f"{plan.column} · row {plan.left_row + 1}",
                    "row": plan.left_row,
                    "column": plan.column,
                    "value": _format(left),
                },
                {
                    "label": f"{plan.column} · row {plan.right_row + 1}",
                    "row": plan.right_row,
                    "column": plan.column,
                    "value": _format(right),
                },
            ]
        else:
            values = [value for _row, value in numeric]
            if plan.operation == "sum":
                result = sum(values, Decimal(0))
                formula = " + ".join(_format(value) for value in values)
            elif plan.operation == "average":
                result = sum(values, Decimal(0)) / Decimal(len(values))
                formula = f"SUM({len(values)} values) / {len(values)}"
            elif plan.operation == "median":
                ordered = sorted(values)
                middle = len(ordered) // 2
                result = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / Decimal(2)
                formula = f"MEDIAN({len(values)} values)"
            elif plan.operation == "min":
                result = min(values)
                formula = f"MIN({len(values)} values)"
            elif plan.operation == "max":
                result = max(values)
                formula = f"MAX({len(values)} values)"
            else:
                raise ValueError("unsupported analytics operation")
            operands = [
                {
                    "label": f"{plan.column} · row {row + 1}",
                    "row": row,
                    "column": plan.column,
                    "value": _format(value),
                }
                for row, value in numeric
            ]
        warnings = [f"{skipped} non-numeric cell(s) omitted"] if skipped else []
    return {
        "operation": plan.operation,
        "evidence_id": plan.evidence_id,
        "source_id": source_id,
        "column": plan.column,
        "operands": operands,
        "formula": formula,
        "result": _format(result),
        "rows_used": len(operands),
        "warnings": locals().get("warnings", []),
    }
