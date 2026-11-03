from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.domain.evidence_analytics import (
    AnalyticsPlan,
    RowFilter,
    execute_analytics_plan,
    parse_decimal,
)


TABLE = {
    "columns": ["Quarter", "Revenue"],
    "rows": [["Q1", "1,200.50"], ["Q2", "1.500,50"], ["Q3", "not reported"]],
}


def test_parse_decimal_supports_bounded_en_and_vietnamese_formats():
    assert parse_decimal("1,200.50") == Decimal("1200.50")
    assert parse_decimal("1.500,50") == Decimal("1500.50")
    assert parse_decimal("12,5%") == Decimal("12.5")


@pytest.mark.parametrize("value", ["=SUM(A1:A3)", "NaN", "Infinity", "__import__('os')", "1e9"])
def test_parse_decimal_rejects_formulas_code_and_non_finite_values(value):
    with pytest.raises(ValueError):
        parse_decimal(value)


def test_average_is_decimal_and_reports_omitted_cells():
    result = execute_analytics_plan(
        AnalyticsPlan(operation="average", evidence_id="evidence-1", column="Revenue"),
        TABLE,
        source_id="C2",
    )

    assert result["result"] == "1350.5"
    assert result["formula"] == "SUM(2 values) / 2"
    assert result["warnings"] == ["1 non-numeric cell(s) omitted"]
    assert result["source_id"] == "C2"


def test_exact_match_filter_and_difference_preserve_source_rows():
    result = execute_analytics_plan(
        AnalyticsPlan(
            operation="difference",
            evidence_id="evidence-1",
            column="Revenue",
            left_row=0,
            right_row=1,
        ),
        TABLE,
        source_id="C1",
    )

    assert result["result"] == "300"
    assert result["operands"] == [
        {
            "label": "Revenue · row 1",
            "row": 0,
            "column": "Revenue",
            "value": "1200.5",
        },
        {
            "label": "Revenue · row 2",
            "row": 1,
            "column": "Revenue",
            "value": "1500.5",
        },
    ]


def test_count_uses_only_exactly_filtered_rows():
    result = execute_analytics_plan(
        AnalyticsPlan(
            operation="count",
            evidence_id="evidence-1",
            filters=[RowFilter(column="Quarter", equals="Q2")],
        ),
        TABLE,
        source_id="C1",
    )

    assert result["result"] == "1"
    assert result["rows_used"] == 1


def test_unknown_operations_and_oversized_tables_are_rejected():
    with pytest.raises(ValidationError):
        AnalyticsPlan(operation="python", evidence_id="evidence-1")
    with pytest.raises(ValueError, match="row limit"):
        execute_analytics_plan(
            AnalyticsPlan(operation="count", evidence_id="evidence-1"),
            {"columns": ["A"], "rows": [[index] for index in range(3)]},
            source_id="C1",
            max_rows=2,
        )


def test_percentage_change_from_zero_is_rejected():
    with pytest.raises(ValueError, match="from zero"):
        execute_analytics_plan(
            AnalyticsPlan(
                operation="percentage_change",
                evidence_id="evidence-1",
                column="Value",
                left_row=0,
                right_row=1,
            ),
            {"columns": ["Value"], "rows": [["0"], ["10"]]},
            source_id="C1",
        )
