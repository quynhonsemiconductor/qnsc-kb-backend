"""Log lines are compact JSON so the Grafana Cloud Loki derived field links trace_id → Tempo."""

import json
import re

import structlog

from src.lib import observability

# The regex of the Grafana-Cloud-provisioned derived field on the Loki datasource (read-only).
DERIVED_FIELD = re.compile(r'[tT]race_?[iI][dD]"?[:=]"?(\w+)')
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"


def test_json_lines_are_compact_and_match_the_loki_derived_field(capsys):
    observability.setup_logging()
    structlog.get_logger("compact-json-test").info("probe", trace_id=TRACE_ID)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["trace_id"] == TRACE_ID
    assert '": ' not in line
    match = DERIVED_FIELD.search(line)
    assert match and match.group(1) == TRACE_ID
