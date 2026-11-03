from src.core.metrics import normalize_path, prometheus_text, record_request
from src.api.main import _metric_path_for, _request_id_from


class _Request:
    def __init__(self, headers=None, route=None):
        self.headers = headers or {}
        self.scope = {"route": route} if route else {}


class _Route:
    path = "/api/v1/articles/{id}"


def test_metrics_normalize_ids_and_export():
    assert normalize_path("/api/v1/articles/123e4567-e89b-12d3-a456-426614174000") == "/api/v1/articles/:id"
    record_request("GET", "/api/v1/health/live", 200, 3.5)
    output = prometheus_text()
    assert "qnsc_http_requests_total" in output
    assert 'status="200"' in output


def test_middleware_uses_a_bounded_route_template_and_safe_request_id():
    assert _metric_path_for(_Request(route=_Route())) == "/api/v1/articles/{id}"
    assert _metric_path_for(_Request()) == "/unmatched"
    assert _request_id_from(_Request(headers={"X-Request-ID": "trace-123"})) == "trace-123"
    assert _request_id_from(_Request(headers={"X-Request-ID": "not valid!"})) != "not valid!"
