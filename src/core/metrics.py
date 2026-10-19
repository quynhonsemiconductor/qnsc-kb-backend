import re
from collections import defaultdict
from threading import Lock

_lock = Lock()
_counts: dict[tuple[str, str, int], int] = defaultdict(int)
_durations: dict[tuple[str, str], tuple[int, float]] = defaultdict(lambda: (0, 0.0))


def normalize_path(path: str) -> str:
    return re.sub(
        r"/[0-9a-f]{8}-[0-9a-f-]{27,36}(?=/|$)", "/:id", path, flags=re.IGNORECASE
    )


def record_request(
    method: str, path: str, status_code: int, duration_ms: float
) -> None:
    key_path = normalize_path(path)
    with _lock:
        _counts[(method, key_path, status_code)] += 1
        count, total = _durations[(method, key_path)]
        _durations[(method, key_path)] = (count + 1, total + duration_ms)


def prometheus_text() -> str:
    lines = [
        "# HELP qnsc_http_requests_total Total HTTP requests.",
        "# TYPE qnsc_http_requests_total counter",
    ]
    with _lock:
        for (method, path, status), count in sorted(_counts.items()):
            lines.append(
                f'qnsc_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP qnsc_http_request_duration_ms_sum HTTP request duration sum in milliseconds.",
                "# TYPE qnsc_http_request_duration_ms_sum counter",
            ]
        )
        for (method, path), (count, total) in sorted(_durations.items()):
            lines.append(
                f'qnsc_http_request_duration_ms_sum{{method="{method}",path="{path}"}} {total:.3f}'
            )
            lines.append(
                f'qnsc_http_request_duration_ms_count{{method="{method}",path="{path}"}} {count}'
            )
    return "\n".join(lines) + "\n"
