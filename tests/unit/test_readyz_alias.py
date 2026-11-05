"""/readyz answers the contract's own body — the platform probe name, not an alias.

The route table is asserted, not a served request: standing the app up needs the
local stack (see test_source_preview_headers.py), and the property under test is
wiring, not transport. The contract body and the draining behaviour live in
test_shutdown_contract.py; here we assert the platform paths exist exactly once
and that the legacy ECS-era endpoints the deploy pipeline polls stay in place.
"""

import src.api.main as main


def _get_routes(path: str):
    return [
        route
        for route in main.app.routes
        if getattr(route, "path", None) == path
        and "GET" in getattr(route, "methods", set())
    ]


def test_the_platform_readiness_paths_exist_exactly_once():
    assert len(_get_routes("/readyz")) == 1
    assert len(_get_routes("/v1/readyz")) == 1
    assert len(_get_routes("/v1/healthz")) == 1


def test_the_legacy_readiness_endpoints_stay_for_the_ecs_estate():
    assert len(_get_routes("/health/ready")) == 1
    assert len(_get_routes("/health/deps")) == 1
    assert len(_get_routes("/health/live")) == 1
