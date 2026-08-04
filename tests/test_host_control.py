from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aethergraph.contracts.integration import HostManifest, IntegrationKind
from aethergraph.services.host import install_host_control_routes
from aethergraph.services.integration import (
    IntegrationConnectionState,
    IntegrationConnectionStatus,
)
from tests._integration_fixtures import contract_compatibility

_DIGEST = "a" * 64
_TOKEN = "control-token-with-more-than-32-characters"


def _manifest() -> HostManifest:
    return HostManifest(
        deployment_id="deployment-1",
        build_id="0123456789abcdef01234567",
        source_digest=_DIGEST,
        build_root="C:/build/0123456789abcdef01234567",
        entrypoint_module="demo_compiled.entry",
        entrypoint_symbol="demo_entry",
        graph_id="demo.graph",
        entry_agent_id="demo",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        release_compatibility=contract_compatibility(),
        integration_routes=(),
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


def _app(*, ready: bool) -> FastAPI:
    status = IntegrationConnectionStatus(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        state=(IntegrationConnectionState.READY if ready else IntegrationConnectionState.FAILED),
        error_code=None if ready else "integration.transport_failed",
    )
    manager = SimpleNamespace(ready=ready, statuses=lambda: (status,))
    host = SimpleNamespace(manifest=_manifest(), integration_manager=manager)
    app = FastAPI()
    install_host_control_routes(app=app, host=host, control_token=_TOKEN)
    return app


def test_host_control_requires_launch_token_and_reports_readiness() -> None:
    with TestClient(_app(ready=True)) as client:
        assert client.get("/_host/ready").status_code == 401

        response = client.get(
            "/_host/ready",
            headers={"X-AG-Host-Control": _TOKEN},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "deployment_id": "deployment-1",
        "build_id": "0123456789abcdef01234567",
        "manifest_digest": _DIGEST,
        "ready": True,
    }


def test_host_control_returns_redacted_failed_diagnostics() -> None:
    with TestClient(_app(ready=False)) as client:
        response = client.get(
            "/_host/diagnostics",
            headers={"X-AG-Host-Control": _TOKEN},
        )

    assert response.status_code == 503
    assert response.json()["providers"] == [
        {
            "integration_id": "slack-main",
            "integration_kind": "slack",
            "state": "failed",
            "error_code": "integration.transport_failed",
        }
    ]
    assert "build_root" not in response.text
    assert "C:/build" not in response.text


def test_host_control_shutdown_requires_token_and_callback() -> None:
    app = _app(ready=True)
    requested: list[bool] = []
    app.state.host_shutdown = lambda: requested.append(True)
    with TestClient(app) as client:
        assert client.post("/_host/shutdown").status_code == 401
        response = client.post(
            "/_host/shutdown",
            headers={"X-AG-Host-Control": _TOKEN},
        )

    assert response.status_code == 202
    assert requested == [True]
