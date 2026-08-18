from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aethergraph.server.app_factory import create_app
from aethergraph.services.host.development import (
    build_development_ui_manifest,
    development_ui_endpoints,
)


class _Registry:
    def list_agents(self, *, include_global: bool):
        assert include_global is True
        return {"agent:chat": "0.1.0", "agent:reviewer": "0.1.0"}

    def get_meta(self, *, nspace: str, name: str, include_global: bool):
        assert nspace == "agent"
        assert include_global is True
        return {"id": f"agent.{name}"}


def test_development_manifest_builds_one_endpoint_per_agent() -> None:
    manifest = build_development_ui_manifest(
        registry=_Registry(),
        workspace_identity="C:/workspace",
    )

    endpoints = development_ui_endpoints(manifest)

    assert manifest.manifest_digest != "0" * 64
    assert manifest.entry_agent_id == "agent.chat"
    assert endpoints == (
        {
            "agent_id": "agent.chat",
            "endpoint_id": manifest.integration_routes[0].endpoint_id,
        },
        {
            "agent_id": "agent.reviewer",
            "endpoint_id": manifest.integration_routes[1].endpoint_id,
        },
    )
    assert len({item["endpoint_id"] for item in endpoints}) == 2


def test_development_app_exposes_ui_bootstrap(tmp_path: Path) -> None:
    app = create_app(workspace=str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/v1/ui/bootstrap")
        payload = response.json()
        endpoint_id = payload["endpoints"][0]["endpoint_id"]
        credential = app.state.endpoint_credentials.take_launch_credentials()[endpoint_id]
        authenticated = client.post(
            f"/api/v1/agent-endpoints/{endpoint_id}/authenticate",
            headers={"Authorization": f"Bearer {credential}"},
        )
        created = client.post(
            f"/api/v1/agent-endpoints/{endpoint_id}/sessions",
            json={"idempotency_key": "development-browser"},
        )
        listed = client.get(f"/api/v1/agent-endpoints/{endpoint_id}/sessions")

    assert response.status_code == 200
    assert payload["mode"] == "development"
    assert payload["default_agent_id"] == "chat_agent"
    assert payload["endpoints"] == [
        {
            "agent_id": "chat_agent",
            "endpoint_id": payload["endpoints"][0]["endpoint_id"],
        }
    ]
    assert authenticated.status_code == 200
    assert created.status_code == 200
    assert [item["session_id"] for item in listed.json()["items"]] == [created.json()["session_id"]]
