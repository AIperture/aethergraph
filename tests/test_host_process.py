from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from aethergraph_engine.compiler import compile_system_project
import httpx

from aethergraph.contracts.integration import (
    HostManifest,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    SemanticEventKind,
)
from aethergraph.services.host import seal_host_manifest
from tests._integration_fixtures import runtime_compatibility, runtime_identity_payload

_FIXTURE = (
    Path(__file__).parents[2] / "ag-engine" / "tests" / "fixtures" / "authoring" / "plain_react_v1"
)
_DIGEST = "a" * 64
_TOKEN = "host-process-control-token-with-32-characters"


def test_host_command_launches_verified_build_and_authenticated_health(tmp_path) -> None:
    compiled = compile_system_project(_FIXTURE, output_root=tmp_path / "builds")
    resolved = compiled.resolved_definition
    compatibility = runtime_compatibility(compiled.output_root)
    route = IntegrationRoute(
        route_id="route-ui",
        endpoint_id="endpoint-ui",
        integration_id="ag-ui",
        integration_kind=IntegrationKind.AG_UI,
        entry_agent_id=resolved.system_id,
        enabled=True,
        match_policy=IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope="conversation_user"),
        required_capabilities=IntegrationCapabilities(
            event_kinds=(
                SemanticEventKind.MESSAGE_COMPLETED,
                SemanticEventKind.INTERACTION_REQUESTED,
            ),
            streaming=False,
            interactions=True,
            attachments=True,
            cancellation=True,
        ),
    )
    settings_path = tmp_path / "application.env"
    settings_path.write_bytes(b"")
    settings_digest = sha256(b"").hexdigest()
    manifest = seal_host_manifest(
        HostManifest(
            deployment_id="deployment-process-test",
            build_id=compiled.build_id,
            source_digest=compiled.source_digest,
            build_root=str(compiled.output_root),
            entrypoint_module=compiled.entrypoint_module,
            entrypoint_symbol=compiled.entrypoint_symbol,
            graph_id=resolved.surface.graph_fn_name,
            entry_agent_id=resolved.system_id,
            environment_snapshot_digest=_DIGEST,
            runtime_profile_digest=_DIGEST,
            application_settings_digest=settings_digest,
            release_compatibility=compatibility,
            integration_routes=(route,),
            workspace_identity="workspace-process-test",
            manifest_digest="0" * 64,
        )
    )
    manifest_path = tmp_path / "host-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    identity_path = tmp_path / "runtime-identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "environment_snapshot_digest": _DIGEST,
                "runtime_profile_digest": _DIGEST,
                "application_settings_digest": settings_digest,
                **runtime_identity_payload(compatibility),
            }
        ),
        encoding="utf-8",
    )
    token_path = tmp_path / "control-token.handle"
    token_path.write_text(_TOKEN, encoding="utf-8")
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aethergraph",
            "host",
            "--manifest",
            str(manifest_path),
            "--runtime-identity",
            str(identity_path),
            "--settings",
            str(settings_path),
            "--workspace",
            str(tmp_path / "runtime"),
            "--control-token",
            str(token_path),
        ],
        cwd=Path(__file__).parents[1],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            line = executor.submit(process.stdout.readline).result(timeout=75)
        if not line:
            stderr = process.stderr.read()
            raise AssertionError(f"Host exited before handshake: {stderr}")
        handshake = json.loads(line)
        assert handshake["schema_version"] == "aethergraph.host-ready/v1"
        endpoint_credential = handshake["endpoint_credentials"]["endpoint-ui"]
        response = httpx.get(
            f"{handshake['base_url']}/_host/ready",
            headers={"X-AG-Host-Control": _TOKEN},
            timeout=10,
        )
        assert response.status_code == 200
        assert response.json()["build_id"] == compiled.build_id
        unauthenticated = httpx.post(
            f"{handshake['base_url']}/api/v1/agent-endpoints/endpoint-ui/sessions",
            json={"idempotency_key": "host-browser"},
            timeout=10,
        )
        assert unauthenticated.status_code == 401
        with httpx.Client(timeout=10) as browser:
            authenticated = browser.post(
                f"{handshake['base_url']}/api/v1/agent-endpoints/endpoint-ui/authenticate",
                headers={"Authorization": f"Bearer {endpoint_credential}"},
            )
            assert authenticated.status_code == 200
            created_session = browser.post(
                f"{handshake['base_url']}/api/v1/agent-endpoints/endpoint-ui/sessions",
                json={"idempotency_key": "host-browser"},
            )
            assert created_session.status_code == 200
        assert httpx.get(f"{handshake['base_url']}/api/v1/agents", timeout=10).status_code == 200
        assert (
            httpx.post(
                f"{handshake['base_url']}/api/v1/runs",
                json={},
                timeout=10,
            ).status_code
            == 405
        )
        assert (
            httpx.post(
                f"{handshake['base_url']}/api/v1/registry/register",
                json={},
                timeout=10,
            ).status_code
            == 404
        )
        shutdown = httpx.post(
            f"{handshake['base_url']}/_host/shutdown",
            headers={"X-AG-Host-Control": _TOKEN},
            timeout=10,
        )
        assert shutdown.status_code == 202
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
