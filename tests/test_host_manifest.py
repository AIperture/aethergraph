from __future__ import annotations

import json
from pathlib import Path

from aethergraph_engine.compiler import compile_system_project
import pytest

from aethergraph.config.config import AppSettings
from aethergraph.contracts.integration import HostManifest
from aethergraph.services.host import (
    HostCompatibilityError,
    HostManifestError,
    HostRuntimeIdentity,
    build_host,
    compute_host_manifest_digest,
    load_host_manifest,
    seal_host_manifest,
)
from tests._integration_fixtures import (
    contract_compatibility,
    runtime_compatibility,
    runtime_identity_payload,
)

_DIGEST = "a" * 64
_FIXTURE = (
    Path(__file__).parents[2] / "ag-engine" / "tests" / "fixtures" / "authoring" / "plain_react_v1"
)


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
        manifest_digest="0" * 64,
    )


def test_seal_and_load_host_manifest(tmp_path) -> None:
    sealed = seal_host_manifest(_manifest())
    path = tmp_path / "host-manifest.json"
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_host_manifest(path)

    assert loaded == sealed
    assert loaded.manifest_digest == compute_host_manifest_digest(loaded)


def test_load_host_manifest_rejects_tampered_content(tmp_path) -> None:
    sealed = seal_host_manifest(_manifest())
    payload = sealed.model_dump(mode="json")
    payload["deployment_id"] = "deployment-tampered"
    path = tmp_path / "host-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HostManifestError, match="digest"):
        load_host_manifest(path)


def test_host_rejects_incompatible_release_before_entrypoint_import(tmp_path) -> None:
    compiled = compile_system_project(_FIXTURE, output_root=tmp_path / "builds")
    resolved = compiled.resolved_definition
    compatible = runtime_compatibility(compiled.output_root)
    incompatible = compatible.model_copy(update={"engine_version": "99.0.0"})
    manifest = seal_host_manifest(
        HostManifest(
            deployment_id="deployment-incompatible",
            build_id=compiled.build_id,
            source_digest=compiled.source_digest,
            build_root=str(compiled.output_root),
            entrypoint_module="entrypoint_must_not_be_imported",
            entrypoint_symbol=compiled.entrypoint_symbol,
            graph_id=resolved.surface.graph_fn_name,
            entry_agent_id=resolved.system_id,
            environment_snapshot_digest=_DIGEST,
            runtime_profile_digest=_DIGEST,
            application_settings_digest=_DIGEST,
            release_compatibility=incompatible,
            integration_routes=(),
            workspace_identity="workspace-incompatible",
            manifest_digest="0" * 64,
        )
    )
    runtime_identity = HostRuntimeIdentity(
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        **runtime_identity_payload(compatible),
    )

    with pytest.raises(
        HostCompatibilityError,
        match="select the pinned interpreter or rebuild the release",
    ):
        build_host(
            manifest=manifest,
            runtime_identity=runtime_identity,
            workspace=tmp_path / "runtime",
            settings=AppSettings(),
        )
