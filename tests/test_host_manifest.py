from __future__ import annotations

import json

import pytest

from aethergraph.contracts.integration import HostManifest
from aethergraph.services.host import (
    HostManifestError,
    compute_host_manifest_digest,
    load_host_manifest,
    seal_host_manifest,
)

_DIGEST = "a" * 64


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
