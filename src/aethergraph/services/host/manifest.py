"""Immutable AG Host manifest loading and compiled-build verification."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aethergraph.contracts.integration import HostManifest


class HostManifestError(RuntimeError):
    """Report a closed, deterministic Host manifest validation failure."""


def compute_host_manifest_digest(manifest: HostManifest) -> str:
    """Compute the canonical digest for one immutable Host manifest.

    The digest covers every manifest field except `manifest_digest` using
    sorted compact JSON. It is the sole digest algorithm accepted by AG Host.

    Examples:
        Compute a manifest digest:
            ```python
            digest = compute_host_manifest_digest(manifest)
            assert len(digest) == 64
            ```

        Seal a copied manifest:
            ```python
            sealed = manifest.model_copy(
                update={"manifest_digest": compute_host_manifest_digest(manifest)}
            )
            ```

    Args:
        manifest: Validated Host manifest whose identity will be hashed.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest of canonical manifest data.

    Notes:
        The supplied `manifest_digest` value never contributes to the digest.
    """

    payload = manifest.model_dump(mode="json", exclude={"manifest_digest"})
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def seal_host_manifest(manifest: HostManifest) -> HostManifest:
    """Return a Host manifest carrying its canonical self digest.

    This helper is used by control planes after all immutable launch fields
    have been selected. It does not mutate the frozen input model.

    Examples:
        Seal a newly created manifest:
            ```python
            sealed = seal_host_manifest(manifest)
            assert sealed.manifest_digest != manifest.manifest_digest
            ```

        Verify the sealed identity:
            ```python
            sealed = seal_host_manifest(manifest)
            assert sealed.manifest_digest == compute_host_manifest_digest(sealed)
            ```

    Args:
        manifest: Fully populated Host manifest to seal.

    Returns:
        HostManifest: Frozen copy with the canonical `manifest_digest` value.

    Notes:
        Resealing is deterministic and replaces any prior digest value.
    """

    return manifest.model_copy(update={"manifest_digest": compute_host_manifest_digest(manifest)})


def load_host_manifest(path: str | Path) -> HostManifest:
    """Load and verify one exact Host manifest JSON file.

    The loader validates the closed schema and requires the stored digest to
    match the canonical manifest content before returning any launch data.

    Examples:
        Load a Studio-authored manifest:
            ```python
            manifest = load_host_manifest("deployment/host-manifest.json")
            ```

        Reject modified launch data:
            ```python
            try:
                load_host_manifest("deployment/tampered.json")
            except HostManifestError:
                pass
            ```

    Args:
        path: Exact JSON file selected by the Host supervisor.

    Returns:
        HostManifest: Validated manifest with a verified self digest.

    Notes:
        Directories, missing files, malformed JSON, and digest mismatches fail
        closed and never trigger source discovery.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise HostManifestError(f"Host manifest is not a file: {manifest_path}")
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = HostManifest.model_validate(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise HostManifestError(f"Invalid Host manifest: {manifest_path}") from exc
    expected = compute_host_manifest_digest(manifest)
    if manifest.manifest_digest != expected:
        raise HostManifestError("Host manifest digest does not match its canonical content.")
    return manifest


def validate_compiled_build(manifest: HostManifest):
    """Inspect and match one compiler-owned build to a Host manifest.

    AG Engine verifies the complete compiler file index first. AG Host then
    compares every launch identity it consumes and rejects route targets that
    do not name the generated package's public AG agent.

    Examples:
        Validate a local compiled build:
            ```python
            inspection = validate_compiled_build(manifest)
            assert inspection.manifest.build_id == manifest.build_id
            ```

        Reject release handles in the local Host:
            ```python
            try:
                validate_compiled_build(remote_manifest)
            except HostManifestError:
                pass
            ```

    Args:
        manifest: Digest-verified Host launch manifest.

    Returns:
        CompiledBuildInspection: Engine-owned verified build inspection.

    Notes:
        The initial local Host accepts `build_root` only. Release acquisition is
        a separate future control-plane operation and has no local fallback.
    """

    if manifest.build_root is None:
        raise HostManifestError("Local AG Host requires an exact build_root.")
    build_root = Path(manifest.build_root).expanduser().resolve()
    if not build_root.is_dir():
        raise HostManifestError(f"Compiled build root is not a directory: {build_root}")
    if build_root.name != manifest.build_id:
        raise HostManifestError("Compiled build directory does not match manifest build_id.")
    try:
        from aethergraph_engine.compiler import inspect_compiled_build
    except ImportError as exc:
        raise HostManifestError("Compatible aethergraph-engine is required by AG Host.") from exc
    try:
        inspection = inspect_compiled_build(build_root)
    except Exception as exc:  # Engine exposes a versioned CompilationError type.
        raise HostManifestError("Compiled build integrity verification failed.") from exc

    compiled = inspection.manifest
    resolved = inspection.resolved_definition
    identities: Mapping[str, tuple[str, str]] = {
        "build_id": (compiled.build_id, manifest.build_id),
        "source_digest": (compiled.source_digest, manifest.source_digest),
        "entrypoint_module": (compiled.entrypoint_module, manifest.entrypoint_module),
        "entrypoint_symbol": (compiled.entrypoint_symbol, manifest.entrypoint_symbol),
        "graph_id": (resolved.surface.graph_fn_name, manifest.graph_id),
        "entry_agent_id": (resolved.system_id, manifest.entry_agent_id),
    }
    mismatched = [name for name, values in identities.items() if values[0] != values[1]]
    if mismatched:
        raise HostManifestError("Compiled build identity mismatch: " + ", ".join(mismatched))
    if not any(agent.resource_ref == resolved.entry_agent_ref for agent in resolved.agents):
        raise HostManifestError("Compiled build entry_agent_ref is unresolved.")
    invalid_routes = sorted(
        route.route_id
        for route in manifest.integration_routes
        if route.entry_agent_id != manifest.entry_agent_id
    )
    if invalid_routes:
        raise HostManifestError(
            "Integration routes target agents outside this compiled build: "
            + ", ".join(invalid_routes)
        )
    return inspection


__all__ = [
    "HostManifestError",
    "compute_host_manifest_digest",
    "load_host_manifest",
    "seal_host_manifest",
    "validate_compiled_build",
]
