"""Release compatibility construction shared by Studio and AG Host."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from aethergraph.contracts.integration import (
    ReleaseCompatibility,
    ReleaseDependency,
)

HOST_CAPABILITIES = frozenset({"canonical_ingress", "origin_delivery", "semantic_event_log"})
HOST_SERVICES = frozenset(
    {"artifacts", "channels", "continuations", "eventlog", "runs", "sessions"}
)
ENTRYPOINT_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["message", "attachments", "session_id", "user_meta"],
}
ENTRYPOINT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "agent_outcome", "workflow_outcome"],
}


def build_release_compatibility(
    *,
    build_root: str | Path,
    aethergraph_version: str,
    engine_version: str,
    python_abi: str,
    platform_name: str,
    architecture: str,
    dependency_lock: tuple[ReleaseDependency, ...],
    dependency_lock_digest: str,
) -> ReleaseCompatibility:
    """Build exact compatibility evidence for one verified compiled release.

    Intro:
        The function inspects compiler-owned metadata without importing generated
        code, binds it to one probed Host environment, and hashes the compiled
        manifest used as release provenance.

    Examples:
        Build local compatibility evidence:
        ```python
        compatibility = build_release_compatibility(
            build_root=compiled.output_root,
            aethergraph_version="0.1.0a16",
            engine_version="0.1.0a1",
            python_abi="cpython-313",
            platform_name="win32",
            architecture="AMD64",
            dependency_lock=lock,
            dependency_lock_digest=digest,
        )
        ```

        Inspect compiler provenance:
        ```python
        assert compatibility.provenance["build_id"] == compiled.build_id
        assert len(compatibility.compiled_manifest_sha256) == 64
        ```

    Args:
        build_root: Exact compiler-owned build directory.
        aethergraph_version: Probed AG distribution version.
        engine_version: Probed Engine distribution version.
        python_abi: Probed Python implementation cache tag.
        platform_name: Probed Python platform identifier.
        architecture: Probed machine architecture.
        dependency_lock: Exact installed Host distribution lock.
        dependency_lock_digest: Canonical digest of the dependency lock.

    Returns:
        ReleaseCompatibility: Closed pre-import Host compatibility contract.

    Notes:
        Generated source is never imported by this function. Compiler integrity
        inspection rejects stale versions and changed file checksums.
    """
    from aethergraph_engine.compiler import inspect_compiled_build

    root = Path(build_root).resolve(strict=True)
    inspection = inspect_compiled_build(root)
    compiled = inspection.manifest
    if compiled.engine_version != engine_version:
        raise ValueError("Compiled Engine version does not match the selected Host environment.")
    manifest_path = root / "manifest.json"
    return ReleaseCompatibility(
        aethergraph_version=aethergraph_version,
        engine_version=engine_version,
        python_abi=python_abi,
        platform=platform_name,
        architecture=architecture,
        dependency_lock=dependency_lock,
        dependency_lock_digest=dependency_lock_digest,
        host_capability_requirements=tuple(sorted(HOST_CAPABILITIES)),
        service_requirements=tuple(sorted(HOST_SERVICES)),
        logical_output_requirements=compiled.logical_output_requirements,
        entrypoint_input_schema=ENTRYPOINT_INPUT_SCHEMA,
        entrypoint_output_schema=ENTRYPOINT_OUTPUT_SCHEMA,
        compiled_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        provenance={
            "build_id": compiled.build_id,
            "source_digest": compiled.source_digest,
            "engine_version": compiled.engine_version,
            "compiler_version": compiled.compiler_version,
            "catalog_digest": compiled.catalog_digest,
        },
    )


__all__ = [
    "ENTRYPOINT_INPUT_SCHEMA",
    "ENTRYPOINT_OUTPUT_SCHEMA",
    "HOST_CAPABILITIES",
    "HOST_SERVICES",
    "build_release_compatibility",
]
