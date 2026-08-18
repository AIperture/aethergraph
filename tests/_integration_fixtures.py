from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from aethergraph import __version__ as AETHERGRAPH_VERSION
from aethergraph.contracts.integration import ReleaseCompatibility, ReleaseDependency
from aethergraph.services.host import build_release_compatibility

_DIGEST = "a" * 64


def suite_repository_path(name: str) -> Path:
    """Resolve a sibling repository from either the main checkout or a worktree."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    raise AssertionError(f"Suite repository not found: {name}")


def contract_compatibility() -> ReleaseCompatibility:
    return ReleaseCompatibility(
        aethergraph_version=AETHERGRAPH_VERSION,
        engine_version="0.1.0a1",
        python_abi=sys.implementation.cache_tag or "unknown",
        platform=sys.platform,
        architecture=platform.machine() or platform.architecture()[0],
        dependency_lock=(
            ReleaseDependency(
                name="aethergraph",
                version=AETHERGRAPH_VERSION,
                content_sha256=_DIGEST,
            ),
        ),
        dependency_lock_digest=_DIGEST,
        host_capability_requirements=(
            "canonical_ingress",
            "origin_delivery",
            "semantic_event_log",
        ),
        service_requirements=(
            "artifacts",
            "channels",
            "continuations",
            "eventlog",
            "runs",
            "sessions",
        ),
        logical_output_requirements=("origin",),
        entrypoint_input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["message", "attachments", "session_id", "user_meta"],
        },
        entrypoint_output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["reply", "agent_outcome", "workflow_outcome"],
        },
        compiled_manifest_sha256=_DIGEST,
        provenance={"fixture": "contract"},
    )


def runtime_compatibility(build_root: str | Path) -> ReleaseCompatibility:
    lock = tuple(_installed_dependency(name) for name in ("aethergraph", "aethergraph-engine"))
    digest = sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in lock],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return build_release_compatibility(
        build_root=build_root,
        aethergraph_version=lock[0].version,
        engine_version=lock[1].version,
        python_abi=sys.implementation.cache_tag or "unknown",
        platform_name=sys.platform,
        architecture=platform.machine() or platform.architecture()[0],
        dependency_lock=lock,
        dependency_lock_digest=digest,
    )


def runtime_identity_payload(compatibility: ReleaseCompatibility) -> dict[str, str]:
    return {
        "aethergraph_version": compatibility.aethergraph_version,
        "engine_version": compatibility.engine_version,
        "python_abi": compatibility.python_abi,
        "platform": compatibility.platform,
        "architecture": compatibility.architecture,
        "dependency_lock_digest": compatibility.dependency_lock_digest,
    }


def _installed_dependency(name: str) -> ReleaseDependency:
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if not record:
        raise AssertionError(f"{name} has no RECORD metadata")
    return ReleaseDependency(
        name=name,
        version=distribution.version,
        content_sha256=sha256(record.encode()).hexdigest(),
    )
