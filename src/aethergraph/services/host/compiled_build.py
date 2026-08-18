"""Engine-independent inspection of immutable compiled build artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class CompiledBuildError(RuntimeError):
    """Report a closed compiled-build validation or integrity failure."""


class _ArtifactContract(BaseModel):
    """Base model for immutable, closed compiled-artifact records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CompiledFile(_ArtifactContract):
    """One file declared by an immutable compiled-build manifest."""

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["authored", "generated"] = "generated"

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Require a normalized path contained beneath the build root.

        Intro:
            Manifest paths are safe to join only when they use normalized
            relative POSIX syntax without parent traversal.

        Examples:
            Accept a generated module:
                ```python
                item = CompiledFile(path="src/demo/entry.py", size=0, sha256="0" * 64)
                ```

            Reject a traversal path:
                ```python
                try:
                    CompiledFile(path="../entry.py", size=0, sha256="0" * 64)
                except ValueError:
                    pass
                ```

        Args:
            value: Candidate manifest-relative file path.

        Returns:
            str: Unchanged normalized relative path.

        Notes:
            Backslashes and absolute paths are rejected on every platform.
        """

        if not value or "\\" in value:
            raise ValueError("compiled file paths must use non-empty POSIX syntax")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("compiled file path must be normalized and contained")
        return value


class CompiledBuildManifest(_ArtifactContract):
    """Runtime-consumed projection of an Engine compiled-build manifest."""

    schema_version: Literal["aethergraph.compiled-system-manifest/v12"]
    build_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    package_name: str
    entrypoint_module: str
    entrypoint_symbol: str
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_version: str
    compiler_version: str
    semantic_event_protocol_version: Literal["aethergraph.semantic-event/v2"]
    logical_output_requirements: tuple[Literal["origin"], ...]
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_definition_path: Literal["resolved-system.json"]
    files: tuple[CompiledFile, ...]
    manifest_self_hash_excluded: Literal[True]

    @model_validator(mode="after")
    def _validate_file_index(self) -> CompiledBuildManifest:
        """Require a sorted, unique index that excludes the manifest itself.

        Intro:
            The manifest is self-hash-excluded, while every other build file
            must appear exactly once in deterministic path order.

        Examples:
            Validate a compiler-produced manifest:
                ```python
                checked = CompiledBuildManifest.model_validate(payload)
                ```

            Reject a duplicated file path:
                ```python
                try:
                    CompiledBuildManifest.model_validate(duplicate_payload)
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed compiled-build manifest.

        Returns:
            CompiledBuildManifest: Unchanged manifest with a valid file index.

        Notes:
            `manifest.json` cannot declare its own cryptographic checksum.
        """

        paths = tuple(item.path for item in self.files)
        if "manifest.json" in paths:
            raise ValueError("manifest.json must not hash itself")
        if len(paths) != len(set(paths)):
            raise ValueError("compiled file paths must be unique")
        if paths != tuple(sorted(paths)):
            raise ValueError("compiled file paths must be sorted")
        return self


class _ProjectionContract(BaseModel):
    """Base model for fields AG consumes from the resolved definition."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class _ResolvedSurface(_ProjectionContract):
    graph_fn_name: str


class _ResolvedAgent(_ProjectionContract):
    resource_ref: str


class ResolvedBuildIdentity(_ProjectionContract):
    """Host-required identity projected from the Engine resolved definition."""

    schema_version: Literal["aethergraph.resolved-system/v10"]
    semantic_event_protocol_version: Literal["aethergraph.semantic-event/v2"]
    logical_output_requirements: tuple[Literal["origin"], ...]
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_id: str
    entry_agent_ref: str
    surface: _ResolvedSurface
    agents: tuple[_ResolvedAgent, ...]


class CompiledBuildInspection(_ArtifactContract):
    """Verified runtime view of one immutable compiled build."""

    manifest: CompiledBuildManifest
    resolved_definition: ResolvedBuildIdentity


def inspect_compiled_build(build_root: str | Path) -> CompiledBuildInspection:
    """Verify one compiled build without importing AG Engine.

    Intro:
        AG validates the versioned JSON artifact, complete file index, file
        sizes, checksums, and cross-file identities before importing generated
        runtime code.

    Examples:
        Inspect a local build:
            ```python
            result = inspect_compiled_build(".agstudio/build/0123456789abcdef01234567")
            ```

        Reject a changed build:
            ```python
            try:
                inspect_compiled_build("tampered-build")
            except CompiledBuildError:
                pass
            ```

    Args:
        build_root: Exact directory containing `manifest.json`.

    Returns:
        CompiledBuildInspection: Verified manifest and Host identity projection.

    Notes:
        The verifier consumes files only. AG Engine is neither imported nor
        required to be installed in the Host environment.
    """

    root = Path(build_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = CompiledBuildManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        resolved_path = root.joinpath(*PurePosixPath(manifest.resolved_definition_path).parts)
        resolved = ResolvedBuildIdentity.model_validate_json(
            resolved_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise CompiledBuildError(f"Invalid compiled build: {root}") from exc
    if manifest.build_id != root.name:
        raise CompiledBuildError("Build directory name does not match manifest build ID.")
    identities = (
        (resolved.source_digest, manifest.source_digest, "source digest"),
        (resolved.catalog_digest, manifest.catalog_digest, "catalog digest"),
        (
            resolved.semantic_event_protocol_version,
            manifest.semantic_event_protocol_version,
            "semantic event protocol",
        ),
        (
            resolved.logical_output_requirements,
            manifest.logical_output_requirements,
            "logical output requirements",
        ),
    )
    mismatched = [label for actual, expected, label in identities if actual != expected]
    if mismatched:
        raise CompiledBuildError(
            "Resolved definition and manifest differ: " + ", ".join(mismatched)
        )
    expected_paths = {item.path for item in manifest.files} | {"manifest.json"}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_paths != expected_paths:
        raise CompiledBuildError("Compiled build files differ from the manifest index.")
    for expected in manifest.files:
        path = root.joinpath(*PurePosixPath(expected.path).parts)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CompiledBuildError(f"Compiled build file is missing: {expected.path}") from exc
        if len(content) != expected.size or sha256(content).hexdigest() != expected.sha256:
            raise CompiledBuildError(f"Compiled build file integrity failed: {expected.path}")
    return CompiledBuildInspection(manifest=manifest, resolved_definition=resolved)


__all__ = [
    "CompiledBuildError",
    "CompiledBuildInspection",
    "CompiledBuildManifest",
    "CompiledFile",
    "ResolvedBuildIdentity",
    "inspect_compiled_build",
]
