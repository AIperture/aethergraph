"""Frozen public Artifact response projection over canonical storage records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.storage.contracts import (
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRetentionRecord,
    StorageIntegrityError,
)

_CONTENT_ROUTE_PREFIX = "/api/v1/artifacts"


def project_public_artifact(
    record: ArtifactRecord,
    *,
    occurrence: ArtifactOccurrence | None = None,
    retention: ArtifactRetentionRecord | None = None,
    deprecated_app_id: str | None = None,
) -> Artifact:
    """Project canonical artifact state into the frozen public Artifact DTO.

    Immutable content, one optional execution occurrence, and independent retention
    intent are combined only at the response boundary. The public URI is the AG
    content endpoint, never the provider blob locator or a local source path.

    Examples:
        Project metadata only:
            ```python
            artifact = project_public_artifact(record)
            ```

        Project one occurrence with deprecated App metadata:
            ```python
            artifact = project_public_artifact(
                record,
                occurrence=occurrence,
                retention=retention,
                deprecated_app_id=request.app_id,
            )
            ```

    Args:
        record: Exact canonical immutable artifact metadata.
        occurrence: Optional exact occurrence selected for this public response.
        retention: Optional current retention state for the same content owner.
        deprecated_app_id: Optional deprecated App compatibility metadata supplied by
            the caller; it is never inferred or used for storage authorization.

    Returns:
        Artifact: Frozen public DTO shape with legacy MIME names serialized only by
        that DTO and a stable AG content URL.

    Notes:
        `app_id` remains response-only deprecated compatibility metadata. `client_id`
        is never reconstructed from canonical scope, and non-SHA-256 content leaves
        the legacy `sha256` field empty rather than relabeling another digest.
    """
    if not isinstance(record, ArtifactRecord):
        raise TypeError("record must be an ArtifactRecord")
    if occurrence is not None:
        _validate_occurrence(record, occurrence)
    if retention is not None:
        _validate_retention(record, retention)
    if deprecated_app_id is not None and (
        not isinstance(deprecated_app_id, str) or not deprecated_app_id.strip()
    ):
        raise ValueError("deprecated_app_id must be a non-empty string when supplied")

    labels = _thaw_mapping(record.labels)
    if occurrence is not None:
        labels.update(_thaw_mapping(occurrence.labels))
    if record.original_filename is not None:
        labels.setdefault("filename", record.original_filename)
    scope = occurrence.scope if occurrence is not None else record.owner_scope
    created_at = occurrence.occurred_at if occurrence is not None else record.created_at
    return Artifact(
        artifact_id=record.artifact_id,
        run_id=scope.run_id,
        graph_id=scope.graph_id,
        node_id=scope.node_id,
        tool_name=occurrence.tool_name if occurrence is not None else None,
        tool_version=occurrence.tool_version if occurrence is not None else None,
        kind=record.kind,
        sha256=record.content_hash if record.hash_algorithm.lower() == "sha256" else None,
        bytes=record.size_bytes,
        mime=record.media_type,
        created_at=created_at.isoformat(),
        tags=_tags(labels),
        labels=labels,
        metrics=dict(occurrence.metrics) if occurrence is not None else {},
        pinned=retention.pinned if retention is not None else False,
        uri=f"{_CONTENT_ROUTE_PREFIX}/{quote(record.artifact_id, safe='')}/content",
        preview_uri=None,
        org_id=scope.org_id,
        user_id=scope.user_id,
        client_id=None,
        app_id=deprecated_app_id,
        session_id=scope.session_id,
        occurrence_id=occurrence.occurrence_id if occurrence is not None else None,
    )


def _validate_occurrence(record: ArtifactRecord, occurrence: ArtifactOccurrence) -> None:
    if occurrence.artifact_id != record.artifact_id:
        raise StorageIntegrityError("Artifact occurrence references different content")
    for name, value in record.owner_scope.as_filter().items():
        if getattr(occurrence.scope, name) != value:
            raise StorageIntegrityError("Artifact occurrence crosses canonical owner scope")


def _validate_retention(record: ArtifactRecord, retention: ArtifactRetentionRecord) -> None:
    if retention.artifact_id != record.artifact_id or retention.scope != record.owner_scope:
        raise StorageIntegrityError("Artifact retention crosses canonical content ownership")


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _thaw_json(item) for key, item in value.items()}


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _thaw_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_thaw_json(item) for item in value]
    return value


def _tags(labels: Mapping[str, Any]) -> list[str] | None:
    value = labels.get("tags")
    if isinstance(value, str):
        tags = [item.strip() for item in value.split(",") if item.strip()]
        return tags or None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        tags = [str(item) for item in value]
        return tags or None
    return None
