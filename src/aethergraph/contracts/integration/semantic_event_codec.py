"""Version-aware decoding for durable semantic integration events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Literal

from pydantic import ValidationError

from .models import SemanticEvent
from .versions import (
    SEMANTIC_EVENT_CODEC_REVISION,
    SEMANTIC_EVENT_PROTOCOL_VERSION,
    SEMANTIC_EVENT_READ_VERSIONS,
)

SemanticEventDecodeErrorCode = Literal[
    "semantic_event_version_missing",
    "semantic_event_version_unsupported",
    "semantic_event_version_newer_than_reader",
    "semantic_event_migration_failed",
    "semantic_event_invalid",
]

_PROTOCOL_PATTERN = re.compile(r"^aethergraph\.semantic-event/v(?P<version>[1-9][0-9]*)$")
_CURRENT_PROTOCOL_NUMBER = 3


@dataclass(frozen=True, slots=True)
class DecodedSemanticEvent:
    """Describe one serialized semantic Event decoded to the current model."""

    event: SemanticEvent
    source_schema_version: str
    normalized_schema_version: str
    migration_path: tuple[str, ...]
    codec_revision: str


class SemanticEventDecodeError(ValueError):
    """Report a bounded semantic-event version or validation failure."""

    def __init__(
        self,
        *,
        code: SemanticEventDecodeErrorCode,
        message: str,
        source_schema_version: str | None,
        event_id: str | None,
        event_kind: str | None,
        migration_stage: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.source_schema_version = source_schema_version
        self.target_schema_version = SEMANTIC_EVENT_PROTOCOL_VERSION
        self.event_id = event_id
        self.event_kind = event_kind
        self.migration_stage = migration_stage
        self.codec_revision = SEMANTIC_EVENT_CODEC_REVISION

    def to_dict(self) -> dict[str, str | None]:
        """Return the safe structured diagnostic fields for this decode failure.

        The result excludes the serialized Event payload and exception internals so
        consumers can persist or publish it at an application boundary.

        Examples:
            Read the stable failure code:
                ```python
                details = error.to_dict()
                assert details["code"].startswith("semantic_event_")
                ```

            Inspect the attempted target:
                ```python
                assert error.to_dict()["target_schema_version"].endswith("/v3")
                ```

        Args:
            None.

        Returns:
            dict[str, str | None]: Bounded version, identity, stage, and codec fields.

        Notes:
            The human-readable exception message remains available through `str(error)`.
        """

        return {
            "code": self.code,
            "source_schema_version": self.source_schema_version,
            "target_schema_version": self.target_schema_version,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "migration_stage": self.migration_stage,
            "codec_revision": self.codec_revision,
        }


def decode_semantic_event(value: Mapping[str, object]) -> DecodedSemanticEvent:
    """Decode one released semantic-event shape into the current Event model.

    The decoder detaches the supplied mapping, applies the exact registered migration
    chain, and validates the result as the current `SemanticEvent`. It never mutates the
    caller's value and never retries current validation as an older schema.

    Examples:
        Decode a current Event:
            ```python
            decoded = decode_semantic_event(event.model_dump(mode="json"))
            assert decoded.event == event
            assert decoded.migration_path == ()
            ```

        Decode released v2 history:
            ```python
            decoded = decode_semantic_event(v2_event)
            assert decoded.source_schema_version.endswith("/v2")
            assert decoded.normalized_schema_version.endswith("/v3")
            ```

    Args:
        value: Detached or immutable serialized semantic Event mapping.

    Returns:
        DecodedSemanticEvent: Current validated Event and exact migration evidence.

    Notes:
        Versions in `SEMANTIC_EVENT_READ_VERSIONS` are durable read contracts. Active
        writers continue to emit only `SEMANTIC_EVENT_PROTOCOL_VERSION`.
    """

    raw = _thaw_json(value)
    source = raw.get("schema_version")
    event_id = _optional_text(raw.get("event_id"))
    event_kind = _optional_text(raw.get("kind"))
    if not isinstance(source, str) or not source:
        raise SemanticEventDecodeError(
            code="semantic_event_version_missing",
            message="Semantic event schema_version is required.",
            source_schema_version=None,
            event_id=event_id,
            event_kind=event_kind,
            migration_stage="version_classification",
        )
    if source not in SEMANTIC_EVENT_READ_VERSIONS:
        match = _PROTOCOL_PATTERN.fullmatch(source)
        code: SemanticEventDecodeErrorCode = "semantic_event_version_unsupported"
        if match is not None and int(match.group("version")) > _CURRENT_PROTOCOL_NUMBER:
            code = "semantic_event_version_newer_than_reader"
        raise SemanticEventDecodeError(
            code=code,
            message=(
                f"Semantic event version {source!r} cannot be decoded by "
                f"{SEMANTIC_EVENT_CODEC_REVISION}."
            ),
            source_schema_version=source,
            event_id=event_id,
            event_kind=event_kind,
            migration_stage="version_classification",
        )

    current = raw
    migration_path: list[str] = []
    migration_stage = f"{source}->current"
    try:
        if source == "aethergraph.semantic-event/v1":
            migration_stage = "aethergraph.semantic-event/v1->v2"
            current = _migrate_v1_to_v2(current)
            migration_path.append(migration_stage)
        if current.get("schema_version") == "aethergraph.semantic-event/v2":
            migration_stage = "aethergraph.semantic-event/v2->v3"
            current = _migrate_v2_to_v3(current)
            migration_path.append(migration_stage)
    except (TypeError, ValueError) as exc:
        raise SemanticEventDecodeError(
            code="semantic_event_migration_failed",
            message=f"Semantic event {source!r} could not be migrated to the current schema.",
            source_schema_version=source,
            event_id=event_id,
            event_kind=event_kind,
            migration_stage=migration_stage,
        ) from exc

    try:
        event = SemanticEvent.model_validate(current)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SemanticEventDecodeError(
            code="semantic_event_invalid",
            message=(
                f"Semantic event {source!r} is invalid after normalization to "
                f"{SEMANTIC_EVENT_PROTOCOL_VERSION}."
            ),
            source_schema_version=source,
            event_id=event_id,
            event_kind=event_kind,
            migration_stage="current_validation",
        ) from exc
    return DecodedSemanticEvent(
        event=event,
        source_schema_version=source,
        normalized_schema_version=SEMANTIC_EVENT_PROTOCOL_VERSION,
        migration_path=tuple(migration_path),
        codec_revision=SEMANTIC_EVENT_CODEC_REVISION,
    )


def _migrate_v1_to_v2(value: dict[str, Any]) -> dict[str, Any]:
    migrated = _thaw_json(value)
    migrated["schema_version"] = "aethergraph.semantic-event/v2"
    payload = migrated.get("payload")
    kind = migrated.get("kind")
    if not isinstance(payload, dict):
        return migrated
    if kind == "tool.activity" and payload.get("status") == "failed" and not payload.get("error"):
        summary = str(payload.get("message") or "A historical Tool call failed.")[:1_000]
        payload["error"] = {
            "kind": "runtime",
            "code": "legacy_tool_failure",
            "summary": summary,
            "retryable": False,
            "details": {},
            "repair_hints": [],
            "allowed_actions": [],
            "reference": None,
        }
    elif kind == "turn.completed":
        migrated["kind"] = "turn.outcome"
        payload = {
            "outcome": "completed",
            "code": "completed",
            "summary": "Turn completed.",
            "resumable": False,
            "engine_turn_id": str(migrated.get("turn_id") or "legacy_turn"),
            "reply_disposition": (
                "message_required" if payload.get("result_available") else "no_message"
            ),
        }
    elif kind == "turn.failed":
        migrated["kind"] = "turn.outcome"
        payload = {
            "outcome": "failed",
            "code": str(payload.get("code") or "legacy_turn_failed"),
            "summary": str(payload.get("message") or "Turn failed."),
            "resumable": bool(payload.get("retryable")),
            "engine_turn_id": str(migrated.get("turn_id") or "legacy_turn"),
            "reply_disposition": None,
        }
    migrated["payload"] = payload
    return migrated


def _migrate_v2_to_v3(value: dict[str, Any]) -> dict[str, Any]:
    migrated = _thaw_json(value)
    migrated["schema_version"] = SEMANTIC_EVENT_PROTOCOL_VERSION
    payload = migrated.get("payload")
    if not isinstance(payload, dict):
        return migrated
    kind = migrated.get("kind")
    if kind == "message.completed":
        artifact_ids = payload.pop("artifact_ids", [])
        payload.setdefault(
            "attachments",
            [
                {
                    "artifact_id": str(artifact_id),
                    "presentation": "auto",
                    "title": "",
                    "alt_text": "",
                }
                for artifact_id in artifact_ids
            ],
        )
        payload.setdefault("actions", [])
    elif kind == "input.accepted":
        if not {"input_kind", "input_type", "source"}.intersection(payload):
            interaction_id = payload.get("interaction_id")
            payload["input_kind"] = "message"
            payload["input_type"] = "interaction.response" if interaction_id else "user.message"
            payload["source"] = f"legacy:{migrated.get('producer') or 'unknown'}"
    migrated["payload"] = payload
    return migrated


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _optional_text(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


__all__ = [
    "DecodedSemanticEvent",
    "SemanticEventDecodeError",
    "SemanticEventDecodeErrorCode",
    "decode_semantic_event",
]
