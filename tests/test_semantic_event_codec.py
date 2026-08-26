from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from aethergraph.contracts.integration import (
    SEMANTIC_EVENT_CODEC_REVISION,
    SEMANTIC_EVENT_PROTOCOL_VERSION,
    SEMANTIC_EVENT_READ_VERSIONS,
    SemanticEventDecodeError,
    SemanticEventKind,
    decode_semantic_event,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "integration" / "semantic_event_legacy.json"
_MANIFEST = (
    Path(__file__).parent
    / "fixtures"
    / "integration"
    / "semantic_event_compatibility_manifest.json"
)


def _events() -> list[dict[str, object]]:
    return list(json.loads(_FIXTURE.read_text(encoding="utf-8"))["events"])


def test_compatibility_manifest_pins_writer_codec_and_released_history() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    fixture_versions = {str(event["schema_version"]) for event in _events()}

    assert manifest["writer_version"] == SEMANTIC_EVENT_PROTOCOL_VERSION
    assert manifest["codec_revision"] == SEMANTIC_EVENT_CODEC_REVISION
    assert tuple(manifest["retained_read_versions"]) == SEMANTIC_EVENT_READ_VERSIONS
    assert fixture_versions == set(SEMANTIC_EVENT_READ_VERSIONS[:-1])
    assert manifest["migration_steps"] == [
        f"{source}->{target.rsplit('/', 1)[-1]}"
        for source, target in zip(
            SEMANTIC_EVENT_READ_VERSIONS,
            SEMANTIC_EVENT_READ_VERSIONS[1:],
            strict=False,
        )
    ]
    assert manifest["historical_fixture"] == _FIXTURE.name


def test_codec_decodes_released_history_without_mutating_source() -> None:
    historical = _events()
    original = deepcopy(historical)

    decoded = [decode_semantic_event(item) for item in historical]

    assert historical == original
    assert all(item.event.schema_version == SEMANTIC_EVENT_PROTOCOL_VERSION for item in decoded)
    assert decoded[0].migration_path == (
        "aethergraph.semantic-event/v1->v2",
        "aethergraph.semantic-event/v2->v3",
    )
    message = decoded[0].event.payload.model_dump(mode="json")
    assert message["attachments"] == [
        {
            "artifact_id": "artifact_fixture",
            "presentation": "auto",
            "title": "",
            "alt_text": "",
        }
    ]
    assert message["actions"] == []
    tool = decoded[1].event.payload.model_dump(mode="json")
    assert tool["error"]["code"] == "legacy_tool_failure"
    assert decoded[2].event.kind is SemanticEventKind.TURN_OUTCOME
    assert decoded[2].event.payload.model_dump(mode="json")["outcome"] == "completed"
    accepted = decoded[3].event.payload.model_dump(mode="json")
    assert accepted["source"] == "legacy:agent.fixture"
    assert decoded[3].codec_revision == SEMANTIC_EVENT_CODEC_REVISION


def test_codec_accepts_current_event_without_a_migration() -> None:
    current = decode_semantic_event(_events()[0]).event.model_dump(mode="json")

    decoded = decode_semantic_event(current)

    assert decoded.migration_path == ()
    assert decoded.source_schema_version == SEMANTIC_EVENT_PROTOCOL_VERSION


@pytest.mark.parametrize(
    ("schema_version", "code"),
    [
        (None, "semantic_event_version_missing"),
        ("aethergraph.semantic-event/v0", "semantic_event_version_unsupported"),
        ("aethergraph.semantic-event/v4", "semantic_event_version_newer_than_reader"),
        ("another.semantic-event/v3", "semantic_event_version_unsupported"),
    ],
)
def test_codec_classifies_unsupported_versions(
    schema_version: str | None,
    code: str,
) -> None:
    value = _events()[0]
    if schema_version is None:
        value.pop("schema_version")
    else:
        value["schema_version"] = schema_version

    with pytest.raises(SemanticEventDecodeError) as raised:
        decode_semantic_event(value)

    assert raised.value.code == code
    assert raised.value.to_dict()["codec_revision"] == SEMANTIC_EVENT_CODEC_REVISION


def test_codec_does_not_retry_invalid_current_data_as_legacy() -> None:
    current = decode_semantic_event(_events()[0]).event.model_dump(mode="json")
    current["payload"] = {"message_id": "missing_text"}

    with pytest.raises(SemanticEventDecodeError) as raised:
        decode_semantic_event(current)

    assert raised.value.code == "semantic_event_invalid"
    assert raised.value.migration_stage == "current_validation"
