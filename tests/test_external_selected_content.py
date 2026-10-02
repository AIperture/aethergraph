"""Selected-content contracts qualify the filesystem-free external provider."""

from dataclasses import replace
import json

import pytest
from storage_conformance.external_provider import DeterministicExternalBundle
from test_local_event_store import _event
from test_local_observation_repository import NOW, SCOPE, _llm_call

from aethergraph.storage.contracts import (
    LLMCallLifecycleStatus,
    LLMCallQuery,
    ObservationCaptureMode,
    ObservationScopeManagementRecord,
    ObservationStatus,
    StorageIntegrityError,
    StorageOpenMode,
)


class Clock:
    def now(self):
        return NOW


@pytest.mark.asyncio
async def test_external_visibility_revision_changes_only_for_scoped_owner():
    bundle = DeterministicExternalBundle(
        StorageOpenMode.READ_WRITE, clock=Clock(), ready=True, close_failures=0
    )
    repository = bundle.observations
    before = await repository.scope_management_revision(SCOPE)
    record = ObservationScopeManagementRecord(
        scope_key="trace:1", scope=SCOPE, revision=1, updated_at=NOW, hidden=True
    )
    await repository.compare_and_set_scope_management(record, 0)
    first = await repository.scope_management_revision(SCOPE)
    assert first != before
    await repository.compare_and_set_scope_management(
        replace(record, scope=replace(SCOPE, project_id="foreign")), 0
    )
    assert await repository.scope_management_revision(SCOPE) == first
    await repository.compare_and_set_scope_management(replace(record, revision=2, hidden=False), 1)
    assert await repository.scope_management_revision(SCOPE) != first


@pytest.mark.asyncio
async def test_external_selected_event_scope_bounds_and_unique_identity():
    bundle = DeterministicExternalBundle(
        StorageOpenMode.READ_WRITE, clock=Clock(), ready=True, close_failures=0
    )
    selected = {"id": "selected", "text": "λ" * 100}
    await bundle.events.append(
        _event(
            "event-1",
            SCOPE,
            payload={"items": [selected, {"id": "private", "text": "private" * 1000}]},
        )
    )
    pieces = []
    offset = 0
    while True:
        result = await bundle.events.read_payload_chunk(
            SCOPE,
            "event-1",
            json_path="$.items",
            match_key="id",
            match_value="selected",
            offset=offset,
            limit=11,
        )
        assert len(result["text"]) <= 11
        assert "private" not in str(result)
        pieces.append(result["text"])
        if not result["has_more"]:
            break
        offset = result["next_offset"]
    assert json.loads("".join(pieces)) == selected
    assert (
        await bundle.events.read_payload_chunk(
            replace(SCOPE, run_id="foreign"), "event-1", json_path="$"
        )
        is None
    )
    with pytest.raises(ValueError, match="offset"):
        await bundle.events.read_payload_chunk(SCOPE, "event-1", json_path="$.items", offset=999999)
    await bundle.events.append(_event("event-2", SCOPE, payload={"items": [selected, selected]}))
    with pytest.raises(StorageIntegrityError, match="not unique"):
        await bundle.events.read_payload_chunk(
            SCOPE, "event-2", json_path="$.items", match_key="id", match_value="selected"
        )


@pytest.mark.asyncio
async def test_external_selected_call_and_compact_projection():
    bundle = DeterministicExternalBundle(
        StorageOpenMode.READ_WRITE, clock=Clock(), ready=True, close_failures=0
    )
    repository = bundle.observations
    for mode in (ObservationCaptureMode.FULL, ObservationCaptureMode.METADATA):
        call = _llm_call(
            mode.value,
            capture_mode=mode,
            captured_request={
                "messages": [{"content": "private" * 1000}, {"content": "selected λ"}]
            },
            captured_response={"text": "response"},
        )
        await repository.begin_llm_call(
            replace(
                call,
                lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
                observation=replace(call.observation, status=ObservationStatus.PENDING),
            )
        )
        await repository.finish_llm_call(call.llm_call_id, call)
    page = await repository.query_llm_calls(
        LLMCallQuery(
            scope=SCOPE,
            llm_call_ids=("full",),
            call_names=("answer",),
            include_payload_metadata=False,
        )
    )
    assert [item.llm_call_id for item in page.items] == ["full"]
    assert not page.items[0].request_options
    assert not page.items[0].observation.attributes
    assert not (
        await repository.query_llm_calls(LLMCallQuery(scope=SCOPE, call_names=("foreign",)))
    ).items
    selected = await repository.read_llm_content_chunk(
        SCOPE, "full", section="request", entry_index=1
    )
    assert json.loads(selected["text"]) == {"content": "selected λ"}
    assert "private" not in str(selected)
    assert (await repository.read_llm_content_chunk(SCOPE, "metadata", section="response"))[
        "unavailable_reason"
    ] == "capture_policy"
    assert not (
        await repository.read_llm_content_chunk(SCOPE, "full", section="request", entry_index=10)
    )["available"]
    assert (
        await repository.read_llm_content_chunk(
            replace(SCOPE, run_id="foreign"), "full", section="response"
        )
        is None
    )
