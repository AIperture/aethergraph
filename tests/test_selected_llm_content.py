"""Exact selected captures are bounded before unrelated body hydration."""

from dataclasses import replace
import json

import pytest
from test_local_observation_repository import SCOPE, _database, _llm_call

from aethergraph.storage.contracts import (
    LLMCallLifecycleStatus,
    ObservationCaptureMode,
    ObservationStatus,
    StorageOpenMode,
)
from aethergraph.storage.providers.local_sqlite import LocalObservationRepository


@pytest.mark.asyncio
async def test_selected_response_and_message_chunks_do_not_hydrate_full_call(tmp_path, monkeypatch):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    completed = _llm_call(
        "call-1",
        capture_mode=ObservationCaptureMode.FULL,
        captured_request={"messages": [{"content": "private" * 100000}, {"content": "selected"}]},
        captured_response={"text": "response λ" * 1000},
    )
    completed = replace(
        completed,
        observation=replace(completed.observation, attributes={"large": "private" * 100000}),
    )
    await repository.begin_llm_call(
        replace(
            completed,
            lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
            observation=replace(completed.observation, status=ObservationStatus.PENDING),
        )
    )
    await repository.finish_llm_call("call-1", completed)
    import aethergraph.storage.providers.local_sqlite.observation_repository as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("Selected content cannot load rich call metadata or a full fragment")

    monkeypatch.setattr(owner, "_load_llm_record", forbidden)
    monkeypatch.setattr(owner, "_read_fragment", forbidden)
    try:
        from aethergraph.storage.contracts import LLMCallQuery

        compact = await repository.query_llm_calls(
            LLMCallQuery(scope=SCOPE, include_payload_metadata=False)
        )
        assert not compact.items[0].observation.attributes
        assert not compact.items[0].observation.resource_links
        assert (
            await repository.query_llm_calls(
                LLMCallQuery(scope=SCOPE, call_names=("answer",), include_payload_metadata=False)
            )
        ).items
        assert not (
            await repository.query_llm_calls(
                LLMCallQuery(scope=SCOPE, call_names=("other",), include_payload_metadata=False)
            )
        ).items
        pieces = []
        offset = 0
        while True:
            chunk = await repository.read_llm_content_chunk(
                SCOPE, "call-1", section="response", offset=offset, limit=101
            )
            assert len(chunk["text"]) <= 101
            assert "private" not in str(chunk)
            pieces.append(chunk["text"])
            if not chunk["has_more"]:
                break
            offset = chunk["next_offset"]
        assert json.loads("".join(pieces)) == completed.captured_response
        message = await repository.read_llm_content_chunk(
            SCOPE, "call-1", section="request", entry_index=1
        )
        assert json.loads(message["text"]) == {"content": "selected"}
        assert message["char_count"] < 100
        assert not (
            await repository.read_llm_content_chunk(
                SCOPE, "call-1", section="request", entry_index=100
            )
        )["available"]
        assert (
            await repository.read_llm_content_chunk(
                replace(SCOPE, run_id="foreign"), "call-1", section="response"
            )
            is None
        )
        with pytest.raises(ValueError, match="offset"):
            await repository.read_llm_content_chunk(
                SCOPE, "call-1", section="response", offset=1000000
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_capture_policy_and_explicit_transport_attempt(tmp_path):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    from aethergraph.storage.contracts import LLMCallAttempt

    completed = replace(
        _llm_call("call-1", capture_mode=ObservationCaptureMode.METADATA),
        attempts=(
            LLMCallAttempt(
                attempt_number=1, elapsed_ms=5, outcome="failed", retryable=True, error_code="429"
            ),
        ),
    )
    await repository.begin_llm_call(
        replace(
            completed,
            lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
            observation=replace(completed.observation, status=ObservationStatus.PENDING),
        )
    )
    await repository.finish_llm_call("call-1", completed)
    try:
        response = await repository.read_llm_content_chunk(SCOPE, "call-1", section="response")
        assert response["unavailable_reason"] == "capture_policy"
        attempt = await repository.read_llm_content_chunk(
            SCOPE, "call-1", section="attempt", entry_index=1
        )
        assert json.loads(attempt["text"])["error_code"] == "429"
        assert not (
            await repository.read_llm_content_chunk(
                SCOPE, "call-1", section="attempt", entry_index=2
            )
        )["available"]
    finally:
        await database.close()
