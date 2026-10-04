"""Exact Event paths avoid loading unrelated payloads into Python."""

from dataclasses import replace
import json

import pytest
from test_local_event_store import _database, _event

from aethergraph.storage.contracts import StorageIntegrityError, StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import LocalEventStore


@pytest.mark.asyncio
async def test_exact_event_selected_path_and_array_identity_are_bounded(tmp_path, monkeypatch):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalEventStore(database=database, stream="memory")
    scope = StorageScope(project_id="project", run_id="run")
    selected = {"step_id": "step-2", "result": "λ result" * 3000}
    await store.append(
        _event(
            "event-1",
            scope,
            payload={
                "data": {
                    "private": "secret" * 100000,
                    "steps": [{"step_id": "step-1", "result": "private" * 100000}, selected],
                    "metadata": {"count": 2},
                }
            },
        )
    )
    import aethergraph.storage.providers.local_sqlite.event_store as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("Selected chunks cannot hydrate whole Event records")

    monkeypatch.setattr(owner, "_record", forbidden)
    try:
        pieces, offset = [], 0
        while True:
            chunk = await store.read_payload_chunk(
                scope,
                "event-1",
                json_path="$.data.steps",
                match_key="step_id",
                match_value="step-2",
                offset=offset,
                limit=111,
            )
            assert len(chunk["text"]) <= 111
            assert "private" not in str(chunk)
            pieces.append(chunk["text"])
            if not chunk["has_more"]:
                break
            offset = chunk["next_offset"]
        assert json.loads("".join(pieces)) == selected
        metadata = await store.read_payload_chunk(scope, "event-1", json_path="$.data.metadata")
        assert json.loads(metadata["text"]) == {"count": 2}
        assert not (
            await store.read_payload_chunk(
                scope,
                "event-1",
                json_path="$.data.steps",
                match_key="step_id",
                match_value="absent",
            )
        )["available"]
        assert (
            await store.read_payload_chunk(
                replace(scope, project_id="foreign"), "event-1", json_path="$"
            )
            is None
        )
        with pytest.raises(ValueError, match="offset"):
            await store.read_payload_chunk(
                scope, "event-1", json_path="$.data.metadata", offset=999
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_exact_array_identity_rejects_duplicate_and_ignores_scalar_items(tmp_path):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalEventStore(database=database, stream="memory")
    scope = StorageScope(project_id="project")
    await store.append(
        _event(
            "event-1", scope, payload={"items": ["not-json", None, {"id": "same"}, {"id": "same"}]}
        )
    )
    try:
        with pytest.raises(StorageIntegrityError, match="not unique"):
            await store.read_payload_chunk(
                scope, "event-1", json_path="$.items", match_key="id", match_value="same"
            )
    finally:
        await database.close()
