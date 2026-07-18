from __future__ import annotations

from pathlib import Path

import pytest

from aethergraph.services.observability import (
    ActiveObservabilityScopeError,
    LLMObservationRecord,
    ObservabilityFacade,
    ObservabilityIdentity,
    ObservabilityNotFoundError,
    ObservationFilter,
    ObservationPolicy,
    ObservationRecord,
    ObservationScope,
    SQLiteObservationStore,
)
from aethergraph.services.observability.retention import (
    RetentionJanitor,
    RetentionPolicy,
)


class _RunStore:
    def __init__(self, runs: list[dict]) -> None:
        self.runs = runs

    async def get(self, run_id: str):
        return next((run for run in self.runs if run["run_id"] == run_id), None)

    async def list(self, *, session_id: str | None = None, **_: object):
        return [run for run in self.runs if session_id is None or run["session_id"] == session_id]


class _EventLog:
    async def query(self, **_: object) -> list[dict]:
        return []


def _llm_record(index: int) -> LLMObservationRecord:
    record = LLMObservationRecord.new(
        call_type="chat",
        provider="openai",
        model="gpt-test",
        dimensions={
            "session_id": "session-long",
            "run_id": f"run-{index}",
            "trace_id": f"run-{index}",
        },
        messages=[
            {"role": "system", "content": "stable agent instructions"},
            {"role": "user", "content": f"new ReAct delta {index}"},
        ],
        reasoning_effort=None,
        max_output_tokens=64,
        output_format="text",
        json_schema=None,
        schema_name=None,
        strict_schema=None,
        validate_json=None,
        extra_params={},
        request_args={"model": "gpt-test"},
        provider_request_args={"temperature": 0},
        compatibility_notes=[],
        trace_payload=None,
    )
    record.raw_text = "stable answer"
    return record


@pytest.mark.asyncio
async def test_long_lived_manifest_growth_stores_only_stable_fragments_and_deltas(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="manifest"),
    )

    await store.append_llm_call(_llm_record(0))
    first = await store.get_storage_stats()
    for index in range(1, 40):
        await store.append_llm_call(_llm_record(index))
    final = await store.get_storage_stats()

    assert final.observations == 40
    assert final.manifests == 40
    assert final.fragments == first.fragments + 39
    assert final.fragment_bytes < 4_096
    with store._connect() as conn:
        stable_fragments = conn.execute(
            "SELECT COUNT(*) FROM content_fragments WHERE body LIKE '%stable agent instructions%'"
        ).fetchone()[0]
    assert stable_fragments == 1
    preview = await store.purge_observations(
        ObservationFilter(session_id="session-long"),
        dry_run=True,
    )
    await store.purge_observations(
        ObservationFilter(session_id="session-long"),
        dry_run=False,
    )
    after = await store.get_storage_stats()
    assert final.logical_bytes - after.logical_bytes == preview.estimated_reclaimed_bytes
    compacted = await store.compact_storage()
    assert compacted.logical_bytes == after.logical_bytes
    assert compacted.database_bytes < final.database_bytes
    assert compacted.physical_bytes < final.physical_bytes


@pytest.mark.asyncio
async def test_resource_revisions_are_queryable_and_purge_accounting_matches_logical_bytes(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    for revision in ("1", "2"):
        await store.append_observation(
            ObservationRecord(
                category="tool",
                name="artifact.update",
                summary=f"revision {revision}",
                scope=ObservationScope(run_id="run-resource"),
                attributes={"payload": "x" * 200},
            ),
            resource_links=[
                {
                    "resource_key": "artifact:report",
                    "relation": "updated",
                    "revision": revision,
                }
            ],
        )

    linked = await store.list_resource_observations("artifact:report", relation="updated")
    before = await store.get_storage_stats()
    preview = await store.purge_observations(
        ObservationFilter(run_id="run-resource"),
        dry_run=True,
    )
    deleted = await store.purge_observations(
        ObservationFilter(run_id="run-resource"),
        dry_run=False,
    )
    after = await store.get_storage_stats()

    assert {item["resource_links"][0]["revision"] for item in linked} == {"1", "2"}
    assert deleted.deleted_observations == 2
    assert before.logical_bytes - after.logical_bytes == preview.estimated_reclaimed_bytes


@pytest.mark.asyncio
async def test_retention_enforces_logical_byte_ceiling_and_purge_rolls_back_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    for index in range(12):
        await store.append_observation(
            ObservationRecord(
                category="log",
                name="cycle",
                summary=f"cycle {index}",
                scope=ObservationScope(run_id=f"run-{index}", trace_id=f"run-{index}"),
                attributes={"payload": "x" * 500},
            )
        )
    initial = await store.get_storage_stats()
    ceiling = initial.logical_bytes // 2
    janitor = RetentionJanitor(
        store,
        RetentionPolicy(
            max_age_days=30,
            error_max_age_days=90,
            max_full_prompt_age_days=3,
            max_bytes_per_trace=10**9,
            max_total_bytes=ceiling,
            max_observations_per_purge=100,
        ),
    )

    await janitor.run_once()
    retained = await store.get_storage_stats()

    assert retained.logical_bytes <= ceiling
    remaining = await store.list_observations()
    assert remaining
    target_id = remaining[0]["observation_id"]

    def fail_gc(*_: object) -> int:
        raise RuntimeError("simulated GC crash")

    monkeypatch.setattr(store, "_garbage_collect_fragments", fail_gc)
    with pytest.raises(RuntimeError, match="simulated GC crash"):
        await store.delete_observation(target_id)
    assert await store.get_observation(target_id) is not None


@pytest.mark.asyncio
async def test_retention_evicts_only_the_trace_above_its_logical_byte_ceiling(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    for trace_id, payload_size in (("trace-small", 32), ("trace-large", 4_096)):
        await store.append_observation(
            ObservationRecord(
                category="log",
                name="sized",
                summary=trace_id,
                scope=ObservationScope(run_id=trace_id, trace_id=trace_id),
                attributes={"payload": "x" * payload_size},
            )
        )
    scopes = {
        row["scope_id"]: row["logical_bytes"] for row in await store.list_scope_storage("trace_id")
    }
    ceiling = (scopes["trace-small"] + scopes["trace-large"]) // 2

    janitor = RetentionJanitor(
        store,
        RetentionPolicy(
            max_age_days=30,
            error_max_age_days=90,
            max_full_prompt_age_days=3,
            max_bytes_per_trace=ceiling,
            max_total_bytes=10**9,
        ),
    )
    await janitor.run_once()

    retained = await store.list_observations()
    suppressed = await store.list_suppressed_scopes()
    assert [row["trace_id"] for row in retained] == ["trace-small"]
    assert suppressed["trace_id"] == {"trace-large"}


@pytest.mark.asyncio
async def test_active_session_deletion_is_atomic_and_completed_session_is_hidden(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    runs = [
        {
            "run_id": "run-active",
            "session_id": "session-active",
            "graph_id": "graph-1",
            "status": "waiting",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    facade = ObservabilityFacade(
        store,
        event_log=_EventLog(),
        run_store=_RunStore(runs),
    )
    await store.append_observation(
        ObservationRecord(
            category="log",
            name="active",
            summary="active",
            scope=ObservationScope(
                session_id="session-active",
                run_id="run-active",
                trace_id="run-active",
            ),
        )
    )

    preview = await facade.delete_session_observations("session-active", dry_run=True)
    with pytest.raises(ActiveObservabilityScopeError):
        await facade.delete_sessions_observations(["session-active"])

    assert preview.matching_observations == 1
    assert (await store.get_storage_stats()).observations == 1
    runs[0]["status"] = "succeeded"

    results = await facade.delete_sessions_observations(["session-active"])

    assert results[0].deleted_observations == 1
    assert (await facade.list_trace_sessions())["items"] == []
    assert await facade.inspect_trace("run-active") is None


@pytest.mark.asyncio
async def test_trace_session_listing_and_deletion_enforce_request_identity(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    runs = [
        {
            "run_id": "run-owned",
            "session_id": "session-owned",
            "graph_id": "graph-1",
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:00+00:00",
            "user_id": "user-1",
            "org_id": "org-1",
        },
        {
            "run_id": "run-other",
            "session_id": "session-other",
            "graph_id": "graph-1",
            "status": "succeeded",
            "started_at": "2026-01-02T00:00:00+00:00",
            "user_id": "user-2",
            "org_id": "org-1",
        },
    ]
    facade = ObservabilityFacade(
        store,
        event_log=_EventLog(),
        run_store=_RunStore(runs),
        identity=ObservabilityIdentity(
            mode="cloud",
            user_id="user-1",
            org_id="org-1",
        ),
    )

    page = await facade.list_trace_sessions()

    assert [item["session_id"] for item in page["items"]] == ["session-owned"]
    with pytest.raises(ObservabilityNotFoundError):
        await facade.delete_session_observations("session-other")
