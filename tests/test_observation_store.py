from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aethergraph.services.observability import (
    ObservabilityFacade,
    ObservationFilter,
    ObservationRecord,
    ObservationScope,
    SQLiteObservationStore,
)
from aethergraph.services.observability.retention import RetentionJanitor, RetentionPolicy


@pytest.mark.asyncio
async def test_observation_crud_resource_links_and_storage_stats(tmp_path: Path) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    facade = ObservabilityFacade(store)
    record = ObservationRecord(
        category="service",
        name="artifact.read",
        summary="read report",
        scope=ObservationScope(run_id="run-1", trace_id="trace-1"),
    )

    observation_id = await facade.append_observation(
        record,
        resource_links=[
            {
                "resource_key": "artifact:report",
                "relation": "read",
                "revision": "2",
                "artifact_id": "report",
            }
        ],
    )

    detail = await facade.get_observation(observation_id)
    assert detail["summary"] == "read report"
    assert detail["resource_links"][0]["resource_key"] == "artifact:report"
    linked = await store.list_resource_observations("artifact:report", relation="read")
    assert [item["observation_id"] for item in linked] == [observation_id]
    assert await facade.list_traces() == ["trace-1"]
    assert len(await facade.get_trace("trace-1")) == 1
    stats = await facade.get_storage_stats()
    assert stats.observations == 1
    with store._connect() as conn:
        link = conn.execute("SELECT * FROM observation_resource_links").fetchone()
    assert link["resource_key"] == "artifact:report"
    assert link["relation"] == "read"

    await facade.delete_observation(observation_id)
    assert await facade.get_observation(observation_id) is None


@pytest.mark.asyncio
async def test_unknown_resource_relation_fails_without_partial_observation(tmp_path: Path) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    record = ObservationRecord(category="tool", name="result", summary="bad relation")

    with pytest.raises(ValueError, match="Unsupported observation resource relation"):
        await store.append_observation(
            record,
            resource_links=[{"resource_key": "artifact:x", "relation": "produced"}],
        )

    assert (await store.get_storage_stats()).observations == 0


@pytest.mark.asyncio
async def test_purge_is_bounded_supports_dry_run_and_tombstones_deleted_trace(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    for index in range(3):
        await store.append_observation(
            ObservationRecord(
                category="log",
                name="test",
                summary=f"log {index}",
                scope=ObservationScope(run_id="run-1", trace_id="trace-1"),
                occurred_at=f"2026-01-0{index + 1}T00:00:00+00:00",
            )
        )
    await store.update_trace_management(
        "trace:trace-1",
        pinned=False,
        label="temporary",
        scope={"trace_id": "trace-1", "run_id": "run-1"},
    )

    preview = await store.purge_observations(
        ObservationFilter(run_id="run-1", limit=1),
        dry_run=True,
    )
    deleted = await store.purge_observations(
        ObservationFilter(run_id="run-1", limit=1),
        dry_run=False,
    )

    assert preview.matching_observations == 1
    assert deleted.dry_run is False
    assert deleted.deleted_observations == 1
    assert len(await store.list_observations(ObservationFilter(run_id="run-1"))) == 2

    target_preview = await store.purge_observations(
        ObservationFilter(run_id="run-1", target_reclaimed_bytes=1),
        dry_run=True,
    )
    assert target_preview.matching_observations == 1
    assert target_preview.estimated_reclaimed_bytes > 0

    await store.delete_trace("trace-1")
    suppressed = await store.list_suppressed_scopes()
    assert suppressed["trace_id"] == {"trace-1"}


@pytest.mark.asyncio
async def test_retention_janitor_preserves_pinned_scope_and_evicts_old_unpinned(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    old = "2025-01-01T00:00:00+00:00"
    for trace_id in ("pinned", "ordinary"):
        await store.append_observation(
            ObservationRecord(
                category="log",
                name="test",
                summary=trace_id,
                scope=ObservationScope(trace_id=trace_id),
                occurred_at=old,
            )
        )
    await store.update_trace_management(
        "trace:pinned",
        pinned=True,
        scope={"trace_id": "pinned"},
    )
    janitor = RetentionJanitor(
        store,
        RetentionPolicy(
            max_age_days=30,
            max_full_prompt_age_days=3,
            max_total_bytes=10**9,
            max_observations_per_purge=10,
        ),
    )

    await janitor.run_once(now=datetime(2026, 1, 1, tzinfo=UTC))

    rows = await store.list_observations()
    assert [row["trace_id"] for row in rows] == ["pinned"]
