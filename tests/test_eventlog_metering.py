# tests/test_metering_eventlog_service.py
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from aethergraph.contracts.services.metering import MeteringStore
from aethergraph.services.metering.eventlog_metering import EventLogMeteringService


class InMemoryMeteringStore(MeteringStore):
    """
    Simple in-memory MeteringStore for tests.

    Implements just the methods that EventLogMeteringService uses:
      - append(event: dict)
      - query(since, until, kinds, limit)
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def query(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        kinds: list[str] | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self.events:
            ts_str = e.get("ts")
            if not ts_str:
                continue
            # All events should be ISO strings from EventLogMeteringService._append
            ts = datetime.fromisoformat(ts_str)

            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            if kinds is not None and e.get("kind") not in kinds:
                continue

            out.append(e)

        if limit is not None:
            out = out[:limit]
        return out


@pytest.mark.asyncio
async def test_overview_and_basic_aggregation():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # 1 LLM call
    await svc.record_llm(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=123,
    )

    # 1 succeeded run
    await svc.record_run(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        graph_id="g1",
        status="succeeded",
        duration_s=1.5,
    )

    # 1 artifact
    await svc.record_artifact(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        graph_id="g1",
        kind="json",
        bytes=100,
        pinned=True,
    )

    # 1 memory event
    await svc.record_event(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        scope_id="scope-1",
        kind="memory.user_msg",
    )

    overview = await svc.get_overview(user_id="u1", org_id="o1", window="9999d")
    print(overview)

    assert overview["llm_calls"] == 1
    assert overview["llm_prompt_tokens"] == 10
    assert overview["llm_completion_tokens"] == 5

    assert overview["runs"] == 1
    assert overview["runs_succeeded"] == 1
    assert overview["runs_failed"] == 0

    assert overview["artifacts"] == 1
    assert overview["artifact_bytes"] == 100

    assert overview["events"] == 1


@pytest.mark.asyncio
async def test_llm_stats_grouped_by_model_and_user_filter():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # u1 calls 2 different models
    await svc.record_llm(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
    )
    await svc.record_llm(
        user_id="u1",
        org_id="o1",
        run_id="r2",
        model="gpt-4o",
        provider="openai",
        prompt_tokens=20,
        completion_tokens=15,
    )

    # u2 call should not appear when filtering by user_id="u1"
    await svc.record_llm(
        user_id="u2",
        org_id="o1",
        run_id="r3",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=30,
        completion_tokens=25,
    )

    stats = await svc.get_llm_stats(user_id="u1", org_id="o1", window="9999d")

    assert set(stats.keys()) == {"gpt-4o-mini", "gpt-4o"}

    mini = stats["gpt-4o-mini"]
    assert mini["calls"] == 1
    assert mini["prompt_tokens"] == 10
    assert mini["completion_tokens"] == 5

    full = stats["gpt-4o"]
    assert full["calls"] == 1
    assert full["prompt_tokens"] == 20
    assert full["completion_tokens"] == 15


@pytest.mark.asyncio
async def test_graph_stats_aggregate_runs_per_graph():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # Graph g1: 2 runs (1 succeeded, 1 failed)
    await svc.record_run(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        graph_id="g1",
        status="succeeded",
        duration_s=1.0,
    )
    await svc.record_run(
        user_id="u1",
        org_id="o1",
        run_id="r2",
        graph_id="g1",
        status="failed",
        duration_s=2.5,
    )

    # Graph g2: 1 succeeded run
    await svc.record_run(
        user_id="u1",
        org_id="o1",
        run_id="r3",
        graph_id="g2",
        status="succeeded",
        duration_s=3.0,
    )

    stats = await svc.get_graph_stats(user_id="u1", org_id="o1", window="9999d")

    assert set(stats.keys()) == {"g1", "g2"}

    g1 = stats["g1"]
    assert g1["runs"] == 2
    assert g1["succeeded"] == 1
    assert g1["failed"] == 1
    # 1.0 + 2.5 = 3.5
    assert g1["total_duration_s"] == pytest.approx(3.5)

    g2 = stats["g2"]
    assert g2["runs"] == 1
    assert g2["succeeded"] == 1
    assert g2["failed"] == 0
    assert g2["total_duration_s"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_artifact_stats_group_by_kind_and_pinned():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # json artifacts
    await svc.record_artifact(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        graph_id="g1",
        kind="json",
        bytes=100,
        pinned=True,
    )
    await svc.record_artifact(
        user_id="u1",
        org_id="o1",
        run_id="r2",
        graph_id="g1",
        kind="json",
        bytes=50,
        pinned=False,
    )

    # image artifact
    await svc.record_artifact(
        user_id="u1",
        org_id="o1",
        run_id="r3",
        graph_id="g2",
        kind="image",
        bytes=200,
        pinned=True,
    )

    stats = await svc.get_artifact_stats(user_id="u1", org_id="o1", window="9999d")

    assert set(stats.keys()) == {"json", "image"}

    json_stats = stats["json"]
    assert json_stats["count"] == 2
    # 100 + 50
    assert json_stats["bytes"] == 150
    # only first pinned
    assert json_stats["pinned_count"] == 1
    assert json_stats["pinned_bytes"] == 100

    img_stats = stats["image"]
    assert img_stats["count"] == 1
    assert img_stats["bytes"] == 200
    assert img_stats["pinned_count"] == 1
    assert img_stats["pinned_bytes"] == 200


@pytest.mark.asyncio
async def test_memory_stats_filters_by_scope_id_and_prefix():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # Two scopes
    await svc.record_event(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        scope_id="scope-1",
        kind="memory.user_msg",
    )
    await svc.record_event(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        scope_id="scope-1",
        kind="memory.tool_result",
    )
    await svc.record_event(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        scope_id="scope-2",
        kind="memory.user_msg",
    )

    # Non-memory event should be ignored
    await svc.record_event(
        user_id="u1",
        org_id="o1",
        run_id="r1",
        scope_id="scope-1",
        kind="other_kind",
    )

    # All scopes
    all_stats = await svc.get_memory_stats(user_id="u1", org_id="o1", window="9999d")
    assert all_stats["memory.user_msg"]["count"] == 2
    assert all_stats["memory.tool_result"]["count"] == 1

    # Scope-specific
    s1_stats = await svc.get_memory_stats(
        user_id="u1",
        org_id="o1",
        scope_id="scope-1",
        window="9999d",
    )
    assert s1_stats["memory.user_msg"]["count"] == 1
    assert s1_stats["memory.tool_result"]["count"] == 1

    s2_stats = await svc.get_memory_stats(
        user_id="u1",
        org_id="o1",
        scope_id="scope-2",
        window="9999d",
    )
    assert s2_stats["memory.user_msg"]["count"] == 1
    assert "memory.tool_result" not in s2_stats


@pytest.mark.asyncio
async def test_window_filtering_excludes_old_events():
    store = InMemoryMeteringStore()
    svc = EventLogMeteringService(store)

    # Manually append an "old" LLM event (48h ago)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await store.append(
        {
            "kind": "meter.llm",
            "user_id": "u1",
            "org_id": "o1",
            "run_id": "old-run",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "prompt_tokens": 999,
            "completion_tokens": 999,
            "ts": old_ts,
        }
    )

    # Recent event via service (ts set to "now")
    await svc.record_llm(
        user_id="u1",
        org_id="o1",
        run_id="recent-run",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
    )

    # With a small window, we should only see the recent event
    recent_stats = await svc.get_llm_stats(user_id="u1", org_id="o1", window="1h")
    assert recent_stats["gpt-4o-mini"]["calls"] == 1
    assert recent_stats["gpt-4o-mini"]["prompt_tokens"] == 10

    # With a large window, both events show up
    all_stats = await svc.get_llm_stats(user_id="u1", org_id="o1", window="9999d")
    assert all_stats["gpt-4o-mini"]["calls"] == 2
    assert all_stats["gpt-4o-mini"]["prompt_tokens"] == 10 + 999
