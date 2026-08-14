from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.api.v1.triggers import router
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.scope.scope import Scope
from aethergraph.services.triggers.engine import TriggerEngine
from aethergraph.services.triggers.scheduling import (
    _advance_after_claim,
    _initial_fire_at,
    _next_recurrence,
    _validate_trigger_config,
)
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.triggers.trigger_service import TriggerServiceImpl
from aethergraph.services.triggers.types import TriggerRecord
from aethergraph.storage.triggers.sqlite_trigger_store import SQLiteTriggerStore

UTC = UTC


def _trigger(
    *,
    trigger_id: str = "trig-test",
    kind: str = "one_shot",
    next_fire_at: datetime | None = None,
    interval_seconds: int | None = None,
    cron_expr: str | None = None,
    tz: str | None = None,
    catch_up_missed: bool = False,
    max_overlap_runs: int | None = None,
    org_id: str = "org-a",
    user_id: str = "user-a",
) -> TriggerRecord:
    return TriggerRecord(
        trigger_id=trigger_id,
        graph_id="graph-a",
        kind=kind,  # type: ignore[arg-type]
        next_fire_at=next_fire_at,
        run_at=next_fire_at if kind == "one_shot" else None,
        interval_seconds=interval_seconds,
        cron_expr=cron_expr,
        tz=tz,
        catch_up_missed=catch_up_missed,
        max_overlap_runs=max_overlap_runs,
        org_id=org_id,
        user_id=user_id,
    )


class FakeRunManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def submit_run(self, graph_id: str, **kwargs: Any) -> Any:
        self.calls.append({"graph_id": graph_id, **kwargs})
        return SimpleNamespace(run_id=kwargs.get("run_id") or f"run-{len(self.calls)}")


class FakeRunStore:
    def __init__(self, records: list[Any] | None = None) -> None:
        self.records = records or []
        self.by_id: dict[str, Any] = {}
        self.list_calls: list[tuple[RunStatus | None, int]] = []

    async def get(self, run_id: str) -> Any | None:
        return self.by_id.get(run_id)

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        self.list_calls.append((status, offset))
        matching = [record for record in self.records if status is None or record.status == status]
        return matching[offset : offset + limit]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "interval", "interval_seconds": 0}, "greater than zero"),
        ({"kind": "cron", "cron_expr": "not a cron"}, "Invalid cron"),
        (
            {"kind": "cron", "cron_expr": "0 9 * * *", "tz": "Mars/Olympus"},
            "Unknown trigger timezone",
        ),
    ],
)
def test_trigger_creation_validation(kwargs: dict[str, Any], message: str) -> None:
    base = {
        "kind": "event",
        "cron_expr": None,
        "interval_seconds": None,
        "run_at": None,
        "event_key": "event-a",
        "tz": None,
        "max_overlap_runs": None,
    }
    base.update(kwargs)
    if base["kind"] != "event":
        base["event_key"] = None
    with pytest.raises(ValueError, match=message):
        _validate_trigger_config(**base)


def test_zero_delay_one_shot_is_due() -> None:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    trig = _trigger(next_fire_at=now)
    trig.run_at = now
    assert _initial_fire_at(trig, now) == now


def test_cron_recurrence_preserves_timezone_across_dst() -> None:
    trig = _trigger(
        kind="cron",
        cron_expr="0 9 * * *",
        tz="America/Los_Angeles",
    )
    before_dst = datetime(2026, 3, 7, 17, 0, tzinfo=UTC)
    assert _next_recurrence(trig, before_dst) == datetime(2026, 3, 8, 16, 0, tzinfo=UTC)


def test_non_catch_up_advances_to_first_future_occurrence() -> None:
    scheduled = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    now = scheduled + timedelta(seconds=35)
    trig = _trigger(kind="interval", interval_seconds=10, catch_up_missed=False)
    assert _advance_after_claim(trig, scheduled_for=scheduled, now=now) == scheduled + timedelta(
        seconds=40
    )


def test_catch_up_advances_one_occurrence_at_a_time() -> None:
    scheduled = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    now = scheduled + timedelta(seconds=35)
    trig = _trigger(kind="interval", interval_seconds=10, catch_up_missed=True)
    assert _advance_after_claim(trig, scheduled_for=scheduled, now=now) == scheduled + timedelta(
        seconds=10
    )


async def test_multi_worker_claim_is_atomic(tmp_path: Any) -> None:
    path = tmp_path / "triggers.db"
    store_a = SQLiteTriggerStore(path)
    store_b = SQLiteTriggerStore(path)
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    await store_a.create(_trigger(next_fire_at=now))
    claims_a, claims_b = await asyncio.gather(
        store_a.claim_due(
            now,
            worker_id="worker-a",
            lease_until=now + timedelta(minutes=1),
            limit=10,
        ),
        store_b.claim_due(
            now,
            worker_id="worker-b",
            lease_until=now + timedelta(minutes=1),
            limit=10,
        ),
    )
    assert len(claims_a) + len(claims_b) == 1


async def test_startup_skips_missed_non_catch_up_recurrence(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    scheduled = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    now = scheduled + timedelta(seconds=35)
    trig = _trigger(
        kind="interval",
        next_fire_at=scheduled,
        interval_seconds=10,
        catch_up_missed=False,
    )
    await store.create(trig)
    claims = await store.claim_due(
        now,
        worker_id="worker-a",
        lease_until=now + timedelta(minutes=1),
        limit=10,
        skip_missed_before=now,
    )
    assert claims == []
    stored = await store.get(trig.trigger_id)
    assert stored is not None
    assert stored.next_fire_at == scheduled + timedelta(seconds=40)


async def test_catch_up_claim_remains_due_for_next_missed_occurrence(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    scheduled = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    now = scheduled + timedelta(seconds=35)
    trig = _trigger(
        kind="interval",
        next_fire_at=scheduled,
        interval_seconds=10,
        catch_up_missed=True,
    )
    await store.create(trig)
    claims = await store.claim_due(
        now,
        worker_id="worker-a",
        lease_until=now + timedelta(minutes=1),
        limit=10,
        skip_missed_before=now,
    )
    assert len(claims) == 1
    stored = await store.get(trig.trigger_id)
    assert stored is not None
    assert stored.next_fire_at == scheduled + timedelta(seconds=10)


async def test_startup_applies_one_shot_catch_up_policy(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    scheduled = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    now = scheduled + timedelta(minutes=1)
    skipped = _trigger(trigger_id="trig-skip", next_fire_at=scheduled)
    catch_up = _trigger(trigger_id="trig-catch-up", next_fire_at=scheduled)
    catch_up.catch_up_missed = True
    await store.create(skipped)
    await store.create(catch_up)
    claims = await store.claim_due(
        now,
        worker_id="worker-a",
        lease_until=now + timedelta(minutes=1),
        limit=10,
        skip_missed_before=now,
    )
    assert [claim.trigger.trigger_id for claim in claims] == ["trig-catch-up"]
    stored_skip = await store.get("trig-skip")
    assert stored_skip is not None
    assert not stored_skip.active
    assert stored_skip.next_fire_at is None


async def test_restart_deduplicates_existing_run_after_stale_lease(tmp_path: Any) -> None:
    store_a = SQLiteTriggerStore(tmp_path / "triggers.db")
    store_b = SQLiteTriggerStore(tmp_path / "triggers.db")
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    await store_a.create(_trigger(next_fire_at=now))
    first = await store_a.claim_due(
        now,
        worker_id="worker-a",
        lease_until=now + timedelta(seconds=1),
        limit=1,
    )
    run_id = f"trg-{first[0].fire_id.removeprefix('trigfire-')}"
    run_store = FakeRunStore()
    run_store.by_id[run_id] = SimpleNamespace(run_id=run_id)
    reclaimed = await store_b.claim_due(
        now + timedelta(seconds=2),
        worker_id="worker-b",
        lease_until=now + timedelta(minutes=1),
        limit=1,
    )
    manager = FakeRunManager()
    engine = TriggerEngine(
        store=store_b,
        run_manager=manager,  # type: ignore[arg-type]
        run_store=run_store,  # type: ignore[arg-type]
        worker_id="worker-b",
    )
    await engine._process_claim(reclaimed[0], now + timedelta(seconds=2))
    assert manager.calls == []
    receipt = await store_b.get_claim(reclaimed[0].fire_id)
    assert receipt is not None
    assert receipt["status"] == "delivered"
    assert receipt["run_id"] == run_id


async def test_overlap_limit_uses_greater_equal_and_paginates(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    trig = _trigger(next_fire_at=now, max_overlap_runs=1001)
    await store.create(trig)
    records = [
        SimpleNamespace(status=RunStatus.running, tags=[f"trigger:{trig.trigger_id}"])
        for _ in range(1001)
    ]
    run_store = FakeRunStore(records)
    manager = FakeRunManager()
    engine = TriggerEngine(
        store=store,
        run_manager=manager,  # type: ignore[arg-type]
        run_store=run_store,  # type: ignore[arg-type]
        worker_id="worker-a",
    )
    await engine._process_due_triggers(now)
    assert manager.calls == []
    assert (RunStatus.running, 1000) in run_store.list_calls


async def test_event_fire_is_tenant_scoped(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    trig = _trigger(kind="event")
    trig.event_key = "invoice.paid"
    trig.next_fire_at = None
    await store.create(trig)
    manager = FakeRunManager()
    engine = TriggerEngine(store=store, run_manager=manager)  # type: ignore[arg-type]
    await engine.fire_event("invoice.paid", org_id="org-b", user_id="user-b")
    assert manager.calls == []
    await engine.fire_event(
        "invoice.paid",
        payload={"invoice_id": "inv-1"},
        org_id="org-a",
        user_id="user-a",
    )
    assert manager.calls[0]["inputs"]["event"] == {"invoice_id": "inv-1"}


async def test_event_reads_require_scope_while_due_scan_is_all_tenant(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    await store.create(_trigger(trigger_id="trig-org-a", next_fire_at=now))
    await store.create(
        _trigger(
            trigger_id="trig-org-b",
            next_fire_at=now,
            org_id="org-b",
            user_id="user-b",
        )
    )
    with pytest.raises(ValueError, match="explicit tenant scope"):
        await store.list_by_event_key("event-a")
    claims = await store.claim_due(
        now,
        worker_id="worker-a",
        lease_until=now + timedelta(minutes=1),
        limit=10,
    )
    assert {claim.trigger.org_id for claim in claims} == {"org-a", "org-b"}


async def test_service_get_and_cancel_are_owner_bound(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    trig = _trigger()
    await store.create(trig)
    service = TriggerServiceImpl(store=store)
    assert (
        await service.get(trig.trigger_id, org_id="org-b", user_id="user-b", client_id=None) is None
    )
    assert not await service.cancel(
        trig.trigger_id, org_id="org-b", user_id="user-b", client_id=None
    )
    assert await service.cancel(trig.trigger_id, org_id="org-a", user_id="user-a", client_id=None)


async def test_service_create_list_and_delete_are_owner_bound(tmp_path: Any) -> None:
    store = SQLiteTriggerStore(tmp_path / "triggers.db")
    service = TriggerServiceImpl(store=store)
    scope = Scope(org_id="org-a", user_id="user-a", mode="local")
    trig = await service.create_from_scope(
        scope=scope,
        graph_id="graph-a",
        default_inputs={"message": "hello"},
        kind="interval",
        interval_seconds=60,
    )
    owned = await service.list_for_owner(org_id="org-a", user_id="user-a")
    assert [item.trigger_id for item in owned] == [trig.trigger_id]
    assert await service.list_for_owner(org_id="org-b", user_id="user-b") == []
    assert not await service.delete(
        trig.trigger_id, org_id="org-b", user_id="user-b", client_id=None
    )
    assert await service.delete(trig.trigger_id, org_id="org-a", user_id="user-a", client_id=None)
    assert await store.get(trig.trigger_id) is None


async def test_facade_forwards_scope_on_get_and_cancel() -> None:
    class Service:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def get(self, trigger_id: str, **scope: Any) -> None:
            self.calls.append(("get", scope))

        async def cancel(self, trigger_id: str, **scope: Any) -> bool:
            self.calls.append(("cancel", scope))
            return True

    service = Service()
    facade = TriggerFacade(
        trigger_service=service,  # type: ignore[arg-type]
        trigger_engine=SimpleNamespace(),  # type: ignore[arg-type]
        scope=Scope(org_id="org-a", user_id="user-a", client_id="client-a"),
    )
    await facade.get("trig-a")
    await facade.cancel("trig-a")
    assert service.calls == [
        ("get", {"org_id": "org-a", "user_id": "user-a", "client_id": "client-a"}),
        ("cancel", {"org_id": "org-a", "user_id": "user-a", "client_id": "client-a"}),
    ]


def test_global_event_fire_route_is_removed() -> None:
    paths = {route.path for route in router.routes}
    assert "/triggers/fire-event-global" not in paths
