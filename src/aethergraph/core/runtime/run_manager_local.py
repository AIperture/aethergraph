from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.services.registry.unified_registry import UnifiedRegistry


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _is_task_graph(obj: Any) -> bool:
    # Replace with proper isinstance check in your codebase
    return hasattr(obj, "spec") and hasattr(obj, "io_signature")


def _is_graphfn(obj: Any) -> bool:
    from aethergraph.core.graph.graph_fn import GraphFunction  # adjust path

    return isinstance(obj, GraphFunction)


class RunManager:
    """
    Core coordinator for runs:

    - Resolves targets from the UnifiedRegistry.
    - Calls run_or_resume_async for TaskGraph/GraphFunction.
    - Records metadata in RunStore.
    - TODO: (Later) can coordinate cancellation via sched_registry or best effort with graph_fn.
    """

    def __init__(
        self, *, run_store: RunStore | None = None, registry: UnifiedRegistry | None = None
    ):
        self._store = run_store
        self._registry = registry

    def registry(self) -> UnifiedRegistry:
        return self._registry or current_registry()

    async def _resolve_target(self, graph_id: str) -> Any:
        reg = self.registry()
        # Try static TaskGraph
        try:
            return reg.get_graph(name=graph_id, version=None)
        except KeyError:
            pass
        # Try GraphFunction
        try:
            return reg.get_graphfn(name=graph_id, version=None)
        except KeyError:
            pass
        raise KeyError(f"Graph '{graph_id}' not found")

    async def start_run(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        run_id: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        The main entrypoint for the API layer.

        Returns:
          (record, outputs, has_waits, continuations)
        """
        from aethergraph.core.runtime.graph_runner import run_or_resume_async

        tags = tags or []
        target = await self._resolve_target(graph_id)
        rid = run_id or f"run-{uuid4().hex[:8]}"

        started_at = _utcnow()

        if _is_task_graph(target):
            kind = "taskgraph"
        elif _is_graphfn(target):
            kind = "graphfn"
        else:
            kind = "other"

        # Initial record
        record = RunRecord(
            run_id=rid,
            graph_id=graph_id,
            kind=kind,
            status=RunStatus.running,  # or pending, but we jump straight to running
            started_at=started_at,
            tags=list(tags),
            user_id=user_id,
            org_id=org_id,
        )

        if self._store is not None:
            await self._store.create(record)

        outputs: dict[str, Any] | None = None
        has_waits = False
        continuations: list[dict[str, Any]] = []
        error_msg: str | None = None

        try:
            result = await run_or_resume_async(target, inputs or {}, run_id=rid)
            # If we get here without GraphHasPendingWaits, run is completed
            outputs = result if isinstance(result, dict) else {"result": result}
            record.status = RunStatus.succeeded
            record.finished_at = _utcnow()

        except GraphHasPendingWaits as e:
            # Graph quiesced with pending waits
            record.status = RunStatus.running
            has_waits = True
            continuations = getattr(e, "continuations", [])
            # outputs stay None

        except Exception as exc:
            record.status = RunStatus.failed
            record.finished_at = _utcnow()
            error_msg = str(exc)
            record.error = error_msg
            # TODO: log here with current_logger_factory if desired
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Run %s failed with exception with %s", rid, error_msg
            )

        if self._store is not None:
            await self._store.update_status(
                rid,
                record.status,
                finished_at=record.finished_at,
                error=error_msg,
            )

        meter = current_metering()
        # Duration: if finished_at is None, use now()
        finished_at = record.finished_at or _utcnow()
        duration_s = (finished_at - started_at).total_seconds()
        # Map Runstatus + waits to status string for metering
        if has_waits:
            meter_status = "waiting"
        else:
            status_str = getattr(record.status, "value", str(record.status))
            meter_status = status_str

        try:
            await meter.record_run(
                user_id=user_id,
                org_id=org_id,
                run_id=rid,
                graph_id=graph_id,
                status=meter_status,
                duration_s=duration_s,
            )
        except Exception:
            # Never fail the run due to metering issues
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Error recording run metering for run_id=%s", rid
            )

        return record, outputs, has_waits, continuations

    async def get_record(self, run_id: str) -> RunRecord | None:
        if self._store is None:
            return None
        return await self._store.get(run_id)

    async def list_records(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        if self._store is None:
            return []
        return await self._store.list(graph_id=graph_id, status=status, limit=limit)

    # Placeholder for future cancellation
    async def cancel_run(self, run_id: str) -> RunRecord | None:
        """
        Later: use container.sched_registry to find scheduler and request cancellation.

        For now, it's a stub that just reads the current record.
        """
        # Future:
        #  - container = current_services()
        #  - sched = container.sched_registry.get(run_id)
        #  - if sched: sched.request_cancel() or sched.terminate()
        return await self.get_record(run_id)
