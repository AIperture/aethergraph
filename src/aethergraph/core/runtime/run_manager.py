from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
from typing import Any
from uuid import uuid4

from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import current_services
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
        self,
        *,
        run_store: RunStore | None = None,
        registry: UnifiedRegistry | None = None,
        sched_registry: Any | None = None,  # placeholder for future use
    ):
        self._store = run_store
        self._registry = registry
        self._sched_registry = sched_registry

    # -------- registry helpers --------

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

    # -------- core execution helper --------

    async def _run_and_finalize(
        self,
        *,
        record: RunRecord,
        target: Any,
        graph_id: str,
        inputs: dict[str, Any],
        user_id: str | None,
        org_id: str | None,
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Shared core logic that actually calls run_or_resume_async, updates
        RunStore, and records metering.

        Returns:
          (record, outputs, has_waits, continuations)
        """
        from aethergraph.core.runtime.graph_runner import run_or_resume_async

        # tags = record.tags or []
        started_at = record.started_at or _utcnow()

        outputs: dict[str, Any] | None = None
        has_waits = False
        continuations: list[dict[str, Any]] = []
        error_msg: str | None = None

        try:
            result = await run_or_resume_async(
                target, inputs or {}, run_id=record.run_id, session_id=record.meta.get("session_id")
            )

            # If we get here without GraphHasPendingWaits, run is completed
            outputs = result if isinstance(result, dict) else {"result": result}
            record.status = RunStatus.succeeded
            record.finished_at = _utcnow()

        except asyncio.CancelledError:
            # Cancellation path: scheduler.terminate() or external cancel.
            import logging

            record.status = RunStatus.canceled
            record.finished_at = _utcnow()
            error_msg = "Run cancelled by user"
            logging.getLogger("aethergraph.runtime.run_manager").info(
                "Run %s was cancelled", record.run_id
            )

        except GraphHasPendingWaits as e:
            # Graph quiesced with pending waits
            record.status = RunStatus.failed  # consider 'waiting' status later
            has_waits = True
            continuations = getattr(e, "continuations", [])
            # outputs remain None

        except Exception as exc:  # noqa: BLE001
            record.status = RunStatus.failed
            record.finished_at = _utcnow()
            error_msg = str(exc)
            record.error = error_msg
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Run %s failed with exception: %s", record.run_id, error_msg
            )

        # Persist status update
        if self._store is not None:
            await self._store.update_status(
                record.run_id,
                record.status,
                finished_at=record.finished_at,
                error=error_msg,
            )

        # Metering
        meter = current_metering()
        finished_at = record.finished_at or _utcnow()
        duration_s = (finished_at - started_at).total_seconds()

        if has_waits:
            meter_status = "waiting"
        else:
            status_str = getattr(record.status, "value", str(record.status))
            meter_status = status_str

        try:
            await meter.record_run(
                user_id=user_id,
                org_id=org_id,
                run_id=record.run_id,
                graph_id=graph_id,
                status=meter_status,
                duration_s=duration_s,
            )
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Error recording run metering for run_id=%s", record.run_id
            )

        return record, outputs, has_waits, continuations

    # -------- new: non-blocking submit_run --------

    async def submit_run(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        run_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> RunRecord:
        """
        Non-blocking entrypoint for the HTTP API.

        - Creates a RunRecord (status=running).
        - Persists it to RunStore.
        - Schedules background execution via asyncio.create_task.
        - Returns immediately with the record (for run_id, status, etc).
        """
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

        # pull flow_id and entrypoint from registry if possible
        flow_id: str | None = None
        reg = self.registry()
        if reg is not None:
            if kind == "taskgraph":
                meta = reg.get_meta(nspace="graph", name=graph_id, version=None) or {}
            elif kind == "graphfn":
                meta = reg.get_meta(nspace="graphfn", name=graph_id, version=None) or {}
            else:
                meta = {}
            flow_id = meta.get("flow_id") or graph_id

        # use run_id as session_id if not provided
        if session_id is None:
            session_id = run_id

        record = RunRecord(
            run_id=rid,
            graph_id=graph_id,
            kind=kind,
            status=RunStatus.running,  # we go straight to running as before
            started_at=started_at,
            tags=list(tags),
            user_id=user_id,
            org_id=org_id,
            meta={},
        )

        if flow_id:
            record.meta["flow_id"] = flow_id
            if f"flow:{flow_id}" not in record.tags:
                record.tags.append(f"flow:{flow_id}")  # add flow tag if missing
        if session_id:
            record.meta["session_id"] = session_id
            if f"session:{session_id}" not in record.tags:
                record.tags.append(f"session:{session_id}")  # add session tag if missing

        if self._store is not None:
            await self._store.create(record)

        async def _bg():
            await self._run_and_finalize(
                record=record,
                target=target,
                graph_id=graph_id,
                inputs=inputs,
                user_id=user_id,
                org_id=org_id,
            )

        # If we're in an event loop (server), schedule in the background.
        # If not (CLI), just run inline so behaviour is still sane.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not inside a running loop – e.g., CLI usage.
            await _bg()
        else:
            loop.create_task(_bg())

        return record

    # -------- old: blocking start_run (CLI/tests) --------

    async def start_run(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        run_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Blocking helper (original behaviour).

        - Resolves target.
        - Creates RunRecord with status=running.
        - Runs once via run_or_resume_async.
        - Updates store + metering.
        - Returns (record, outputs, has_waits, continuations).

        Still useful for tests/CLI, but the HTTP route should prefer submit_run().
        """
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

        # pull flow_id and entrypoint from registry if possible
        flow_id: str | None = None
        reg = self.registry()
        if reg is not None:
            if kind == "taskgraph":
                meta = reg.get_meta(nspace="graph", name=graph_id, version=None) or {}
            elif kind == "graphfn":
                meta = reg.get_meta(nspace="graphfn", name=graph_id, version=None) or {}
            else:
                meta = {}
            flow_id = meta.get("flow_id") or graph_id

        # use run_id as session_id if not provided
        if session_id is None:
            session_id = run_id

        record = RunRecord(
            run_id=rid,
            graph_id=graph_id,
            kind=kind,
            status=RunStatus.running,
            started_at=started_at,
            tags=list(tags),
            user_id=user_id,
            org_id=org_id,
            meta={},
        )

        if flow_id:
            record.meta["flow_id"] = flow_id
            if f"flow:{flow_id}" not in record.tags:
                record.tags.append(f"flow:{flow_id}")  # add flow tag if missing
        if session_id:
            record.meta["session_id"] = session_id
            if f"session:{session_id}" not in record.tags:
                record.tags.append(f"session:{session_id}")  # add session tag if missing

        if self._store is not None:
            await self._store.create(record)

        return await self._run_and_finalize(
            record=record,
            target=target,
            graph_id=graph_id,
            inputs=inputs,
            user_id=user_id,
            org_id=org_id,
        )

    async def get_record(self, run_id: str) -> RunRecord | None:
        if self._store is None:
            return None
        out = await self._store.get(run_id)
        return out

    async def list_records(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        flow_id: str | None = None,  # NEW
        limit: int = 100,
    ) -> list[RunRecord]:
        if self._store is None:
            return []

        # First filter by graph_id/status in the store (TODO: implement self._store.list with flow_id for efficiency)
        records = await self._store.list(graph_id=graph_id, status=status, limit=limit)

        if flow_id is not None:
            records = [r for r in records if r.meta.get("flow_id") == flow_id]

        return records

    def _get_sched_registry(self):
        if self._sched_registry is not None:
            return self._sched_registry
        try:
            container = current_services()
        except Exception:
            return None
        return getattr(container, "sched_registry", None)

    async def cancel_run(self, run_id: str) -> RunRecord | None:
        """
        Best-effort cancellation for a run.

        Behaviour:
        - If the run is found and not yet terminal:
            - Mark status = cancellation_requested and persist.
            - Look up scheduler in sched_registry and call terminate().
        - If the run is already terminal, return it unchanged.
        - If no record is found, we still try scheduler-level termination
          (in case the run hasn't been persisted yet), then return None.

        The actual transition to RunStatus.canceled happens inside
        _run_and_finalize() when the scheduler raises asyncio.CancelledError.
        """
        record: RunRecord | None = None
        print("🍎 Debug: RunManager.cancel_run called for run_id =", run_id)
        if self._store is not None:
            record = await self._store.get(run_id)

        # Helper: scheduler-level termination
        async def _terminate_scheduler() -> None:
            reg = self._get_sched_registry()
            if reg is None:
                return
            sched = reg.get(run_id)
            if sched is None:
                return

            try:
                terminate = getattr(sched, "terminate", None) or getattr(
                    sched, "request_cancel", None
                )
                if terminate is None:
                    return

                if inspect.iscoroutinefunction(terminate):
                    await terminate()
                else:
                    # allow sync terminate() implementations
                    terminate()
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger("aethergraph.runtime.run_manager").exception(
                    "Error terminating scheduler for run_id=%s", run_id
                )

        # No record in store – still try to terminate scheduler, then bail
        if record is None:
            await _terminate_scheduler()
            return None

        # If already terminal, don't change status
        if record.status in {
            RunStatus.succeeded,
            RunStatus.failed,
            RunStatus.canceled,
        }:
            return record

        # Mark cancellation requested so UI can react immediately
        record.status = RunStatus.cancellation_requested
        if self._store is not None:
            await self._store.update_status(
                run_id,
                RunStatus.cancellation_requested,
                finished_at=None,
                error=None,
            )

        # Ask the scheduler to stop
        await _terminate_scheduler()

        return record
