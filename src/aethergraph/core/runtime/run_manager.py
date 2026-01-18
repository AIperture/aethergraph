from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.execution.forward_scheduler import ForwardScheduler
from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunStatus,
    RunVisibility,
    _make_preview,
)
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
    TODO: for global schedulers, we may want to have a dedicated run manager -- current
    implementation utilize the async_run which create a local ForwardScheduler instance
    each graph run. This is fine for concurrent graphs under thousands but may
    not scale well for large number of concurrent graphs.
    """

    def __init__(
        self,
        *,
        run_store: RunStore | None = None,
        registry: UnifiedRegistry | None = None,
        sched_registry: Any | None = None,  # placeholder for future use
        max_concurrent_runs: int | None = None,
    ):
        self._store = run_store
        self._registry = registry
        self._sched_registry = sched_registry
        self._max_concurrent_runs = max_concurrent_runs
        self._running = 0
        self._lock = asyncio.Lock()
        self._run_waiters: dict[str, asyncio.Future] = {}
        self._run_waiters_lock = (
            asyncio.Lock()
        )  # no need for thread lock because run_manager is used within event loop

    # -------- concurrency helpers --------
    async def _acquire_run_slot(self) -> None:
        if self._max_concurrent_runs is None:
            return
        async with self._lock:
            if self._running >= self._max_concurrent_runs:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many runs are currently executing. Please wait and try again.",
                )
            self._running += 1

    async def _release_run_slot(self) -> None:
        if self._max_concurrent_runs is None:
            return
        async with self._lock:
            self._running = max(0, self._running - 1)

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
    async def _build_run_record(
        self,
        *,
        graph_id: str,
        run_id: str | None,
        session_id: str | None,
        tags: list[str] | None,
        identity: RequestIdentity,
        origin: RunOrigin | None,
        visibility: RunVisibility | None,
        importance: RunImportance | None,
        agent_id: str | None,
        app_id: str | None,
    ) -> tuple[RunRecord, Any]:
        """
        Shared helper for submit_run and run_and_wait:
        - Resolves target
        - Determines kind
        - Attaches flow_id and session tags

        Return:
        - RunRecord (not yet persisted)
        - target object: graph or graphfn
        """
        rid = run_id or f"run-{uuid4().hex[:8]}"
        started_at = _utcnow()
        tags = list(tags or [])

        target = await self._resolve_target(graph_id)
        if _is_task_graph(target):
            kind = "taskgraph"
        elif _is_graphfn(target):
            kind = "graphfn"
        else:
            kind = "other"

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

        if session_id is None:
            session_id = rid

        record = RunRecord(
            run_id=rid,
            graph_id=graph_id,
            kind=kind,
            status=RunStatus.running,
            started_at=started_at,
            tags=tags,
            user_id=identity.user_id,
            org_id=identity.org_id,
            meta={},
            session_id=session_id,
            origin=origin or RunOrigin.app,
            visibility=visibility or RunVisibility.normal,
            importance=importance or RunImportance.normal,
            agent_id=agent_id,
            app_id=app_id,
        )

        if flow_id:
            record.meta["flow_id"] = flow_id
            if f"flow:{flow_id}" not in record.tags:
                record.tags.append(f"flow:{flow_id}")
        if session_id:
            record.meta["session_id"] = session_id
            if f"session:{session_id}" not in record.tags:
                record.tags.append(f"session:{session_id}")

        return record, target

    async def _run_and_finalize(
        self,
        *,
        record: RunRecord,
        target: Any,
        graph_id: str,
        inputs: dict[str, Any],
        identity: RequestIdentity,
        # user_id: str | None,
        # org_id: str | None,
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Shared core logic that actually calls run_or_resume_async, updates
        RunStore, and records metering.

        Returns:
          (record, outputs, has_waits, continuations)
        """
        from aethergraph.core.runtime.graph_runner import run_or_resume_async

        user_id = identity.user_id
        org_id = identity.org_id

        # tags = record.tags or []
        started_at = record.started_at or _utcnow()

        outputs: dict[str, Any] | None = None
        has_waits = False
        continuations: list[dict[str, Any]] = []
        error_msg: str | None = None

        try:
            result = await run_or_resume_async(
                target,
                inputs or {},
                run_id=record.run_id,
                session_id=record.meta.get("session_id"),
                identity=identity,
                agent_id=record.agent_id,
                app_id=record.app_id,
            )
            # If we get here without GraphHasPendingWaits, run is completed
            outputs = result if isinstance(result, dict) else {"result": result}
            record.status = RunStatus.succeeded
            record.finished_at = _utcnow()

            # Optional: store a UI-only output preview
            try:
                preview, truncated = _make_preview(outputs)
                record.meta["output_preview"] = preview
                record.meta["output_truncated"] = truncated
            except Exception:
                import logging

                logging.getLogger("aethergraph.runtime.run_manager").exception(
                    "Error creating output preview for run_id=%s", record.run_id
                )

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

        try:
            if record.status in {RunStatus.succeeded, RunStatus.failed, RunStatus.canceled}:
                # IMPORTANT: now resolve with (record, outputs)
                await self._resolve_run_future(record.run_id, (record, outputs))
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Error resolving run future for run_id=%s", record.run_id
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
        identity: RequestIdentity | None = None,
        origin: RunOrigin | None = None,
        visibility: RunVisibility | None = None,
        importance: RunImportance | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ) -> RunRecord:
        """
        Non-blocking entrypoint for the HTTP API.

        - Creates a RunRecord (status=running).
        - Persists it to RunStore.
        - Schedules background execution via asyncio.create_task.
        - Returns immediately with the record (for run_id, status, etc).
        """
        if identity is None:
            identity = RequestIdentity(user_id="local", org_id="local", mode="local")

        # Acquire run slot (rate limiting)
        await self._acquire_run_slot()
        # Tracks whether responsibility for releasing the slot has been handed
        # over to the background runner (_bg). If False, submit_run must
        # release the slot on exception; if True, _bg will do it its finally.
        slot_handed_to_bg = False

        try:
            tags = tags or []

            record, target = await self._build_run_record(
                graph_id=graph_id,
                run_id=run_id,
                session_id=session_id,
                tags=tags,
                identity=identity,
                origin=origin,
                visibility=visibility,
                importance=importance,
                agent_id=agent_id,
                app_id=app_id,
            )

            # Optional: store a UI-only input preview
            try:
                preview, truncated = _make_preview(inputs)
                record.meta["input_preview"] = preview
                record.meta["input_truncated"] = truncated
            except Exception:
                import logging

                logging.getLogger("aethergraph.runtime.run_manager").exception(
                    "Error creating input preview for run_id=%s", record.run_id
                )

            if self._store is not None:
                await self._store.create(record)

            async def _bg():
                try:
                    await self._run_and_finalize(
                        record=record,
                        target=target,
                        graph_id=graph_id,
                        inputs=inputs,
                        identity=identity,
                    )
                finally:
                    await self._release_run_slot()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                slot_handed_to_bg = True
                await _bg()
            else:
                slot_handed_to_bg = True
                loop.create_task(_bg())

            return record

        except Exception:
            # If submit_run itself fails *before* handing off to _bg, we must release the slot here.
            # Once slot_handed_to_bg is True, _bg is responsible for releasing the slot.
            if not slot_handed_to_bg:
                await self._release_run_slot()
            raise

    async def run_and_wait(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        run_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        identity: RequestIdentity | None = None,
        origin: RunOrigin | None = None,
        visibility: RunVisibility | None = None,
        importance: RunImportance | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        count_slot: bool = False,  # important for nested orchestration
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Blocking run that still goes through RunStore so UI can visualize it.

        - Creates + persists RunRecord (status=running)
        - Runs inline (awaits completion)
        - Updates RunStore status + metering (via _run_and_finalize)
        - Returns (record, outputs, has_waits, continuations)

        count_slot=False is recommended for "parent run awaiting child run" orchestration
        to avoid deadlocks when max_concurrent_runs is small.
        """
        if identity is None:
            identity = RequestIdentity(user_id="local", org_id="local", mode="local")

        if count_slot:
            await self._acquire_run_slot()

        try:
            tags = tags or []

            record, target = await self._build_run_record(
                graph_id=graph_id,
                run_id=run_id,
                session_id=session_id,
                tags=tags,
                identity=identity,
                origin=origin,
                visibility=visibility,
                importance=importance,
                agent_id=agent_id,
                app_id=app_id,
            )

            # Optional: UI-only input preview
            try:
                preview, truncated = _make_preview(inputs)
                record.meta["input_preview"] = preview
                record.meta["input_truncated"] = truncated
            except Exception:
                import logging

                logging.getLogger("aethergraph.runtime.run_manager").exception(
                    "Error creating input preview for run_id=%s", record.run_id
                )

            if self._store is not None:
                await self._store.create(record)

            return await self._run_and_finalize(
                record=record,
                target=target,
                graph_id=graph_id,
                inputs=inputs,
                identity=identity,
            )
        finally:
            if count_slot:
                await self._release_run_slot()

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
        flow_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        records = await self._store.list(
            graph_id=graph_id,
            status=status,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
        # Optional: still filter flow_id in Python for now since it's in meta/tags
        if flow_id is not None:
            records = [rec for rec in records if (rec.meta or {}).get("flow_id") == flow_id]

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
                # if local scheduler -> terminate
                # if global scheduler -> terminate_run(run_id)
                if isinstance(sched, GlobalForwardScheduler):
                    await sched.terminate_run(run_id)
                    return
                elif isinstance(sched, ForwardScheduler):
                    await sched.terminate()
                    return
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

    # ------- run waiters for orchestration --------
    async def wait_run(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
        return_outputs: bool = False,
    ) -> RunRecord | tuple[RunRecord, dict[str, Any] | None]:
        """
        Wait for a run to reach a terminal state.

        - If return_outputs=False (default), returns RunRecord (backwards compatible).
        - If return_outputs=True, returns (RunRecord, outputs_dict_or_none).

        NOTE:
        - outputs are only guaranteed when the run was executed in THIS process
          via RunManager. If the run is already terminal in the store and no
          in-process outputs exist, outputs will be None.
        """
        # Fast path: already terminal in store
        rec = await self.get_record(run_id)
        if rec and rec.status in {RunStatus.succeeded, RunStatus.failed, RunStatus.canceled}:
            if return_outputs:
                return rec, None
            return rec

        fut = await self._get_or_create_run_future(run_id)

        if timeout_s is not None:
            result = await asyncio.wait_for(fut, timeout=timeout_s)
        else:
            result = await fut

        # result is either:
        # - RunRecord (old-style resolvers)
        # - or (RunRecord, outputs) from _run_and_finalize
        if isinstance(result, RunRecord):
            if return_outputs:
                return result, None
            return result

        rec2, outputs = result
        if return_outputs:
            return rec2, outputs
        return rec2

    async def _get_or_create_run_future(self, run_id: str) -> asyncio.Future:
        async with self._run_waiters_lock:
            fut = self._run_waiters.get(run_id)
            if fut is None or fut.done():
                fut = asyncio.get_running_loop().create_future()
                self._run_waiters[run_id] = fut
            return fut

    async def _resolve_run_future(self, run_id: str, value: Any) -> None:
        async with self._run_waiters_lock:
            fut = self._run_waiters.get(run_id)
            if fut and not fut.done():
                fut.set_result(value)
            # optional cleanup
            self._run_waiters.pop(run_id, None)

    async def _reject_run_future(self, run_id: str, err: Exception) -> None:
        async with self._run_waiters_lock:
            fut = self._run_waiters.get(run_id)
            if fut and not fut.done():
                fut.set_exception(err)
            self._run_waiters.pop(run_id, None)

    # -------- old: blocking start_run (CLI/tests) --------
    async def start_run(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        run_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        identity: RequestIdentity | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ) -> tuple[RunRecord, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Blocking helper (original behaviour).

        - Resolves target.
        - Creates RunRecord with status=running.
        - Runs once via run_or_resume_async.
        - Updates store + metering.
        - Returns (record, outputs, has_waits, continuations).

        Still useful for tests/CLI, but the HTTP route should prefer submit_run().

        NOTE:
        agent_id and app_id will override any value pulled from original graphs. Use it
        only when you want to explicitly set these fields for tracking purpose.
        """
        if identity is None:
            identity = RequestIdentity(user_id="local", org_id="local", mode="local")

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
            session_id = rid

        record = RunRecord(
            run_id=rid,
            graph_id=graph_id,
            kind=kind,
            status=RunStatus.running,  # we go straight to running as before
            started_at=started_at,
            tags=list(tags),
            user_id=identity.user_id,
            org_id=identity.org_id,
            meta={},
            session_id=session_id,
            origin=RunOrigin.app,  # app is a typical default for graph runs
            visibility=RunVisibility.normal,
            importance=RunImportance.normal,
            agent_id=agent_id,
            app_id=app_id,
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
            identity=identity,
            # agent_id=agent_id,
            # app_id=app_id,
        )
