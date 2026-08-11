from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import json
import logging
import traceback
from typing import Any
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.errors.errors import GraphBuildError, GraphHasPendingWaits
from aethergraph.contracts.services.runs import RunResultStore, RunStore
from aethergraph.contracts.services.state_stores import GraphStateStore
from aethergraph.core.execution.forward_scheduler import ForwardScheduler
from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler
from aethergraph.core.runtime.run_cancellation import (
    RunCancellationHandle,
    RunCancellationRegistry,
    RunCancellationRequestedError,
    get_run_cancellation_registry,
)
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunResult,
    RunStatus,
    RunVisibility,
    _make_preview,
)
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.scope.tenant import registry_tenant_from_identity


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _is_task_graph(obj: Any) -> bool:
    # Replace with proper isinstance check in your codebase
    return hasattr(obj, "spec") and hasattr(obj, "io_signature")


def _is_graphfn(obj: Any) -> bool:
    from aethergraph.core.graph.graph_fn import GraphFunction  # adjust path

    return isinstance(obj, GraphFunction)


class DuplicateRunIdError(RuntimeError):
    """Raised when a requested run_id already exists in the RunStore."""


def _is_duplicate_run_id_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "unique constraint failed" in msg and "runs.run_id" in msg


def _clone_jsonish_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(json.dumps(value, default=repr))


def _run_status(value: Any) -> RunStatus | None:
    if isinstance(value, RunStatus):
        return value
    try:
        return RunStatus(str(value))
    except Exception:
        return None


def _run_status_text(value: Any) -> str:
    status = _run_status(value)
    return status.value if status is not None else str(value)


_log = logging.getLogger("aethergraph.runtime.run_manager")

# Root-turn admission is a session-consistency guard, not a general scheduler
# throttle. It exists because UI/channel callers may submit the next visible
# user turn as soon as a planner message is displayed, while the previous root
# run is still saving final session state. Starting the next root run during
# that finalization window lets the new planner observe stale session artifacts.
#
# Only statuses that still imply "the prior root turn may be mutating session
# state" block admission. RunStatus.waiting is intentionally excluded: approval
# and resume traffic must be able to enter a session once the previous root turn
# has deliberately parked on a continuation.
_SESSION_ROOT_TURN_BARRIER_STATUSES = frozenset(
    {
        RunStatus.pending,
        RunStatus.running,
        RunStatus.cancellation_requested,
    }
)
_SESSION_ROOT_TURN_BARRIER_TIMEOUT_S = 10.0
_SESSION_ROOT_TURN_BARRIER_POLL_S = 0.05

# These runs are orchestration/runtime work created underneath an already
# admitted root turn. They must be allowed to overlap the parent; otherwise
# async children, completion notifiers, and resumption helpers can deadlock
# behind the very root run they are supposed to finish.
_INTERNAL_RUN_TAGS = frozenset(
    {
        "aethergraph_engine._internal",
        "async_hop",
        "notifier",
        "runner_resumption",
    }
)
_INTERNAL_RUN_TAG_PREFIXES = ("plan_step:", "trigger:")

# These origins represent visible/user-facing root turns. RunOrigin.agent and
# RunOrigin.schedule are intentionally excluded so child/subagent runs and
# scheduled work do not inherit chat-turn serialization semantics by accident.
_ROOT_TURN_ORIGINS = frozenset(
    {
        RunOrigin.app,
        RunOrigin.chat,
        RunOrigin.playground,
        RunOrigin.api,
        RunOrigin.cli,
        RunOrigin.local,
    }
)


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
        result_store: RunResultStore | None = None,
        state_store: GraphStateStore | None = None,
        registry: UnifiedRegistry | None = None,
        sched_registry: Any | None = None,  # placeholder for future use
        cancellation_registry: RunCancellationRegistry | None = None,
        max_concurrent_runs: int | None = None,
    ):
        self._store = run_store
        self._result_store = result_store
        self._state_store = state_store
        self._registry = registry
        self._sched_registry = sched_registry
        self._cancellation_registry = cancellation_registry
        self._max_concurrent_runs = max_concurrent_runs
        self._running = 0
        self._lock = asyncio.Lock()
        self._run_waiters: dict[str, asyncio.Future] = {}
        self._run_waiters_lock = (
            asyncio.Lock()
        )  # no need for thread lock because run_manager is used within event loop
        self._session_root_turn_locks: dict[str, asyncio.Lock] = {}
        self._session_root_turn_locks_lock = asyncio.Lock()

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

    async def _acquire_session_root_turn_admission(
        self,
        *,
        session_id: str | None,
        graph_id: str,
        run_id: str | None,
        tags: list[str] | None,
        origin: RunOrigin | None,
        visibility: RunVisibility | None,
        run_config: dict[str, Any] | None,
    ) -> asyncio.Lock | None:
        """Acquire same-session root-turn admission when this run needs it.

        This protects only the short admission window: we hold the per-session
        lock while checking persisted blockers and creating the new run record.
        After the record exists, later callers can rely on the run store to see
        that this root turn is running and should block behind it.
        """
        if not self._should_gate_session_root_turn(
            session_id=session_id,
            tags=tags,
            origin=origin,
            visibility=visibility,
            run_config=run_config,
        ):
            return None
        session_key = str(session_id)
        async with self._session_root_turn_locks_lock:
            lock = self._session_root_turn_locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_root_turn_locks[session_key] = lock
        await lock.acquire()
        try:
            await self._wait_for_session_root_turn_barrier(
                session_id=session_key,
                graph_id=graph_id,
                run_id=run_id,
            )
        except Exception:
            lock.release()
            raise
        return lock

    @staticmethod
    def _release_session_root_turn_admission(lock: asyncio.Lock | None) -> None:
        if lock is not None and lock.locked():
            lock.release()

    def _should_gate_session_root_turn(
        self,
        *,
        session_id: str | None,
        tags: list[str] | None,
        origin: RunOrigin | None,
        visibility: RunVisibility | None,
        run_config: dict[str, Any] | None,
    ) -> bool:
        # Future control-plane implementations should decide deliberately
        # whether they are root turns. Direct APIs such as cancel_run/get_record
        # bypass this path already. If a natural-language "cancel/check status"
        # message must go through submit_run, give it a distinct origin/tag or
        # run_config marker and handle that marker here.
        if self._store is None or not session_id:
            return False
        if visibility == RunVisibility.hidden:
            return False
        if self._is_internal_run_metadata(tags=tags, run_config=run_config):
            return False
        return (origin or RunOrigin.app) in _ROOT_TURN_ORIGINS

    @staticmethod
    def _is_internal_run_metadata(
        *,
        tags: list[str] | None,
        run_config: dict[str, Any] | None = None,
    ) -> bool:
        tag_set = {str(tag) for tag in list(tags or [])}
        if tag_set.intersection(_INTERNAL_RUN_TAGS):
            return True
        if any(tag.startswith(_INTERNAL_RUN_TAG_PREFIXES) for tag in tag_set):
            return True
        resume_mode = str((run_config or {}).get("resume_mode") or "")
        return resume_mode == "runner_resumption"

    def _is_session_root_turn_record(self, record: RunRecord) -> bool:
        if getattr(record, "visibility", None) == RunVisibility.hidden:
            return False
        if self._is_internal_run_metadata(
            tags=list(record.tags or []), run_config=record.meta or {}
        ):
            return False
        return (getattr(record, "origin", None) or RunOrigin.app) in _ROOT_TURN_ORIGINS

    async def _wait_for_session_root_turn_barrier(
        self,
        *,
        session_id: str,
        graph_id: str,
        run_id: str | None,
    ) -> None:
        if self._store is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _SESSION_ROOT_TURN_BARRIER_TIMEOUT_S
        first_blocker_id = ""
        waited = False
        while True:
            blocker = await self._find_session_root_turn_blocker(session_id=session_id)
            if blocker is None:
                if waited:
                    _log.debug(
                        "session root-turn barrier released",
                        extra={
                            "session_id": session_id,
                            "graph_id": graph_id,
                            "run_id": run_id,
                            "blocked_by": first_blocker_id,
                        },
                    )
                return
            waited = True
            if not first_blocker_id:
                first_blocker_id = blocker.run_id
                _log.debug(
                    "session root-turn barrier waiting",
                    extra={
                        "session_id": session_id,
                        "graph_id": graph_id,
                        "run_id": run_id,
                        "blocked_by": blocker.run_id,
                        "blocked_status": _run_status_text(blocker.status),
                    },
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                _log.warning(
                    "session root-turn barrier timed out; admitting run",
                    extra={
                        "session_id": session_id,
                        "graph_id": graph_id,
                        "run_id": run_id,
                        "blocked_by": blocker.run_id,
                        "blocked_status": _run_status_text(blocker.status),
                    },
                )
                return
            await asyncio.sleep(min(_SESSION_ROOT_TURN_BARRIER_POLL_S, remaining))

    async def _find_session_root_turn_blocker(self, *, session_id: str) -> RunRecord | None:
        if self._store is None:
            return None
        try:
            records = await self._store.list(session_id=session_id, limit=25)
        except Exception:
            _log.exception(
                "session root-turn barrier failed to list runs",
                extra={"session_id": session_id},
            )
            return None
        for record in records:
            status = _run_status(record.status)
            if status not in _SESSION_ROOT_TURN_BARRIER_STATUSES:
                continue
            # Non-root records may legitimately be newer than the prior root
            # turn in the same session. Keep scanning until we find a visible
            # root turn that can still be mutating shared session state.
            if self._is_session_root_turn_record(record):
                return record
        return None

    # -------- registry helpers --------

    def registry(self) -> UnifiedRegistry:
        return self._registry or current_registry()

    def _get_result_store(self) -> RunResultStore | None:
        if self._result_store is not None:
            return self._result_store
        try:
            container = current_services()
        except Exception:
            return None
        return getattr(container, "run_result_store", None)

    def _get_state_store(self) -> GraphStateStore | None:
        if self._state_store is not None:
            return self._state_store
        try:
            container = current_services()
        except Exception:
            return None
        return getattr(container, "state_store", None)

    @staticmethod
    def _identity_for_record(record: RunRecord) -> RequestIdentity:
        return RequestIdentity(
            user_id=record.user_id,
            org_id=record.org_id,
            mode="local",
        )

    async def _persist_run_result(
        self,
        *,
        record: RunRecord,
        outputs: dict[str, Any],
        source: str,
        snapshot_rev: int | None = None,
    ) -> RunResult | None:
        result_store = self._get_result_store()
        if result_store is None:
            return None
        try:
            now = _utcnow()
            existing = await result_store.get(record.run_id)
            result = RunResult(
                run_id=record.run_id,
                graph_id=record.graph_id,
                session_id=record.session_id,
                status=RunStatus.succeeded,
                outputs=dict(outputs),
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
                source=source,
                snapshot_rev=snapshot_rev,
            )
            await result_store.save(record.run_id, result)
            record.result_available = True
            record.result_updated_at = result.updated_at
            if self._store is not None:
                await self._store.update_status(
                    record.run_id,
                    record.status,
                    finished_at=record.finished_at,
                    error=record.error,
                    field_updates={
                        "result_available": True,
                        "result_updated_at": result.updated_at,
                    },
                )
            return result
        except Exception:
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Error persisting durable run outputs for run_id=%s", record.run_id
            )
            return None

    async def _recover_outputs_from_snapshot(
        self,
        record: RunRecord,
    ) -> tuple[dict[str, Any] | None, int | None]:
        store = self._get_state_store()
        if store is None:
            return None, None
        snap = await store.load_latest_snapshot(record.run_id)
        if snap is None:
            return None, None
        graph_outputs = snap.state.get("graph_outputs")
        if isinstance(graph_outputs, dict):
            return dict(graph_outputs), snap.rev

        from aethergraph.core.runtime.graph_runner import (
            _materialize_task_graph,
            _resolve_graph_outputs,
            _seed_outputs_from_snapshot,
        )
        from aethergraph.core.runtime.runtime_env import RuntimeEnv
        from aethergraph.services.container.default_container import build_default_container

        identity = self._identity_for_record(record)
        self._resolve_target_identity = identity
        try:
            target = await self._resolve_target(record.graph_id)
        finally:
            self._resolve_target_identity = None
        try:
            graph = _materialize_task_graph(target)
        except Exception:
            return None, snap.rev

        inputs = dict((record.meta or {}).get("original_inputs") or {})
        try:
            container = current_services()
        except Exception:
            container = build_default_container()
        env = RuntimeEnv(
            run_id=record.run_id,
            graph_id=record.graph_id,
            session_id=record.session_id,
            identity=identity,
            graph_inputs=inputs,
            outputs_by_node={},
            container=container,
            agent_id=record.agent_id,
            app_id=record.app_id,
        )
        _seed_outputs_from_snapshot(env, snap)
        outputs = await _resolve_graph_outputs(graph, inputs, env)
        return outputs, snap.rev

    async def _durable_outputs_for_record(self, record: RunRecord) -> dict[str, Any] | None:
        if record.status != RunStatus.succeeded:
            return None
        result_store = self._get_result_store()
        if result_store is not None:
            existing = await result_store.get(record.run_id)
            if existing is not None:
                record.result_available = True
                record.result_updated_at = existing.updated_at
                return dict(existing.outputs)

        outputs, snapshot_rev = await self._recover_outputs_from_snapshot(record)
        if outputs is None:
            return None
        await self._persist_run_result(
            record=record,
            outputs=outputs,
            source="snapshot_recovered",
            snapshot_rev=snapshot_rev,
        )
        return outputs

    async def _resolve_target(self, graph_id: str) -> Any:
        reg = self.registry()
        identity = getattr(self, "_resolve_target_identity", None)
        tenant = registry_tenant_from_identity(identity) if identity is not None else None
        # Try static TaskGraph
        try:
            return reg.get_graph(name=graph_id, version=None, tenant=tenant, include_global=True)
        except KeyError:
            pass
        # Try GraphFunction
        try:
            return reg.get_graphfn(name=graph_id, version=None, tenant=tenant, include_global=True)
        except KeyError:
            pass
        raise KeyError(f"Graph '{graph_id}' not found")

    # -------- core execution helper --------
    async def _build_run_record(
        self,
        *,
        graph_id: str,
        inputs: dict[str, Any],
        run_id: str | None,
        session_id: str | None,
        tags: list[str] | None,
        identity: RequestIdentity,
        origin: RunOrigin | None,
        visibility: RunVisibility | None,
        importance: RunImportance | None,
        agent_id: str | None,
        app_id: str | None,
        app_name: str | None,
        run_config: dict[str, Any] | None,
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
        rid = run_id or f"run-{uuid4().hex[:12]}"
        started_at = _utcnow()
        tags = list(tags or [])

        tenant = registry_tenant_from_identity(identity)
        self._resolve_target_identity = identity
        try:
            target = await self._resolve_target(graph_id)
        finally:
            self._resolve_target_identity = None
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
                meta = (
                    reg.get_meta(
                        nspace="graph",
                        name=graph_id,
                        version=None,
                        tenant=tenant,
                        include_global=True,
                    )
                    or {}
                )
            elif kind == "graphfn":
                meta = (
                    reg.get_meta(
                        nspace="graphfn",
                        name=graph_id,
                        version=None,
                        tenant=tenant,
                        include_global=True,
                    )
                    or {}
                )
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
            tags=list(tags),
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

        record.meta["original_inputs"] = _clone_jsonish_dict(inputs)
        record.meta["original_run_config"] = _clone_jsonish_dict(run_config)
        record.meta["original_tags"] = list(tags or [])
        record.meta["original_session_id"] = session_id
        record.meta["original_app_id"] = app_id
        if app_name:
            record.meta["app_name"] = app_name
        if agent_id:
            record.meta["agent_id"] = agent_id

        resume_from_run_id = None
        resume_mode = None
        if run_config:
            resume_from_run_id = run_config.get("resume_from_run_id")
            resume_mode = run_config.get("resume_mode")
        if resume_from_run_id:
            record.meta["resume_from_run_id"] = resume_from_run_id
        if resume_mode:
            record.meta["resume_mode"] = resume_mode

        return record, target

    async def _run_and_finalize(
        self,
        *,
        record: RunRecord,
        target: Any,
        graph_id: str,
        inputs: dict[str, Any],
        identity: RequestIdentity,
        run_config: dict[str, Any] | None = None,
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
        handle = await self._ensure_cancellation_handle(record.run_id)

        try:
            result = await run_or_resume_async(
                target,
                inputs or {},
                run_id=record.run_id,
                session_id=record.meta.get("session_id"),
                identity=identity,
                agent_id=record.agent_id,
                app_id=record.app_id,
                **(run_config or {}),
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
            await self._persist_run_result(
                record=record,
                outputs=outputs,
                source="direct",
            )

        except (asyncio.CancelledError, RunCancellationRequestedError):
            # Cancellation path: scheduler.terminate() or external cancel.
            import logging

            backend_state = await handle.backend_state()
            handle.mark_backend_stopped(backend_state=backend_state)
            record.status = RunStatus.canceled
            record.finished_at = _utcnow()
            error_msg = "Run cancelled by user"
            record.error = error_msg
            record.meta["error_kind"] = "cancellation"
            record.meta["error_code"] = "run_cancelled"
            record.meta["error_stage"] = "run_execution"
            record.meta["error_hints"] = []
            record.meta["error_message"] = error_msg
            record.meta["error_detail"] = None
            record.meta["error_is_traceback"] = False
            record.meta["error_info"] = {
                "message": error_msg,
                "detail": None,
                "kind": "cancellation",
                "stage": "run_execution",
                "code": "run_cancelled",
                "hints": [],
                "is_traceback": False,
            }
            logging.getLogger("aethergraph.runtime.run_manager").info(
                "Run %s was cancelled", record.run_id
            )

        except GraphHasPendingWaits as e:
            # Graph quiesced with pending waits
            record.status = RunStatus.waiting
            has_waits = True
            continuations = getattr(e, "continuations", [])
            # outputs remain None

        except GraphBuildError as exc:
            record.status = RunStatus.failed
            record.finished_at = _utcnow()
            error_msg = str(exc)
            record.error = error_msg
            record.meta["error_kind"] = "build"
            record.meta["error_code"] = exc.code
            record.meta["error_stage"] = exc.stage
            record.meta["error_hints"] = list(exc.hints or [])
            record.meta["error_message"] = error_msg
            record.meta["error_detail"] = traceback.format_exc()
            record.meta["error_is_traceback"] = True
            record.meta["error_info"] = {
                "message": error_msg,
                "detail": record.meta["error_detail"],
                "kind": "build",
                "stage": exc.stage,
                "code": exc.code,
                "hints": list(exc.hints or []),
                "is_traceback": True,
            }
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Run %s failed with build error: %s", record.run_id, error_msg
            )

        except Exception as exc:  # noqa: BLE001
            record.status = RunStatus.failed
            record.finished_at = _utcnow()
            error_msg = str(exc)
            record.error = error_msg
            record.meta["error_kind"] = "runtime"
            record.meta["error_code"] = None
            record.meta["error_stage"] = None
            record.meta["error_hints"] = []
            record.meta["error_message"] = error_msg
            record.meta["error_detail"] = traceback.format_exc()
            record.meta["error_is_traceback"] = True
            record.meta["error_info"] = {
                "message": error_msg,
                "detail": record.meta["error_detail"],
                "kind": "runtime",
                "stage": "run_execution",
                "code": exc.__class__.__name__,
                "hints": [],
                "is_traceback": True,
            }
            import logging

            logging.getLogger("aethergraph.runtime.run_manager").exception(
                "Run %s failed with exception: %s", record.run_id, error_msg
            )

        # Persist status update
        record.meta.update({k: v for k, v in handle.metadata().items() if v is not None})
        if self._store is not None:
            await self._store.update_status(
                record.run_id,
                record.status,
                finished_at=record.finished_at,
                error=error_msg,
                meta_update={k: v for k, v in handle.metadata().items() if v is not None},
                field_updates={
                    "result_available": record.result_available,
                    "result_updated_at": record.result_updated_at,
                },
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

        if record.status in {RunStatus.succeeded, RunStatus.failed, RunStatus.canceled}:
            await self._get_cancellation_registry().pop(record.run_id)

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
        app_name: str | None = None,
        run_config: dict[str, Any] | None = None,
        admission_callback: Callable[[RunRecord], Awaitable[None]] | None = None,
    ) -> RunRecord:
        """Persist and admit one run before scheduling background execution.

        Examples:
            Submit an ordinary run::

                record = await manager.submit_run("research", inputs={"topic": "AI"})

            Bind an external execution lease before scheduling::

                record = await manager.submit_run(
                    "research",
                    inputs={"topic": "AI"},
                    admission_callback=persist_execution_lease,
                )

        Args:
            graph_id: Exact registered graph or graph-function identity.
            inputs: Inputs passed to the selected graph.
            run_id: Optional caller-owned run identity.
            session_id: Optional session used for root-turn serialization.
            tags: Run classification tags.
            identity: Authenticated request identity.
            origin: Run origin used by scheduling and inspection.
            visibility: Run visibility policy.
            importance: Run retention importance.
            agent_id: Optional owning Agent identity.
            app_id: Optional owning application identity.
            app_name: Optional application display name.
            run_config: Optional trusted runtime configuration.
            admission_callback: Optional Host callback invoked after durable run creation
                and before execution becomes eligible to start.

        Returns:
            RunRecord: Persisted and admitted run metadata.

        Notes:
            A callback failure terminalizes the persisted run as failed and schedules
            no graph work. There is no post-scheduling admission path.
        """
        if identity is None:
            identity = RequestIdentity(user_id="local", org_id="local", mode="local")

        # Gate before creating the RunRecord. Once this method persists a root
        # run, that record itself becomes the blocker for later same-session
        # root turns; the in-process lock only closes the check/create race.
        admission_lock = await self._acquire_session_root_turn_admission(
            session_id=session_id,
            graph_id=graph_id,
            run_id=run_id,
            tags=tags,
            origin=origin,
            visibility=visibility,
            run_config=run_config,
        )
        # Acquire run slot (rate limiting)
        # Tracks whether responsibility for releasing the slot has been handed
        # over to the background runner (_bg). If False, submit_run must
        # release the slot on exception; if True, _bg will do it its finally.
        slot_handed_to_bg = False

        try:
            await self._acquire_run_slot()
            tags = tags or []

            record: RunRecord | None = None
            target: Any = None
            max_attempts = 5 if run_id is None else 1
            for _ in range(max_attempts):
                record, target = await self._build_run_record(
                    graph_id=graph_id,
                    inputs=inputs,
                    run_id=run_id,
                    session_id=session_id,
                    tags=tags,
                    identity=identity,
                    origin=origin,
                    visibility=visibility,
                    importance=importance,
                    agent_id=agent_id,
                    app_id=app_id,
                    app_name=app_name,
                    run_config=run_config,
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

                if self._store is None:
                    break

                try:
                    await self._store.create(record)
                    break
                except Exception as e:
                    if not _is_duplicate_run_id_error(e):
                        raise
                    if run_id is not None:
                        raise DuplicateRunIdError(f"Run id '{run_id}' already exists") from e
                    # Auto-generated id collision: retry with a fresh id.
                    continue
            else:
                raise RuntimeError("Failed to allocate a unique run_id after retries")

            if record is None:
                raise RuntimeError("Failed to create run record")
            await self._ensure_cancellation_handle(record.run_id)
            if admission_callback is not None:
                try:
                    await admission_callback(record)
                except Exception as exc:
                    record.status = RunStatus.failed
                    record.finished_at = _utcnow()
                    record.error = "Run admission callback failed"
                    record.meta.update(
                        {
                            "error_kind": "admission",
                            "error_code": "run_admission_failed",
                            "error_stage": "run_admission",
                            "error_message": record.error,
                            "error_detail": None,
                            "error_is_traceback": False,
                        }
                    )
                    if self._store is not None:
                        await self._store.update_status(
                            record.run_id,
                            RunStatus.failed,
                            finished_at=record.finished_at,
                            error=record.error,
                            meta_update={
                                "error_kind": "admission",
                                "error_code": "run_admission_failed",
                                "error_stage": "run_admission",
                                "error_message": record.error,
                                "error_detail": None,
                                "error_is_traceback": False,
                            },
                        )
                    await self._resolve_run_future(record.run_id, (record, None))
                    await self._get_cancellation_registry().pop(record.run_id)
                    raise RuntimeError("Run admission callback failed") from exc

            async def _bg():
                try:
                    finalize_kwargs = {
                        "record": record,
                        "target": target,
                        "graph_id": graph_id,
                        "inputs": inputs,
                        "identity": identity,
                    }
                    if run_config is not None:
                        finalize_kwargs["run_config"] = run_config
                    await self._run_and_finalize(
                        **finalize_kwargs,
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
        finally:
            self._release_session_root_turn_admission(admission_lock)

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
        app_name: str | None = None,
        run_config: dict[str, Any] | None = None,
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

        # run_and_wait also creates persisted root records, so it participates
        # in the same session admission semantics as submit_run. The lock is
        # released immediately after record creation; execution can still run
        # inline while later root turns observe this record in the store.
        admission_lock = await self._acquire_session_root_turn_admission(
            session_id=session_id,
            graph_id=graph_id,
            run_id=run_id,
            tags=tags,
            origin=origin,
            visibility=visibility,
            run_config=run_config,
        )
        try:
            if count_slot:
                await self._acquire_run_slot()
            tags = tags or []

            record, target = await self._build_run_record(
                graph_id=graph_id,
                inputs=inputs,
                run_id=run_id,
                session_id=session_id,
                tags=tags,
                identity=identity,
                origin=origin,
                visibility=visibility,
                importance=importance,
                agent_id=agent_id,
                app_id=app_id,
                app_name=app_name,
                run_config=run_config,
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
            self._release_session_root_turn_admission(admission_lock)
            admission_lock = None
            await self._ensure_cancellation_handle(record.run_id)

            finalize_kwargs = {
                "record": record,
                "target": target,
                "graph_id": graph_id,
                "inputs": inputs,
                "identity": identity,
            }
            if run_config is not None:
                finalize_kwargs["run_config"] = run_config
            return await self._run_and_finalize(**finalize_kwargs)
        finally:
            self._release_session_root_turn_admission(admission_lock)
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

    def _get_cancellation_registry(self) -> RunCancellationRegistry:
        if self._cancellation_registry is not None:
            return self._cancellation_registry
        try:
            container = current_services()
        except Exception:
            container = None
        return get_run_cancellation_registry(container)

    async def _ensure_cancellation_handle(self, run_id: str) -> RunCancellationHandle:
        return await self._get_cancellation_registry().create(run_id)

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "user_requested",
    ) -> RunRecord | None:
        """Request best-effort cancellation and preserve its semantic cause.

        Examples:
            Cancel a user-requested run:
            ```python
            await manager.cancel_run("run-1")
            ```

            Cancel a child owned by a stopped parent:
            ```python
            await manager.cancel_run("run-child", reason="parent_cancelled")
            ```

        Args:
            run_id: Exact run identity to cancel.
            reason: Exact supported cancellation cause.

        Returns:
            RunRecord | None: Current run record when present, otherwise
            `None` after best-effort cancellation dispatch.

        Notes:
            The scheduler remains responsible for physical termination and the
            existing terminal `RunStatus.canceled` transition.
        """

        reason = str(reason or "")
        if reason not in {"user_requested", "parent_cancelled"}:
            raise ValueError(f"Unsupported cancellation reason: {reason}")
        record: RunRecord | None = None
        if self._store is not None:
            record = await self._store.get(run_id)
        handle = await self._get_cancellation_registry().get(run_id)

        # Helper: scheduler-level termination
        async def _terminate_scheduler() -> dict[str, Any] | None:
            reg = self._get_sched_registry()
            if reg is None:
                return None
            sched = reg.get(run_id)
            if sched is None:
                return None

            try:
                # if local scheduler -> terminate
                # if global scheduler -> terminate_run(run_id)
                if isinstance(sched, GlobalForwardScheduler):
                    await sched.terminate_run(run_id)
                    return {
                        "kind": sched.__class__.__name__,
                        "run_id": run_id,
                        "state": "cancellation_requested",
                    }
                elif isinstance(sched, ForwardScheduler):
                    await sched.terminate()
                    return {
                        "kind": sched.__class__.__name__,
                        "run_id": run_id,
                        "state": "cancellation_requested",
                    }
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger("aethergraph.runtime.run_manager").exception(
                    "Error terminating scheduler for run_id=%s", run_id
                )
                return None

        # No record in store – still try to terminate scheduler, then bail
        if record is None:
            if handle is None:
                handle = await self._ensure_cancellation_handle(run_id)
            await handle.request_cancel(reason=reason)
            if handle.adapter_kind is None:
                await _terminate_scheduler()
            return None

        # If already terminal, don't change status
        if record.status in {
            RunStatus.succeeded,
            RunStatus.failed,
            RunStatus.canceled,
        }:
            return record

        if handle is None:
            handle = await self._ensure_cancellation_handle(run_id)

        was_waiting = record.status == RunStatus.waiting
        # Mark cancellation requested so UI can react immediately
        record.status = RunStatus.cancellation_requested
        if self._store is not None:
            await self._store.update_status(
                run_id,
                RunStatus.cancellation_requested,
                finished_at=None,
                error=None,
                meta_update={k: v for k, v in handle.metadata().items() if v is not None}
                if handle is not None
                else None,
            )

        await handle.request_cancel(reason=reason)
        if handle.adapter_kind is None:
            backend_state = await _terminate_scheduler()
            if backend_state:
                handle.backend_state_value = dict(backend_state)
        else:
            backend_state = await handle.backend_state()
        record.meta.update({k: v for k, v in handle.metadata().items() if v is not None})

        if was_waiting:
            handle.mark_backend_stopped(backend_state=backend_state)
            record.status = RunStatus.canceled
            record.finished_at = _utcnow()
            record.error = "Run cancelled by user"
            record.meta.update({k: v for k, v in handle.metadata().items() if v is not None})
            if self._store is not None:
                await self._store.update_status(
                    run_id,
                    RunStatus.canceled,
                    finished_at=record.finished_at,
                    error=record.error,
                    meta_update={k: v for k, v in handle.metadata().items() if v is not None},
                )
            await self._resolve_run_future(run_id, (record, None))
            await self._get_cancellation_registry().pop(run_id)

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

        Output semantics when return_outputs=True:
        - succeeded: returns durable final graph outputs when available
          (prefer in-memory completion data, then persisted run results, then
          snapshot-based recovery as a fallback)
        - failed / canceled: returns (record, None)

        This keeps wait_run useful across process boundaries instead of limiting
        output retrieval to only in-process waiters.
        """
        # Fast path: already terminal in store
        rec = await self.get_record(run_id)
        if rec and rec.status in {RunStatus.succeeded, RunStatus.failed, RunStatus.canceled}:
            if return_outputs:
                if rec.status == RunStatus.succeeded:
                    return rec, await self._durable_outputs_for_record(rec)
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
            if rec2.status == RunStatus.succeeded and outputs is None:
                outputs = await self._durable_outputs_for_record(rec2)
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
        app_name: str | None = None,
        run_config: dict[str, Any] | None = None,
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
        tenant = registry_tenant_from_identity(identity)
        self._resolve_target_identity = identity
        try:
            target = await self._resolve_target(graph_id)
        finally:
            self._resolve_target_identity = None
        rid = run_id or f"run-{uuid4().hex[:12]}"
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
                meta = (
                    reg.get_meta(
                        nspace="graph",
                        name=graph_id,
                        version=None,
                        tenant=tenant,
                        include_global=True,
                    )
                    or {}
                )
            elif kind == "graphfn":
                meta = (
                    reg.get_meta(
                        nspace="graphfn",
                        name=graph_id,
                        version=None,
                        tenant=tenant,
                        include_global=True,
                    )
                    or {}
                )
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

        record.meta["original_inputs"] = _clone_jsonish_dict(inputs)
        record.meta["original_run_config"] = _clone_jsonish_dict(run_config)
        record.meta["original_tags"] = list(tags or [])
        record.meta["original_session_id"] = session_id
        record.meta["original_app_id"] = app_id
        if app_name:
            record.meta["app_name"] = app_name
        if agent_id:
            record.meta["agent_id"] = agent_id

        resume_from_run_id = run_config.get("resume_from_run_id") if run_config else None
        resume_mode = run_config.get("resume_mode") if run_config else None
        if resume_from_run_id:
            record.meta["resume_from_run_id"] = resume_from_run_id
        if resume_mode:
            record.meta["resume_mode"] = resume_mode

        if self._store is not None:
            await self._store.create(record)
        await self._ensure_cancellation_handle(record.run_id)

        finalize_kwargs = {
            "record": record,
            "target": target,
            "graph_id": graph_id,
            "inputs": inputs,
            "identity": identity,
        }
        if run_config is not None:
            finalize_kwargs["run_config"] = run_config
        return await self._run_and_finalize(
            **finalize_kwargs,
            # agent_id=agent_id,
            # app_id=app_id,
        )
