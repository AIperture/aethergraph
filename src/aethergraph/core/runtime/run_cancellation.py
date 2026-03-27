from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event
from typing import Any, Protocol


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class RunCancellationRequestedError(RuntimeError):
    """Raised when cooperative cancellation is requested for the current run."""


class CancellationAdapter(Protocol):
    async def request_cancel(self) -> None: ...

    async def backend_state(self) -> dict[str, Any] | None: ...

    async def wait_stopped(self, timeout_s: float | None = None) -> bool: ...


@dataclass
class RemoteJobCancellationAdapter:
    """Placeholder adapter for future remote/cloud job cancellation backends."""

    remote_job_id: str

    async def request_cancel(self) -> None:
        raise NotImplementedError("Remote job cancellation is not implemented yet.")

    async def backend_state(self) -> dict[str, Any] | None:
        return {"kind": "remote_job", "remote_job_id": self.remote_job_id}

    async def wait_stopped(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        return False


@dataclass
class RunCancellationHandle:
    run_id: str
    cancel_event: Event = field(default_factory=Event)
    requested_at: datetime | None = None
    backend_stop_requested_at: datetime | None = None
    backend_stopped_at: datetime | None = None
    cancel_reason: str | None = None
    adapter_kind: str | None = None
    backend_state_value: dict[str, Any] | None = None
    remote_job_id: str | None = None
    cancel_dispatch_at: datetime | None = None
    cancel_dispatch_status: str | None = None
    backend_terminal_status: str | None = None
    backend_terminal_at: datetime | None = None
    _adapter: CancellationAdapter | None = None
    _backend_stopped: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def register_adapter(
        self, adapter: CancellationAdapter, *, adapter_kind: str | None = None
    ) -> None:
        self._adapter = adapter
        self.adapter_kind = adapter_kind or adapter.__class__.__name__

    def is_cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def raise_if_cancel_requested(self) -> None:
        if self.is_cancel_requested():
            raise RunCancellationRequestedError(f"Run {self.run_id} cancellation requested")

    def thread_cancel_event(self) -> Event:
        return self.cancel_event

    async def request_cancel(self, reason: str = "user_requested") -> None:
        async with self._lock:
            now = _utcnow()
            if self.requested_at is None:
                self.requested_at = now
            self.cancel_reason = reason
            self.cancel_event.set()
            if self._adapter is None:
                if self.adapter_kind is None:
                    self.adapter_kind = "none"
                self.cancel_dispatch_at = now
                self.cancel_dispatch_status = "no_adapter"
                if self.backend_state_value is None:
                    self.backend_state_value = {"kind": "none", "state": "cancel_requested"}
                return

            if self.backend_stop_requested_at is None:
                self.backend_stop_requested_at = now
            self.cancel_dispatch_at = now
            self.cancel_dispatch_status = "requested"
            await self._adapter.request_cancel()
            state = await self._adapter.backend_state()
            if state:
                self.backend_state_value = dict(state)
                self.remote_job_id = str(state.get("remote_job_id") or self.remote_job_id or "")
                if not self.remote_job_id:
                    self.remote_job_id = None

    async def backend_state(self) -> dict[str, Any] | None:
        if self._adapter is None:
            return self.backend_state_value
        state = await self._adapter.backend_state()
        if state:
            self.backend_state_value = dict(state)
        return self.backend_state_value

    def mark_backend_stopped(
        self,
        *,
        backend_state: dict[str, Any] | None = None,
        terminal_status: str = "canceled",
    ) -> None:
        now = _utcnow()
        self._backend_stopped = True
        self.backend_stopped_at = now
        self.backend_terminal_status = terminal_status
        self.backend_terminal_at = now
        if backend_state:
            self.backend_state_value = dict(backend_state)

    def backend_has_stopped(self) -> bool:
        return self._backend_stopped

    async def wait_stopped(self, timeout_s: float | None = None) -> bool:
        if self._backend_stopped:
            return True
        if self._adapter is None:
            return False
        stopped = await self._adapter.wait_stopped(timeout_s=timeout_s)
        if stopped and not self._backend_stopped:
            state = await self._adapter.backend_state()
            self.mark_backend_stopped(backend_state=state)
        return self._backend_stopped

    def metadata(self) -> dict[str, Any]:
        return {
            "cancel_requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "cancel_finalized_at": self.backend_stopped_at.isoformat()
            if self.backend_stopped_at
            else None,
            "cancel_reason": self.cancel_reason,
            "cancel_backend_kind": self.adapter_kind,
            "cancel_backend_state": self.backend_state_value,
            "remote_job_id": self.remote_job_id,
            "cancel_dispatch_at": self.cancel_dispatch_at.isoformat()
            if self.cancel_dispatch_at
            else None,
            "cancel_dispatch_status": self.cancel_dispatch_status,
            "backend_terminal_status": self.backend_terminal_status,
            "backend_terminal_at": self.backend_terminal_at.isoformat()
            if self.backend_terminal_at
            else None,
        }


@dataclass
class RunCancellationRegistry:
    _handles: dict[str, RunCancellationHandle] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def register(self, handle: RunCancellationHandle) -> RunCancellationHandle:
        async with self._lock:
            self._handles[handle.run_id] = handle
            return handle

    async def create(self, run_id: str) -> RunCancellationHandle:
        async with self._lock:
            handle = self._handles.get(run_id)
            if handle is None:
                handle = RunCancellationHandle(run_id=run_id)
                self._handles[run_id] = handle
            return handle

    async def get(self, run_id: str) -> RunCancellationHandle | None:
        async with self._lock:
            return self._handles.get(run_id)

    async def pop(self, run_id: str) -> RunCancellationHandle | None:
        async with self._lock:
            return self._handles.pop(run_id, None)


class LocalSchedulerCancellationAdapter:
    def __init__(self, scheduler: Any, *, run_id: str | None = None):
        self._scheduler = scheduler
        self._run_id = run_id

    async def request_cancel(self) -> None:
        if self._run_id is not None and hasattr(self._scheduler, "terminate_run"):
            await self._scheduler.terminate_run(self._run_id)
            return
        if hasattr(self._scheduler, "terminate"):
            await self._scheduler.terminate()

    async def backend_state(self) -> dict[str, Any] | None:
        return {
            "kind": self._scheduler.__class__.__name__,
            "run_id": self._run_id,
            "state": "cancellation_requested",
        }

    async def wait_stopped(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        return False


_GLOBAL_RUN_CANCELLATION_REGISTRY = RunCancellationRegistry()


def get_run_cancellation_registry(container: Any | None = None) -> RunCancellationRegistry:
    if container is not None:
        reg = getattr(container, "run_cancellation_registry", None)
        if reg is not None:
            return reg
    try:
        from aethergraph.core.runtime.runtime_services import current_services

        reg = getattr(current_services(), "run_cancellation_registry", None)
        if reg is not None:
            return reg
    except Exception:
        pass
    return _GLOBAL_RUN_CANCELLATION_REGISTRY
