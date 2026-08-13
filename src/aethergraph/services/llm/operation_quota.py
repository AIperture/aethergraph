"""Atomic per-run quota accounting shared by non-Chat model operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
from typing import Any

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.llm.types import (
    ModelOperationRunQuotaExceededError,
    ModelOperationRunQuotaUnverifiableError,
    ModelOperationRunQuotaWouldExceedError,
)

_LEDGER_KEY = "_model_operation_usage_quota_state"
_LOCK_CREATION_GUARD = threading.Lock()
_RLOCK_TYPE = type(threading.RLock())


@dataclass
class _OperationQuotaReservation:
    """Retain one active operation reservation until release or reconciliation."""

    operation: str
    run_id: str
    state: dict[str, Any]
    lock: threading.RLock
    requested: dict[str, int]
    active: bool = True


class OperationQuotaLedger:
    """Reserve and reconcile exact non-Chat model-operation usage atomically."""

    def __init__(self, operation: str, limits: Mapping[str, int | None] | None = None) -> None:
        self.operation = str(operation).strip()
        if not self.operation:
            raise ValueError("model operation quota identity must not be empty")
        self._limits = {
            str(metric): int(limit)
            for metric, limit in dict(limits or {}).items()
            if limit is not None
        }
        if any(limit < 0 for limit in self._limits.values()):
            raise ValueError("model operation quota limits must be non-negative")

    @property
    def enabled(self) -> bool:
        return bool(self._limits)

    def _state(self) -> tuple[str, dict[str, Any]] | None:
        if not self.enabled:
            return None
        context = current_meter_context.get()
        run_id = context.get("run_id")
        if not run_id:
            return None
        with _LOCK_CREATION_GUARD:
            ledger = context.setdefault(_LEDGER_KEY, {})
            if not isinstance(ledger, dict):
                raise TypeError("model operation usage quota ledger is invalid")
            state = ledger.setdefault(
                self.operation,
                {"consumed": {}, "reserved": {}, "_reservation_lock": threading.RLock()},
            )
        if not isinstance(state, dict):
            raise TypeError("model operation usage quota state is invalid")
        return str(run_id), state

    @staticmethod
    def _lock(state: dict[str, Any]) -> threading.RLock:
        lock = state.get("_reservation_lock")
        if lock is None:
            with _LOCK_CREATION_GUARD:
                lock = state.setdefault("_reservation_lock", threading.RLock())
        if not isinstance(lock, _RLOCK_TYPE):
            raise TypeError("model operation usage quota reservation lock is invalid")
        return lock

    @staticmethod
    def _normalize(metrics: Mapping[str, int | None]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for metric, value in metrics.items():
            if value is None:
                continue
            amount = int(value)
            if amount < 0:
                raise ValueError(f"operation quota metric '{metric}' must be non-negative")
            normalized[str(metric)] = amount
        return normalized

    def reserve(
        self,
        requested: Mapping[str, int | None],
    ) -> _OperationQuotaReservation | None:
        quota_state = self._state()
        if quota_state is None:
            return None
        run_id, state = quota_state
        request = self._normalize(requested)
        lock = self._lock(state)
        with lock:
            consumed = state.setdefault("consumed", {})
            reserved = state.setdefault("reserved", {})
            if not isinstance(consumed, dict) or not isinstance(reserved, dict):
                raise TypeError("model operation usage quota counters are invalid")
            for metric, limit in self._limits.items():
                amount = request.get(metric, 0)
                before = int(consumed.get(metric, 0)) + int(reserved.get(metric, 0))
                projected = before + amount
                if projected > limit:
                    raise ModelOperationRunQuotaWouldExceedError(
                        operation=self.operation,
                        run_id=run_id,
                        quota=metric,
                        consumed=before,
                        requested=amount,
                        projected=projected,
                        limit=limit,
                        phase="would be exceeded before provider dispatch",
                    )
            for metric, amount in request.items():
                reserved[metric] = int(reserved.get(metric, 0)) + amount
        return _OperationQuotaReservation(
            operation=self.operation,
            run_id=run_id,
            state=state,
            lock=lock,
            requested=request,
        )

    def release(self, reservation: _OperationQuotaReservation | None) -> None:
        if reservation is None or not reservation.active:
            return
        with reservation.lock:
            if not reservation.active:
                return
            reserved = reservation.state.setdefault("reserved", {})
            if not isinstance(reserved, dict):
                raise TypeError("model operation usage quota reserved counters are invalid")
            for metric, amount in reservation.requested.items():
                reserved[metric] = max(0, int(reserved.get(metric, 0)) - amount)
            reservation.active = False

    def reconcile(
        self,
        reservation: _OperationQuotaReservation | None,
        actual: Mapping[str, int | None],
        *,
        usage: dict[str, Any] | None = None,
    ) -> ModelOperationRunQuotaExceededError | ModelOperationRunQuotaUnverifiableError | None:
        if reservation is None:
            return None
        if reservation.operation != self.operation:
            raise ValueError("model operation quota reservation belongs to another operation")
        actual_usage = self._normalize(actual)
        with reservation.lock:
            self.release(reservation)
            consumed = reservation.state.setdefault("consumed", {})
            reserved = reservation.state.setdefault("reserved", {})
            if not isinstance(consumed, dict) or not isinstance(reserved, dict):
                raise TypeError("model operation usage quota counters are invalid")
            for metric, amount in actual_usage.items():
                consumed[metric] = int(consumed.get(metric, 0)) + amount
            for metric, limit in self._limits.items():
                if metric not in actual_usage:
                    continue
                amount = actual_usage.get(metric, 0)
                projected = int(consumed.get(metric, 0)) + int(reserved.get(metric, 0))
                if projected > limit:
                    return ModelOperationRunQuotaExceededError(
                        operation=self.operation,
                        run_id=reservation.run_id,
                        quota=metric,
                        consumed=projected - amount,
                        requested=amount,
                        projected=projected,
                        limit=limit,
                        phase="was exceeded by actual provider usage",
                        usage=usage,
                    )
            missing_metrics = tuple(metric for metric in self._limits if metric not in actual_usage)
            if missing_metrics:
                return ModelOperationRunQuotaUnverifiableError(
                    operation=self.operation,
                    run_id=reservation.run_id,
                    quotas=missing_metrics,
                    usage=usage,
                )
        return None


def embedding_quota_ledger(settings: Any | None) -> OperationQuotaLedger:
    """Build the shared ledger adapter for one embedding quota policy."""

    return OperationQuotaLedger(
        "embedding",
        {
            "calls": getattr(settings, "max_calls_per_run", None),
            "texts": getattr(settings, "max_texts_per_run", None),
            "input_tokens": getattr(settings, "max_input_tokens_per_run", None),
        },
    )


def image_generation_quota_ledger(settings: Any | None) -> OperationQuotaLedger:
    """Build the shared ledger adapter for one image-generation quota policy."""

    return OperationQuotaLedger(
        "image_generation",
        {
            "calls": getattr(settings, "max_calls_per_run", None),
            "images": getattr(settings, "max_images_per_run", None),
            "input_tokens": getattr(settings, "max_input_tokens_per_run", None),
            "output_tokens": getattr(settings, "max_output_tokens_per_run", None),
            "total_tokens": getattr(settings, "max_total_tokens_per_run", None),
        },
    )
