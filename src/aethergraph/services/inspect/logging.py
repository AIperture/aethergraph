from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any

from aethergraph.core.runtime.runtime_metering import current_meter_context

_STANDARD_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

_SCOPE_KEYS = (
    "run_id",
    "session_id",
    "agent_id",
    "app_id",
    "graph_id",
    "node_id",
    "trace_id",
    "span_id",
    "user_id",
    "org_id",
    "client_id",
)


class RuntimeContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = dict(current_meter_context.get() or {})
        for key in _SCOPE_KEYS:
            if getattr(record, key, None) in (None, "-", "") and ctx.get(key) is not None:
                setattr(record, key, ctx.get(key))
        return True


class EventLogInspectionHandler(logging.Handler):
    def __init__(self, event_log: Any, *, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._event_log = event_log
        self.addFilter(RuntimeContextFilter())

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "inspection_skip", False):
            return
        try:
            row = self._to_row(record)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._event_log.append(row))
            else:
                loop.create_task(self._event_log.append(row))
        except Exception:
            # Never break application logging because inspection logging failed.
            self.handleError(record)

    def _to_row(self, record: logging.LogRecord) -> dict[str, Any]:
        scope = {
            key: getattr(record, key, None) for key in _SCOPE_KEYS if getattr(record, key, None)
        }
        error = None
        if record.exc_info:
            formatter = logging.Formatter()
            error = {
                "type": record.exc_info[0].__name__
                if record.exc_info and record.exc_info[0]
                else None,
                "message": str(record.exc_info[1])
                if record.exc_info and record.exc_info[1]
                else None,
                "detail": formatter.formatException(record.exc_info),
            }
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS
            and key not in _SCOPE_KEYS
            and key != "inspection_skip"
        }
        log_id = f"log_{int(record.created * 1000)}_{abs(hash((record.name, record.msg, record.lineno))) % 10_000_000}"
        payload = {
            "id": log_id,
            "ts": record.created,
            "kind": "inspect_log",
            "summary": record.getMessage(),
            "severity": record.levelname.lower(),
            "status": record.levelname.lower(),
            "producer": {
                "family": "logger",
                "name": record.name,
                "version": None,
            },
            "scope": scope,
            "tags": [record.levelname.lower()],
            "payload": {
                "logger": record.name,
                "level": record.levelname.lower(),
                "message": record.getMessage(),
                "error": error,
                "extra": extra,
            },
        }
        return {
            "id": log_id,
            "ts": record.created,
            "scope_id": scope.get("run_id")
            or (f"session:{scope['session_id']}" if scope.get("session_id") else None)
            or f"logger:{record.name}",
            "kind": "inspect_log",
            "payload": payload,
            "tags": payload["tags"],
            "user_id": scope.get("user_id"),
            "org_id": scope.get("org_id"),
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
