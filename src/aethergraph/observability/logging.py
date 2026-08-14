from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
from typing import Any

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.server.security.redaction import sanitize_content

from .models import ObservationRecord, ObservationScope

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

_SCOPE_KEYS = tuple(ObservationScope.__dataclass_fields__)


class RuntimeContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = dict(current_meter_context.get() or {})
        for key in _SCOPE_KEYS:
            if getattr(record, key, None) in (None, "-", "") and context.get(key) is not None:
                setattr(record, key, context[key])
        return True


class ObservationLogHandler(logging.Handler):
    def __init__(self, observation_store: Any, *, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._store = observation_store
        self.addFilter(RuntimeContextFilter())

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "observation_skip", False):
            return
        try:
            observation = self._to_observation(record)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._store.append_observation(observation))
            else:
                loop.create_task(self._store.append_observation(observation))
        except Exception:
            self.handleError(record)

    @staticmethod
    def _to_observation(record: logging.LogRecord) -> ObservationRecord:
        scope = ObservationScope(
            **{key: getattr(record, key, None) for key in _SCOPE_KEYS if getattr(record, key, None)}
        )
        error = None
        if record.exc_info:
            formatter = logging.Formatter()
            error = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "detail": formatter.formatException(record.exc_info),
            }
        extra = {
            key: sanitize_content(value)
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS
            and key not in _SCOPE_KEYS
            and key != "observation_skip"
        }
        level = record.levelname.lower()
        return ObservationRecord(
            category="log",
            name=record.name,
            summary=record.getMessage(),
            occurred_at=datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            status="error" if record.levelno >= logging.ERROR else "ok",
            severity=level,
            scope=scope,
            attributes={
                "logger": record.name,
                "level": level,
                "message": record.getMessage(),
                "error": error,
                "extra": extra,
            },
        )
