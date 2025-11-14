from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class SafeFormatter(logging.Formatter):
    """
    Text formatter that won't explode if `extra` keys are missing.
    Use %(run_id)s etc. in format strings without having to always bind them.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Provide default values for our known keys so %()s doesn't KeyError
        for k in ("run_id", "node_id", "graph_id", "agent_id"):
            if not hasattr(record, k):
                setattr(record, k, "-")
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    Structured JSON logs; safe for missing extras.
    """

    def __init__(self, *, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.include_timestamp:
            payload["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))
        # standard attrs we care about
        payload.update(
            {
                "run_id": getattr(record, "run_id", None),
                "node_id": getattr(record, "node_id", None),
                "graph_id": getattr(record, "graph_id", None),
                "agent_id": getattr(record, "agent_id", None),
            }
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False)


class ColorFormatter(SafeFormatter):
    """
    Console/file formatter that adds ANSI color only to:
      - level name (INFO/WARNING/ERROR/...)
      - run_id / node_id / graph_id tokens

    Everything else stays uncolored.
    """

    RESET = "\033[0m"
    LEVEL_COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }

    ID_COLOR = "\033[35m"  # magenta for IDs

    def __init__(self, fmt: str, datefmt: str | None = None, use_color: bool | None = None):
        super().__init__(fmt, datefmt=datefmt)
        # auto-disable color if not a TTY unless explicitly forced
        if use_color is None:
            use_color = sys.stderr.isatty()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # First let SafeFormatter fill in missing run_id/node_id/etc
        base = super().format(record)

        if not self.use_color:
            return base

        reset = self.RESET
        level = record.levelname
        level_color = self.LEVEL_COLORS.get(level, "")

        # 1) Color only the level name token (first occurrence)
        if level_color:
            base = base.replace(level, f"{level_color}{level}{reset}", 1)

        # 2) Color run_id / node_id / graph_id tokens like `run=...`, `node=...`, `graph=...`
        id_color = self.ID_COLOR
        for key in ("run_id", "node_id", "graph_id"):
            val = getattr(record, key, None)
            if val and val != "-":
                token = f"{key}={val}"
                colored = f"{id_color}{token}{reset}"
                base = base.replace(token, colored)

        return base
