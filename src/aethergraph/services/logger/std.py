from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import logging, queue
import logging.handlers

from typing import Optional, Mapping, Any

from aethergraph.config.config import AppSettings

from .base import LoggerService, LogContext
from .formatters import SafeFormatter, JsonFormatter, ColorFormatter


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LoggingConfig:
    """
    Configure sinks & formats.

    Attributes:
      root_ns: base logger name to use (`aethergraph`).
      level: default level for root logger.
      log_dir: directory for file logs (rotated).
      use_json: True => JSON logs for files; console stays text by default.
      enable_queue: True => offload file IO via QueueHandler/Listener (non-blocking).
      per_namespace_levels: optional map (e.g. {"aethergraph.node": "DEBUG"}).
      console_pattern: text format string for console.
      file_pattern: text format string for file when use_json=False.
      max_bytes / backup_count: rotation for file handlers.
    """
    root_ns: str = "aethergraph"
    level: str = "INFO"
    log_dir: str = "./logs"
    use_json: bool = False
    enable_queue: bool = False
    per_namespace_levels: Mapping[str, str] = None
    console_pattern: str = "%(asctime)s %(levelname)s \t%(name)s    run=%(run_id)s    node=%(node_id)s - %(message)s"
    file_pattern: str = "%(asctime)s %(levelname)s %(name)s %(run_id)s %(node_id)s %(graph_id)s %(message)s"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5

    @staticmethod
    def from_env() -> "LoggingConfig":
        return LoggingConfig(
            root_ns=os.getenv("AETHERGRAPH_LOG_ROOT", "aethergraph"),
            level=os.getenv("AETHERGRAPH_LOG_LEVEL", "INFO"),
            log_dir=os.getenv("AETHERGRAPH_LOG_DIR", "./logs"),
            use_json=os.getenv("AETHERGRAPH_LOG_JSON", "0") == "1",
            enable_queue=os.getenv("AETHERGRAPH_LOG_ASYNC", "0") == "1",
        )
    
    @staticmethod
    def from_cfg(cfg: AppSettings, log_dir: Optional[str] = None) -> "LoggingConfig":
        return LoggingConfig(
            root_ns="aethergraph",
            level=cfg.logging.level,
            log_dir=log_dir or "./logs",
            use_json=cfg.logging.json_logs,
            enable_queue=True,
        )


class _ContextAdapter(logging.LoggerAdapter):
    """
    Injects contextual fields into LogRecord via `extra`.
    Preserves original logger API (info, debug, etc.).
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra") or {}
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


class StdLoggerService(LoggerService):
    """
      • text/JSON formatters
      • per-namespace levels
      • optional async file IO via QueueHandler
      • context helpers (with_context / for_*_ctx)
    """
    def __init__(self, base: logging.Logger, *, cfg: LoggingConfig):
        self._base = base
        self._cfg = cfg

    # --- LoggerService interface ---

    def base(self) -> logging.Logger:
        return self._base

    def for_namespace(self, ns: str) -> logging.Logger:
        return self._base.getChild(ns)

    def with_context(self, logger: logging.Logger, ctx: LogContext) -> logging.Logger:
        return _ContextAdapter(logger, ctx.as_extra())

    # Back-compat helpers
    def for_node(self, node_id: str) -> logging.Logger:
        return self.for_namespace(f"node.{node_id}")

    def for_run(self) -> logging.Logger:
        return self.for_namespace("run")

    def for_inspect(self) -> logging.Logger:
        return self.for_namespace("inspect")

    def for_scheduler(self) -> logging.Logger:
        return self.for_namespace("scheduler")

    def for_node_ctx(self, *, run_id: str, node_id: str, graph_id: Optional[str] = None) -> logging.Logger:
        base = self.for_node(node_id)
        return self.with_context(base, LogContext(run_id=run_id, node_id=node_id, graph_id=graph_id))

    def for_run_ctx(self, *, run_id: str, graph_id: Optional[str] = None) -> logging.Logger:
        base = self.for_run()
        return self.with_context(base, LogContext(run_id=run_id, graph_id=graph_id))

    # --- builder ---

    @staticmethod
    def build(cfg: Optional[LoggingConfig] = None) -> "StdLoggerService":
        cfg = cfg or LoggingConfig.from_env()

        root = logging.getLogger(cfg.root_ns)
        # Reset handlers if rebuilding (idempotent server restarts)
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
        root.propagate = False

        # Per-namespace levels
        if cfg.per_namespace_levels:
            for ns, lvl in cfg.per_namespace_levels.items():
                logging.getLogger(ns).setLevel(getattr(logging, str(lvl).upper(), logging.INFO))

        # Console handler (text)
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
        console.setFormatter(ColorFormatter(cfg.console_pattern))
        root.addHandler(console)

        # File handler (rotating)
        _ensure_dir(Path(cfg.log_dir))
        file_path = Path(cfg.log_dir) / "aethergraph.log"

        if cfg.enable_queue:
            # Non-blocking file IO
            q = queue.Queue(-1)
            qh = logging.handlers.QueueHandler(q)
            root.addHandler(qh)

            fh = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8"
            )
            if cfg.use_json:
                fh.setFormatter(JsonFormatter())
            else:
                fh.setFormatter(SafeFormatter(cfg.file_pattern))
            fh.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
            listener = logging.handlers.QueueListener(q, fh, respect_handler_level=True)
            listener.daemon = True
            listener.start()
        else:
            fh = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8"
            )
            if cfg.use_json:
                fh.setFormatter(JsonFormatter())
            else:
                fh.setFormatter(SafeFormatter(cfg.file_pattern))
            fh.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
            root.addHandler(fh)

        return StdLoggerService(root, cfg=cfg)

