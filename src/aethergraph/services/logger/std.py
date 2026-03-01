from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import logging.handlers
import os
from pathlib import Path
import queue

from aethergraph.config.config import AppSettings
from aethergraph.core.graph.graph_refs import GRAPH_INPUTS_NODE_ID

from .base import LogContext, LoggerService
from .formatters import ColorFormatter, JsonFormatter, SafeFormatter


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LoggingConfig:
    """
    Configure sinks & formats.

    Attributes:
      root_ns: base logger name to use (`aethergraph`).
      level: default level for root logger (what we *emit* at all).
      console_level: threshold for console output; defaults to `level`.
      file_level: threshold for file logs; defaults to `level`.
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
    console_pattern = "%(asctime)s %(levelname)s %(name)s " "run=%(run_id)s - %(message)s"

    file_pattern = "%(asctime)s %(levelname)s %(name)s " "run=%(run_id)s %(message)s"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5

    # per-sink levels
    console_level: str | None = None
    file_level: str | None = None

    # external loggers
    external_level: str = "WARNING"
    quiet_loggers: tuple[str, ...] = ("httpx", "faiss", "faiss.loader")

    @staticmethod
    def from_env() -> LoggingConfig:
        level = os.getenv("AETHERGRAPH_LOG_LEVEL", "INFO")
        return LoggingConfig(
            root_ns=os.getenv("AETHERGRAPH_LOG_ROOT", "aethergraph"),
            level=level,
            log_dir=os.getenv("AETHERGRAPH_LOG_DIR", "./logs"),
            use_json=os.getenv("AETHERGRAPH_LOG_JSON", "0") == "1",
            enable_queue=os.getenv("AETHERGRAPH_LOG_ASYNC", "0") == "1",
            console_level=os.getenv("AETHERGRAPH_LOG_CONSOLE_LEVEL") or None,
            file_level=os.getenv("AETHERGRAPH_LOG_FILE_LEVEL") or None,
        )

    @staticmethod
    def from_cfg(cfg: AppSettings, log_dir: str | None = None) -> LoggingConfig:
        return LoggingConfig(
            root_ns=cfg.logging.nspace or "aethergraph",
            level=cfg.logging.level,
            log_dir=log_dir or "./logs",
            use_json=cfg.logging.json_logs,
            enable_queue=cfg.logging.enable_queue,
            external_level=cfg.logging.external_level,
            quiet_loggers=tuple(cfg.logging.quiet_loggers),
            console_level=cfg.logging.console_level,
            file_level=cfg.logging.file_level,
        )

    def _resolve_console_level(self) -> int:
        lvl = (self.console_level or self.level).upper()
        return getattr(logging, lvl, logging.INFO)

    def _resolve_file_level(self) -> int:
        lvl = (self.file_level or self.level).upper()
        return getattr(logging, lvl, logging.INFO)


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


class HideGraphInputsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # if this is a "graph inputs" pseudo-node, don't show on console
        node_id = getattr(record, "node_id", None)
        return node_id not in (None, "-", "__graph_inputs__")


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

    def for_node(self, node_id: str) -> logging.Logger:
        # Special case: "graph inputs" pseudo-node should be treated as graph-level logger
        if node_id == GRAPH_INPUTS_NODE_ID:
            return self.for_namespace("graph")
        return self.for_namespace(f"node.{node_id}")

    def for_run(self) -> logging.Logger:
        return self.for_namespace("run")

    def for_inspect(self) -> logging.Logger:
        return self.for_namespace("inspect")

    def for_channel(self) -> logging.Logger:
        return self.for_namespace("channel")

    def for_scheduler(self) -> logging.Logger:
        return self.for_namespace("scheduler")

    def for_service(self, ns: str) -> logging.Logger:
        """Service-level logger with no node/graph context."""
        return self.for_namespace(f"service.{ns}")

    def for_service_ctx(
        self,
        ns: str,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> logging.Logger:
        """Service-level logger with only run/agent context."""
        base = self.for_service(ns)
        ctx = LogContext(run_id=run_id, agent_id=agent_id)
        return self.with_context(base, ctx)

    def for_node_ctx(
        self, *, run_id: str, node_id: str, graph_id: str | None = None
    ) -> logging.Logger:
        # Graph-level logs: use "aethergraph.graph.<graph_id>" instead of node.__graph_inputs__
        if node_id == GRAPH_INPUTS_NODE_ID:
            if graph_id:
                base = self.for_namespace(f"graph.{graph_id}")
            else:
                base = self.for_namespace("graph")

            # Don't attach node_id here; treat as pure graph-level context
            return self.with_context(base, LogContext(run_id=run_id, graph_id=graph_id))

        # Normal nodes: keep existing behavior
        base = self.for_node(node_id)
        return self.with_context(
            base, LogContext(run_id=run_id, node_id=node_id, graph_id=graph_id)
        )

    def for_run_ctx(self, *, run_id: str, graph_id: str | None = None) -> logging.Logger:
        base = self.for_run()
        return self.with_context(base, LogContext(run_id=run_id, graph_id=graph_id))

    # --- builder ---

    @staticmethod
    def build(cfg: LoggingConfig | None = None) -> StdLoggerService:
        cfg = cfg or LoggingConfig.from_env()

        root = logging.getLogger(cfg.root_ns)
        # Reset handlers if rebuilding (idempotent server restarts)
        for h in list(root.handlers):
            root.removeHandler(h)

        # Root should usually be DEBUG or `cfg.level`, but since we
        # now tune at handler level, it's safe to set it low:
        root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
        root.propagate = False

        # Ensure key AG namespaces *inherit* from root (no stale WARNING overrides)
        for ns in ("graph", "node", "service", "run", "inspect", "scheduler", "channel"):
            logging.getLogger(f"{cfg.root_ns}.{ns}").setLevel(logging.NOTSET)

        # Per-namespace levels
        if cfg.per_namespace_levels:
            for ns, lvl in cfg.per_namespace_levels.items():
                logging.getLogger(ns).setLevel(getattr(logging, str(lvl).upper(), logging.INFO))

        # Console handler (usually higher threshold)
        console = logging.StreamHandler()
        console.setLevel(cfg._resolve_console_level())
        console.addFilter(HideGraphInputsFilter())
        console.setFormatter(ColorFormatter(cfg.console_pattern))
        root.addHandler(console)

        # File handler (usually lower / same threshold)
        _ensure_dir(Path(cfg.log_dir))
        file_path = Path(cfg.log_dir) / "aethergraph.log"

        if cfg.enable_queue:
            q = queue.Queue(-1)
            qh = logging.handlers.QueueHandler(q)
            root.addHandler(qh)

            fh = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            if cfg.use_json:
                fh.setFormatter(JsonFormatter())
            else:
                fh.setFormatter(SafeFormatter(cfg.file_pattern))
            fh.setLevel(cfg._resolve_file_level())
            listener = logging.handlers.QueueListener(q, fh, respect_handler_level=True)
            listener.daemon = True
            listener.start()
        else:
            fh = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            if cfg.use_json:
                fh.setFormatter(JsonFormatter())
            else:
                fh.setFormatter(SafeFormatter(cfg.file_pattern))
            fh.setLevel(cfg._resolve_file_level())
            root.addHandler(fh)

        ext_level = getattr(logging, cfg.external_level.upper(), logging.WARNING)
        for name in cfg.quiet_loggers:
            lg = logging.getLogger(name)
            lg.setLevel(ext_level)
            lg.propagate = True

        return StdLoggerService(root, cfg=cfg)
