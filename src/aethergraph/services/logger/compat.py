from __future__ import annotations
import logging
from typing import Optional
from .std import StdLoggerService, LoggingConfig
from .base import LogContext

"""For backward compatibility with v2 LoggerFactory interface. Delegates to StdLoggerService under the hood. Will deprecate in future."""

def build_logging(force: bool = True, cfg: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Back-compat entry point. Returns the base logger (as before), but created via StdLoggerService.
    `force` is kept for signature parity; handlers are already reset in StdLoggerService.build().
    """
    svc = StdLoggerService.build(cfg)
    return svc.base()


class LoggerFactory:
    """
    Back-compat API that delegates to StdLoggerService.
    """
    def __init__(self, base: Optional[logging.Logger] = None, *, cfg: Optional[LoggingConfig] = None):
        self._svc = StdLoggerService.build(cfg) if base is None else StdLoggerService(base, cfg=cfg or LoggingConfig.from_env())
        self.base = self._svc.base()  # keep original attr for callers that relied on it

    def for_node(self, node_id: str) -> logging.Logger:
        return self._svc.for_node(node_id)

    def for_inspect(self) -> logging.Logger:
        return self._svc.for_inspect()

    def for_run(self) -> logging.Logger:
        return self._svc.for_run()

    def for_scheduler(self) -> logging.Logger:
        return self._svc.for_scheduler()

    def for_node_ctx(self, *, run_id: str, node_id: str, graph_id: str | None = None) -> logging.Logger:
        return self._svc.for_node_ctx(run_id=run_id, node_id=node_id, graph_id=graph_id)

    def for_run_ctx(self, *, run_id: str, graph_id: str | None = None) -> logging.Logger:
        return self._svc.for_run_ctx(run_id=run_id, graph_id=graph_id)
