from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Mapping, Any
import logging


@dataclass(frozen=True)
class LogContext:
    run_id: Optional[str] = None
    node_id: Optional[str] = None
    graph_id: Optional[str] = None
    agent_id: Optional[str] = None

    def as_extra(self) -> Mapping[str, Any]:
        # Only include non-None fields; logging.Formatter will lookup keys by name.
        return {k: v for k, v in self.__dict__.items() if v is not None}


class LoggerService(Protocol):
    """Contract used by the rest of the system (NodeContext, schedulers, etc.)."""

    def base(self) -> logging.Logger: ...
    def for_namespace(self, ns: str) -> logging.Logger: ...
    def with_context(self, logger: logging.Logger, ctx: LogContext) -> logging.Logger: ...

    # Back-compat helpers
    def for_node(self, node_id: str) -> logging.Logger: ...
    def for_run(self) -> logging.Logger: ...
    def for_inspect(self) -> logging.Logger: ...
    def for_scheduler(self) -> logging.Logger: ...
    def for_node_ctx(self, *, run_id: str, node_id: str, graph_id: Optional[str] = None) -> logging.Logger: ...
    def for_run_ctx(self, *, run_id: str, graph_id: Optional[str] = None) -> logging.Logger: ...
