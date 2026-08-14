from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any, Protocol


@dataclass(frozen=True)
class LogContext:
    run_id: str | None = None
    node_id: str | None = None
    graph_id: str | None = None
    agent_id: str | None = None

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
    def for_node_ctx(
        self, *, run_id: str, node_id: str, graph_id: str | None = None
    ) -> logging.Logger: ...
    def for_run_ctx(self, *, run_id: str, graph_id: str | None = None) -> logging.Logger: ...
