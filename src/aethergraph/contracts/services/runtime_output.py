from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class RuntimeOutputFrame:
    execution_id: str
    run_id: str
    session_id: str | None
    graph_id: str | None
    node_id: str
    tool_name: str | None
    stream: Literal["stdout", "stderr"]
    sequence: int
    text: str
    eof: bool = False
    partial: bool = False
    truncated: bool = False
    source: str = "python.stream"


class RuntimeOutputSink(Protocol):
    def emit(self, frame: RuntimeOutputFrame) -> None: ...

    async def flush_execution(self, execution_id: str) -> None: ...

    async def flush_run(self, run_id: str) -> None: ...
