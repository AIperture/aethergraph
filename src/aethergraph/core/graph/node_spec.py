from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    TOOL = "tool"
    LLM = "llm"
    HUMAN = "human"
    ROBOT = "robot"
    CUSTOM = "custom"


@dataclass
class NodeEvent:
    run_id: str
    graph_id: str
    node_id: str
    status: str  # one of NodeStatus
    outputs: dict[str, Any]
    timestamp: float  # event time (time.time())


@dataclass
class TaskNodeSpec:
    node_id: str
    type: str | NodeType  # one of NodeType
    logic: str | callable | None = None
    dependencies: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)  # static inputs

    expected_input_keys: list[str] = field(default_factory=list)
    expected_output_keys: list[str] = field(default_factory=lambda: ["result"])
    output_keys: list[str] = field(default_factory=lambda: ["result"])

    # Allowed if it's *static* condition -- NOT IMPLEMENTED YET
    condition: bool | dict[str, Any] | callable[[dict[str, Any]], bool] = True

    metadata: dict[str, Any] = field(default_factory=dict)
    reads: set[str] = field(default_factory=set)  # state keys to read
    writes: set[str] = field(default_factory=set)  # state keys to write

    tool_name: str | None = None  # used for logging/monitoring
    tool_version: str | None = None  # used for logging/monitoring
