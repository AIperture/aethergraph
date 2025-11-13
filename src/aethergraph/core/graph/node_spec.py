from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

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
    status: str     # one of NodeStatus 
    outputs: Dict[str, Any]
    timestamp: float # event time (time.time()) 

@dataclass
class TaskNodeSpec:
    node_id: str
    type: str | NodeType                     # one of NodeType
    logic: str | callable | None = None 
    dependencies: list[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)  # static inputs

    expected_input_keys: List[str] = field(default_factory=list)
    expected_output_keys: List[str] = field(default_factory=lambda: ["result"])
    output_keys: List[str] = field(default_factory=lambda: ["result"])

    # Allowed if it's *static* condition -- NOT IMPLEMENTED YET
    condition: Union[bool, Dict[str, Any], callable[[Dict[str, Any]], bool]] = True

    metadata: Dict[str, Any] = field(default_factory=dict)
    reads: Set[str] = field(default_factory=set)   # state keys to read
    writes: Set[str] = field(default_factory=set)  # state keys to write

    tool_name: Optional[str] = None  # used for logging/monitoring
    tool_version: Optional[str] = None # used for logging/monitoring


