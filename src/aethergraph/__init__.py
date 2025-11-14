__version__ = "0.1.0a1"

# Server
# Channel buttons
from .contracts.services.channel import Button

# Graphs
from .core.graph.graph_fn import graph_fn  # full-featured graph decorator
from .core.graph.graphify import graphify  # graphify decorator to build TaskGraphs from functions
from .core.graph.task_graph import (
    TaskGraph,  # full task graph object for type checking, serialization, etc.
)
from .core.runtime.base_service import Service  # base service class for custom services

# Runtime
from .core.runtime.node_context import NodeContext  # per-node execution context (run_id)

# Tools
from .core.tools.toolkit import tool
from .server.start import (
    start_server,  # start a local sidecar server
    start_server_async,  # async version of start_server
    stop_server,  # stop the sidecar server
)

__all__ = [
    # Server
    "start_server",
    "stop_server",
    "start_server_async",
    # Tools
    "tool",
    "graph_fn",
    "graphify",
    "TaskGraph",
    "RuntimeEnv",
    "NodeContext",
    # Services
    "Service",
    # Channel buttons
    "Button",
]
