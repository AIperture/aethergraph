__version__ = "0.1.0a1"

# Server
from .server.start import start_server  # start a local sidecar server
from .server.start import stop_server   # stop the sidecar server
from .server.start import start_server_async  # async version of start_server

# Tools
from .core.tools.toolkit import tool 

# Graphs
from .core.graph.graph_fn import graph_fn  # full-featured graph decorator
from .core.graph.graphify import graphify  # graphify decorator to build TaskGraphs from functions
from .core.graph.task_graph import TaskGraph  # full task graph object for type checking, serialization, etc.

# Runtime
from .core.runtime.node_context import NodeContext # per-node execution context (run_id)
from .core.runtime.base_service import Service  # base service class for custom services

# Channel buttons
from .contracts.services.channel import Button

__all__ = [
    # Server
    "start_server", "stop_server", "start_server_async",
    # Tools
    "tool", "graph_fn","graphify", "TaskGraph",
    "RuntimeEnv", "NodeContext",
    # Services
    "Service",
    # Channel buttons
    "Button",
]