from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from aethergraph.services.container.default_container import DefaultContainer, get_container 

from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.logger.std import LoggingConfig, StdLoggerService
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore    # AsyncContinuationStore
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.resume.multi_scheduler_resume_bus import MultiSchedulerResumeBus 
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.services.waits.wait_registry import WaitRegistry

from aethergraph.services.kv.ephemeral import EphemeralKV
from aethergraph.services.kv.sqlite_kv import SQLiteKV 

# ---- artifact services ----
from aethergraph.services.artifacts.fs_store import FSArtifactStore # AsyncArtifactStore 
from aethergraph.services.artifacts.jsonl_index import JsonlArtifactIndex # AsyncArtifactIndex

# ---- channel services ----
from aethergraph.services.channel.factory import make_channel_adapters_from_env, build_bus
from aethergraph.services.channel.channel_bus import ChannelBus 

# ---- memory services ----
from aethergraph.services.memory.facade import MemoryFacade

from ..graph.task_node  import TaskNodeRuntime
from .node_services import NodeServices
from .bound_memory import BoundMemoryAdapter
from .execution_context import ExecutionContext


@dataclass
class RuntimeEnv:
    """Unified runtime env that is built from DefaultContainer and can spawn NodeContexts."""
    run_id: str
    graph_id: Optional[str] = None
    graph_inputs: Dict[str, Any] = field(default_factory=dict)
    outputs_by_node: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # container (DI)
    container: DefaultContainer = field(default_factory=get_container)

    # optional predicate to skip execution
    should_run_fn: Optional[Callable[[], bool]] = None

    # --- convenience projections of commonly used services ---
    @property
    def schedulers(self) -> Dict[str, Any]:   return self.container.schedulers
    @property
    def registry(self):          return self.container.registry
    @property
    def logger_factory(self):    return self.container.logger
    @property
    def clock(self) -> SystemClock:     return self.container.clock
    @property
    def channels(self) -> ChannelBus:   return self.container.channels
    @property
    def continuation_store(self) -> FSContinuationStore: return self.container.cont_store
    @property
    def wait_registry(self) -> WaitRegistry:  return self.container.wait_registry
    @property
    def artifacts(self) -> FSArtifactStore:   return self.container.artifacts
    @property
    def artifact_index(self) -> JsonlArtifactIndex: return self.container.artifact_index
    @property
    def memory_factory(self):    return self.container.memory_factory
    @property
    def llm_service(self):    return self.container.llm
    @property
    def rag_facade(self):    return self.container.rag
    @property
    def mcp_service(self):    return self.container.mcp
    @property
    def resume_router(self) -> ResumeRouter: return self.container.resume_router

    def make_ctx(self, *, node: "TaskNodeRuntime", resume_payload: Optional[Dict[str, Any]] = None) -> Any:

        defaults = {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "node_id": node.node_id,
            "agent_id": getattr(node, "tool_name", None),
            "tags": [], "entities": [],
        }
        mem: MemoryFacade = self.memory_factory.for_session(
            run_id=self.run_id, graph_id=self.graph_id,
            node_id=node.node_id, agent_id=defaults["agent_id"]
        )

        from aethergraph.services.artifacts.facade import ArtifactFacade
        artifact_facade = ArtifactFacade(
            run_id=self.run_id, graph_id=self.graph_id or "",
            node_id=node.node_id, tool_name=node.tool_name, tool_version=node.tool_version, # to be filled from node if available
            store=self.artifacts, index=self.artifact_index
        )

        services = NodeServices(
            channels=self.channels,
            continuation_store=self.continuation_store,
            artifact_store=artifact_facade,
            wait_registry=self.wait_registry,
            clock=self.clock,
            logger=self.logger_factory,
            kv=self.container.kv_hot,                   # keep using hot kv for ephemeral
            memory=self.memory_factory,                  # factory (for other sessions if needed)
            memory_facade=mem,                           # bound memory for this run/node
            llm=self.llm_service,                           # LLMService
            rag=self.rag_facade,                           # RAGService
            mcp=self.mcp_service,                        # MCPService
        )
        return ExecutionContext(
            run_id=self.run_id, graph_id=self.graph_id, graph_inputs=self.graph_inputs,
            outputs_by_node=self.outputs_by_node, services=services,
            logger_factory=self.logger_factory, clock=self.clock,
            resume_payload=resume_payload, should_run_fn=self.should_run_fn,
            # Back-compat shim for old ctx.mem()
            bound_memory=BoundMemoryAdapter(mem, defaults),
            resume_router=self.resume_router,
        )
