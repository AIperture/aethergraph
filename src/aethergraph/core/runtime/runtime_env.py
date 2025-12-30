from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex

# ---- artifact services ----
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore

# ---- channel services ----
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.container.default_container import DefaultContainer, get_container
from aethergraph.services.continuations.stores.fs_store import (
    FSContinuationStore,  # AsyncContinuationStore
)

# ---- memory services ----
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.rag.node_rag import NodeRAG
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry

from ..graph.task_node import TaskNodeRuntime
from .bound_memory import BoundMemoryAdapter
from .execution_context import ExecutionContext
from .node_services import NodeServices


@dataclass
class RuntimeEnv:
    """Unified runtime env that is built from DefaultContainer and can spawn NodeContexts."""

    run_id: str
    graph_id: str | None = None
    session_id: str | None = None
    identity: RequestIdentity | None = None
    graph_inputs: dict[str, Any] = field(default_factory=dict)
    outputs_by_node: dict[str, dict[str, Any]] = field(default_factory=dict)

    # agent and app ids
    agent_id: str | None = None  # for agent-invoked runs
    app_id: str | None = None  # for app-invoked runs

    # container (DI)
    container: DefaultContainer = field(default_factory=get_container)

    # optional predicate to skip execution
    should_run_fn: Callable[[], bool] | None = None

    # --- convenience projections of commonly used services ---
    @property
    def schedulers(self) -> dict[str, Any]:
        return self.container.schedulers

    @property
    def registry(self):
        return self.container.registry

    @property
    def logger_factory(self):
        return self.container.logger

    @property
    def clock(self) -> SystemClock:
        return self.container.clock

    @property
    def channels(self) -> ChannelBus:
        return self.container.channels

    @property
    def continuation_store(self) -> FSContinuationStore:
        return self.container.cont_store

    @property
    def wait_registry(self) -> WaitRegistry:
        return self.container.wait_registry

    @property
    def artifacts(self) -> AsyncArtifactStore:
        return self.container.artifacts

    @property
    def artifact_index(self) -> AsyncArtifactIndex:
        return self.container.artifact_index

    @property
    def memory_factory(self):
        return self.container.memory_factory

    @property
    def llm_service(self):
        return self.container.llm

    @property
    def rag_facade(self):
        return self.container.rag

    @property
    def mcp_service(self):
        return self.container.mcp

    @property
    def resume_router(self) -> ResumeRouter:
        return self.container.resume_router

    def make_ctx(
        self, *, node: "TaskNodeRuntime", resume_payload: dict[str, Any] | None = None
    ) -> Any:
        defaults = {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "node_id": node.node_id,
            "tags": [],
            "entities": [],
        }

        level, custom_scope_id = self._resolve_memory_config()
        mem_scope = (
            self.container.scope_factory.for_memory(
                identity=self.identity,
                run_id=self.run_id,
                graph_id=self.graph_id,
                node_id=node.node_id,
                session_id=self.session_id,
                level=level,
                custom_scope_id=custom_scope_id,
            )
            if self.container.scope_factory
            else None
        )

        mem: MemoryFacade = self.memory_factory.for_session(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=node.node_id,
            session_id=self.session_id,
            scope=mem_scope,
        )

        node_scope = (
            self.container.scope_factory.for_node(
                identity=self.identity,
                run_id=self.run_id,
                graph_id=self.graph_id,
                node_id=node.node_id,
                session_id=self.session_id,
            )
            if self.container.scope_factory
            else None
        )

        from aethergraph.services.artifacts.facade import ArtifactFacade

        artifact_facade = ArtifactFacade(
            run_id=self.run_id,
            graph_id=self.graph_id or "",
            node_id=node.node_id,
            tool_name=node.tool_name,
            tool_version=node.tool_version,  # to be filled from node if available
            store=self.artifacts,
            index=self.artifact_index,
            scope=node_scope,
        )

        # ------- Viz Service tied to this node/run -------'
        vis_facade = VizFacade(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=node.node_id,
            tool_name=node.tool_name,
            tool_version=node.tool_version,
            artifacts=artifact_facade,
            viz_service=self.container.viz_service,
            scope=node_scope,
        )

        # ------- RAG Facade in Memory tied to this node/run -------'
        rag_for_node = None
        if self.rag_facade is not None and node_scope is not None:
            rag_for_node = NodeRAG(
                rag=self.rag_facade,
                scope=node_scope,
                default_scope_id=(mem_scope.memory_scope_id() if mem_scope else None),
            )

        services = NodeServices(
            channels=self.channels,
            continuation_store=self.continuation_store,
            artifact_store=artifact_facade,
            wait_registry=self.wait_registry,
            clock=self.clock,
            logger=self.logger_factory,
            kv=self.container.kv_hot,  # keep using hot kv for ephemeral
            memory=self.memory_factory,  # factory (for other sessions if needed)
            memory_facade=mem,  # bound memory for this run/node
            viz=vis_facade,
            llm=self.llm_service,  # LLMService
            rag=rag_for_node,  # RAGService
            mcp=self.mcp_service,  # MCPService
            run_manager=self.container.run_manager,  # RunManager
        )
        return ExecutionContext(
            run_id=self.run_id,
            session_id=self.session_id,
            identity=self.identity,
            graph_id=self.graph_id,
            agent_id=self.agent_id,
            app_id=self.app_id,
            graph_inputs=self.graph_inputs,
            outputs_by_node=self.outputs_by_node,
            services=services,
            logger_factory=self.logger_factory,
            clock=self.clock,
            resume_payload=resume_payload,
            should_run_fn=self.should_run_fn,
            scope=node_scope,
            # Back-compat shim for old ctx.mem()
            bound_memory=BoundMemoryAdapter(mem, defaults),
            resume_router=self.resume_router,
        )

    def _resolve_memory_config(self) -> tuple[str, str | None]:
        """
        Returns (level, custom_scope_id).

        Resolution order:
        1) If this run has an agent_id, read from the agent registry meta.
        2) Else if this run has an app_id, read from the app registry meta.
        3) Else fall back to graph/graphfn meta.
        4) Defaults:
           - agent/app-backed runs -> "session"
           - plain graph runs      -> "run"
        """
        registry = self.registry
        level: str = "session"  # safe default
        custom_scope_id: str | None = None
        meta: dict[str, Any] = {}

        if registry:
            # Prefer agent meta
            if self.agent_id:
                meta = (
                    registry.get_meta(
                        nspace="agent",
                        name=self.agent_id,
                        version=None,
                    )
                    or {}
                )
            # Then app meta
            elif self.app_id:
                meta = (
                    registry.get_meta(
                        nspace="app",
                        name=self.app_id,
                        version=None,
                    )
                    or {}
                )
            # Finally, bare graph meta (graphfn or taskgraph)
            elif self.graph_id:
                meta = (
                    registry.get_meta("graphfn", self.graph_id, None)
                    or registry.get_meta("graph", self.graph_id, None)
                    or {}
                )

        if meta:
            # Top-level keys from as_agent/as_app extras
            if "memory_level" in meta:
                level = meta["memory_level"]
            else:
                # Fallback by kind if not explicitly set
                kind = meta.get("kind")
                level = "session" if kind == "agent" else "run"

            custom_scope_id = meta.get("memory_scope_id")
        else:
            # If we have an agent_id but no meta, still bias to session-level
            level = "session" if self.agent_id else "run"

        return level, custom_scope_id
