from collections.abc import Callable
from dataclasses import dataclass, field
import logging
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
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.knowledge.node_kb import NodeKB
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.tracing import NoopTracer
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.websearch.facade import WebSearchFacade

from ..graph.task_node import TaskNodeRuntime
from .bound_memory import BoundMemoryAdapter
from .execution_context import ExecutionContext
from .node_services import NodeServices

logger = logging.getLogger(__name__)


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

    # memory override (for testing/demo purposes)
    memory_level_override: str | None = None
    memory_scope_override: str | None = None

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
    def mcp_service(self):
        return self.container.mcp

    @property
    def web_search_service(self):
        return self.container.web_search

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

        node_scope = (
            self.container.scope_factory.for_node(
                identity=self.identity,
                run_id=self.run_id,
                graph_id=self.graph_id,
                node_id=node.node_id,
                session_id=self.session_id,
                app_id=self.app_id,
                agent_id=self.agent_id,
            )
            if self.container.scope_factory
            else None
        )

        level, custom_scope_id = self._resolve_memory_config()
        mem_scope = (
            self.container.scope_factory.for_memory(
                identity=self.identity,
                run_id=self.run_id,
                graph_id=self.graph_id,
                node_id=node.node_id,
                session_id=self.session_id,
                app_id=self.app_id,
                agent_id=self.agent_id,
                level=level,
                custom_scope_id=custom_scope_id,
            )
            if self.container.scope_factory
            else None
        )

        indices: ScopedIndices | None = None  # scoped indices for this node
        if self.container.global_indices is not None and node_scope is not None:
            # Attach scoped indices to container for this node's scope
            # Prefer memory scope id if available for memory-tied corpora
            base_scope = mem_scope or node_scope
            if base_scope:
                scope_id = mem_scope.memory_scope_id() if mem_scope else None
                indices = self.container.global_indices.for_scope(
                    scope=base_scope,
                    scope_id=scope_id,
                )

        mem: MemoryFacade = self.memory_factory.for_session(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=node.node_id,
            session_id=self.session_id,
            scope=mem_scope,
            scoped_indices=indices,
        )

        from aethergraph.services.artifacts.facade import ArtifactFacade

        artifact_facade = ArtifactFacade(
            run_id=self.run_id,
            graph_id=self.graph_id or "",
            node_id=node.node_id,
            tool_name=node.tool_name,
            tool_version=node.tool_version,  # to be filled from node if available
            art_store=self.artifacts,
            art_index=self.artifact_index,
            scoped_indices=indices,
            scope=mem_scope,
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

        kb = NodeKB(
            backend=self.container.kb_backend,
            scope=mem_scope,
        )

        # ----- TriggerFacade tied to this node/run -----
        # trigger_scope = self.container.scope_factory.for_trigger(identity=self.identity)
        trigger_scope = (
            mem_scope  # for now we need trigger to launch runs with the same session id etc
        )
        triggers = TriggerFacade(
            trigger_service=self.container.trigger_service,
            trigger_engine=self.container.trigger_engine,
            scope=trigger_scope,
        )

        web_search = None
        if self.web_search_service is not None:
            web_search = WebSearchFacade(self.web_search_service)

        runner = RunFacade(
            run_manager=self.container.run_manager,
            identity=self.identity,
            session_id=self.session_id,
            agent_id=self.agent_id,
            app_id=self.app_id,
            current_run_id=self.run_id,
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
            mcp=self.mcp_service,  # MCPService
            runner=runner,  # RunFacade
            indices=indices,  # ScopedIndices for this node
            execution=self.container.execution
            if self.container.execution is not None
            else None,  # ExecutionService
            planner_service=self.container.planner_service,
            skills=self.container.skills_registry,
            kb=kb,  # NodeKB
            triggers=triggers,  # TriggerFacade for this node
            web_search=web_search,  # WebSearchFacade or None
            registry=RegistryFacade(
                registry=self.registry,
                scope=mem_scope or node_scope,
                registration_service=getattr(self.container, "registration_service", None),
            ),
            tracer=self.container.tracer or NoopTracer(),
        )
        try:
            from aethergraph.services.harness.overrides import wrap_node_services

            services = wrap_node_services(
                services,
                graph_id=self.graph_id,
                node_id=node.node_id,
            )
        except Exception:
            pass
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
        # Explicit overrides from RuntimeEnv take highest precedence
        if self.memory_level_override:
            return self.memory_level_override, self.memory_scope_override

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

        # print(f"Resolved registry meta for memory config: {meta}")
        if meta:
            # Top-level keys from as_agent/as_app extras
            if "memory" in meta:
                level = meta["memory"].get("level", level)
                custom_scope_id = meta["memory"].get("scope")
        else:
            # If we have an agent_id but no meta, still bias to session-level
            level = "session" if self.agent_id else "run"

        logger.debug(
            f"Resolved memory config: level={level} custom_scope_id={custom_scope_id} from meta={meta}"
        )

        return level, custom_scope_id
