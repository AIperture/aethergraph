from collections.abc import Callable
from dataclasses import dataclass, field, replace
import logging
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import OriginBinding

# ---- channel services ----
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.container.default_container import DefaultContainer, get_container

# ---- memory services ----
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.storage.contracts import StorageScope

from ..graph.task_node import TaskNodeRuntime
from .execution_context import ExecutionContext
from .node_context import NodeContext
from .node_services import NodeServices

logger = logging.getLogger(__name__)


def _canonical_scope(scope: Any) -> StorageScope:
    return StorageScope(
        org_id=getattr(scope, "org_id", None),
        user_id=getattr(scope, "user_id", None),
        session_id=getattr(scope, "session_id", None),
        run_id=getattr(scope, "run_id", None),
        graph_id=getattr(scope, "graph_id", None),
        node_id=getattr(scope, "node_id", None),
        agent_id=getattr(scope, "agent_id", None),
    )


def _canonical_memory_scope(scope: Any) -> StorageScope:
    common = {
        "org_id": getattr(scope, "org_id", None),
        "user_id": getattr(scope, "user_id", None),
    }
    custom_scope = getattr(scope, "_memory_scope_id", None)
    if custom_scope:
        return StorageScope(**common, scope_key=custom_scope)
    level = getattr(scope, "memory_level", None)
    if level == "session":
        return StorageScope(**common, session_id=getattr(scope, "session_id", None))
    if level == "run":
        return StorageScope(**common, run_id=getattr(scope, "run_id", None))
    if level == "user":
        return StorageScope(**common)
    if level == "org":
        return StorageScope(org_id=getattr(scope, "org_id", None))
    if level == "scope":
        return StorageScope()
    if getattr(scope, "session_id", None):
        return StorageScope(**common, session_id=scope.session_id)
    if getattr(scope, "user_id", None):
        return StorageScope(**common)
    if getattr(scope, "run_id", None):
        return StorageScope(**common, run_id=scope.run_id)
    if getattr(scope, "org_id", None):
        return StorageScope(org_id=scope.org_id)
    return StorageScope()


@dataclass
class RuntimeEnv:
    """Unified runtime env that is built from DefaultContainer and can spawn NodeContexts."""

    run_id: str
    graph_id: str | None = None
    session_id: str | None = None
    origin_binding: OriginBinding | None = None
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
    def continuation_store(self) -> Any:
        return self.container.cont_store

    @property
    def wait_registry(self) -> WaitRegistry:
        return self.container.wait_registry

    @property
    def memory_factory(self):
        return self.container.memory_factory

    @property
    def llm_service(self):
        return self.container.llm

    @property
    def embedding_service(self):
        return self.container.embed_service

    @property
    def image_model_service(self):
        return self.container.image_service

    @property
    def resume_router(self) -> ResumeRouter:
        return self.container.resume_router

    def make_ctx(
        self, *, node: "TaskNodeRuntime", resume_payload: dict[str, Any] | None = None
    ) -> Any:
        # defaults = {
        #     "run_id": self.run_id,
        #     "graph_id": self.graph_id,
        #     "node_id": node.node_id,
        #     "tags": [],
        #     "entities": [],
        # }

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

        if mem_scope is None or node_scope is None:
            raise RuntimeError("RuntimeEnv requires a scope factory for canonical storage")
        memory_storage_scope = _canonical_memory_scope(mem_scope)
        memory_provenance_scope = _canonical_scope(mem_scope)
        node_storage_scope = _canonical_scope(node_scope)
        mem = self.memory_factory.for_public_execution(
            memory_storage_scope,
            logical_scope_id=mem_scope.memory_scope_id(),
            provenance_scope=memory_provenance_scope,
            deprecated_app_id=self.app_id,
            projection_logger=self.logger_factory.for_node_ctx(
                run_id=self.run_id,
                node_id=node.node_id,
                graph_id=self.graph_id,
            ),
        )

        artifact_facade = self.container.artifact_factory.for_public_execution(
            node_storage_scope,
            tool_name=node.tool_name,
            tool_version=node.tool_version,
            deprecated_app_id=self.app_id,
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

        runner = RunFacade(
            run_manager=self.container.run_manager,
            identity=self.identity,
            session_id=self.session_id,
            agent_id=self.agent_id,
            app_id=self.app_id,
            current_run_id=self.run_id,
            origin_binding=self.origin_binding,
        )

        services = NodeServices(
            channels=self.channels,
            continuation_store=self.continuation_store,
            artifact_store=artifact_facade,
            wait_registry=self.wait_registry,
            clock=self.clock,
            logger=self.logger_factory,
            kv=self.container.kv,
            memory=self.memory_factory,  # factory (for other sessions if needed)
            memory_facade=mem,  # bound memory for this run/node
            agent_state=self.container.storage_services.agent_state(node_storage_scope),
            viz=vis_facade,
            llm=self.llm_service,  # LLMService
            embedding=self.embedding_service,  # EmbeddingService
            image_model=self.image_model_service,  # ImageGenerationService
            runner=runner,  # RunFacade
            triggers=triggers,  # TriggerFacade for this node
            registry=RegistryFacade(
                registry=self.registry,
                scope=mem_scope or node_scope,
                registration_service=getattr(self.container, "registration_service", None),
            ),
            agent_context_factory=lambda context, *, agent_id: self._make_agent_context(
                context,
                node=node,
                agent_id=agent_id,
            ),
        )
        return ExecutionContext(
            run_id=self.run_id,
            session_id=self.session_id,
            origin_binding=self.origin_binding,
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
            resume_router=self.resume_router,
            runtime_output_sink=getattr(self.container, "runtime_output_sink", None),
        )

    def _make_agent_context(
        self,
        context: NodeContext,
        *,
        node: TaskNodeRuntime,
        agent_id: str,
    ) -> NodeContext:
        derived_env = replace(self, agent_id=agent_id)
        derived_execution = derived_env.make_ctx(
            node=node,
            resume_payload=context.resume_payload,
        )
        derived = derived_execution.create_node_context(node)
        derived.chat_tag_provider = context.chat_tag_provider
        return derived

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
