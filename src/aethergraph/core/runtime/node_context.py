from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from aethergraph.services.artifacts.facade import ArtifactFacade

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunVisibility,
)
from aethergraph.core.runtime.runtime_services import get_ext_context_service
from aethergraph.services.channel.session import ChannelSession
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.llm.providers import Provider
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.services.viz.facade import VizFacade

from .base_service import _ServiceHandle
from .bound_memory import BoundMemoryAdapter
from .node_services import NodeServices


@dataclass
class NodeContext:
    run_id: str
    session_id: str
    graph_id: str
    node_id: str
    services: NodeServices
    identity: Any = None
    resume_payload: dict[str, Any] | None = None
    scope: Scope | None = None
    agent_id: str | None = None  # for agent-invoked runs
    app_id: str | None = None  # for app-invoked runs
    bound_memory: BoundMemoryAdapter | None = None  # back-compat

    # --- accessors (compatible names) ---
    def runtime(self) -> NodeServices:
        return self.services

    async def spawn_run(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        session_id: str | None = None,
        tags: list[str] | None = None,
        visibility: RunVisibility | None = None,
        origin: RunOrigin | None = None,
        importance: RunImportance | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        Launch a new run from within this node/graph.

        - Uses RunManager.submit_run under the hood.
        - Does NOT wait for completion; just returns the new run_id.
        - session_id defaults to this context's session_id.
        """
        rm: RunManager | None = getattr(self.services, "run_manager", None)
        if rm is None:
            raise RuntimeError("NodeContext.services.run_manager is not configured")
        effective_session_id = session_id or self.session_id

        record = await rm.submit_run(
            graph_id=graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=effective_session_id,
            tags=tags,
            visibility=visibility or RunVisibility.normal,
            origin=origin or (RunOrigin.agent if agent_id is not None else RunOrigin.app),
            importance=importance or RunImportance.normal,
            agent_id=agent_id,
            app_id=app_id,
            identity=self.identity,  # internal spawn; not coming from HTTP directly
        )

        return record.run_id

    async def run_and_wait(
        self,
        graph_id: str,
        *,
        inputs: dict[str, Any],
        session_id: str | None = None,
        tags: list[str] | None = None,
        visibility: RunVisibility | None = None,
        origin: RunOrigin | None = None,
        importance: RunImportance | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[str, dict[str, Any] | None, bool, list[dict[str, Any]]]:
        """
        Run a child graph as a first-class RunManager run, and wait for completion.

        Returns:
          (child_run_id, outputs, has_waits, continuations)
        """
        rm: RunManager | None = getattr(self.services, "run_manager", None)
        if rm is None:
            raise RuntimeError("NodeContext.services.run_manager is not configured")

        effective_session_id = session_id or self.session_id

        record, outputs, has_waits, continuations = await rm.run_and_wait(
            graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=effective_session_id,
            tags=tags,
            visibility=visibility or RunVisibility.normal,
            origin=origin or (RunOrigin.agent if agent_id is not None else RunOrigin.app),
            importance=importance or RunImportance.normal,
            agent_id=agent_id,
            app_id=app_id,
            identity=self.identity,  # keep provenance consistent with spawn_run
            count_slot=False,  # nested orchestration: avoid deadlock
        )

        return record.run_id, outputs, has_waits, continuations

    async def wait_run(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
    ) -> RunRecord:
        """
        Fetch and wait for a run to complete. This version simply polls the RunManager.
        In the future, we will use ContinuationStore to manage waits more efficiently.

        Example:
            r1 = await context.spawn_run("g1", inputs={...})
            r2 = await context.spawn_run("g2", inputs={...})

            rec1, rec2 = await asyncio.gather(
                context.wait_run(r1),
                context.wait_run(r2),
            )
        """
        rm: RunManager | None = getattr(self.services, "run_manager", None)
        if rm is None:
            raise RuntimeError("NodeContext.services.run_manager is not configured")
        return await rm.wait_run(run_id, timeout_s=timeout_s)

    def logger(self):
        return self.services.logger.for_node_ctx(
            run_id=self.run_id, node_id=self.node_id, graph_id=self.graph_id
        )

    def ui_session_channel(self) -> "ChannelSession":
        """
        Convenience helper for the web UI: channel bound to this context's session.

        Uses channel key: "ui:session/<session_id>".
        """
        if not self.session_id:
            raise RuntimeError("NodeContext.session_id is not set")
        return ChannelSession(self, f"ui:session/{self.session_id}")

    def ui_run_channel(self) -> "ChannelSession":
        """
        Convenience helper for the web UI: channel bound to this specific run.

        Uses channel key: "ui:run/<run_id>".
        """
        return ChannelSession(self, f"ui:run/{self.run_id}")

    def channel(self, channel_key: str | None = None):
        return ChannelSession(self, channel_key)

    # New way: prefer memory_facade directly
    def memory(self) -> MemoryFacade:
        if not self.services.memory_facade:
            raise RuntimeError("MemoryFacade not bound")
        return self.services.memory_facade

    # Back-compat: old ctx.mem() -> To be deprecated
    def mem(self) -> BoundMemoryAdapter:
        if not self.bound_memory:
            raise RuntimeError("BoundMemory adapter not available")
        return self.bound_memory

    # Artifacts / index
    def artifacts(self) -> ArtifactFacade:
        return self.services.artifact_store

    def kv(self):
        if not self.services.kv:
            raise RuntimeError("KV not available")
        return self.services.kv

    def viz(self) -> VizFacade:
        if not self.services.viz:
            raise RuntimeError("Viz service (facade) not available")
        return self.services.viz

    def llm(
        self,
        profile: str = "default",
        *,
        provider: Provider | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
    ) -> LLMClientProtocol:
        """
        Get an LLM client by profile.
        - If no overrides are provided, just return existing profile.
        - If overrides are provided, cr`eate/update that profile at runtime.
        """
        svc = self.services.llm

        if (
            provider is None
            and model is None
            and base_url is None
            and api_key is None
            and azure_deployment is None
            and timeout is None
        ):
            return svc.get(profile)

        return svc.configure_profile(
            profile=profile,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            azure_deployment=azure_deployment,
            timeout=timeout,
        )

    def llm_set_key(self, provider: str, model: str, api_key: str, profile: str = "default"):
        """
        Quickly configure or override the provider/key for a profile.
        """
        svc = self.services.llm
        svc.set_key(provider=provider, model=model, api_key=api_key, profile=profile)

    def rag(self):
        if not self.services.rag:
            raise RuntimeError("RAGService not available")
        return self.services.rag

    def mcp(self, name):
        if not self.services.mcp:
            raise RuntimeError("MCPService not available")
        return self.services.mcp.get(name)

    def continuations(self):
        return self.services.continuation_store

    def prepare_wait_for_resume(self, token: str):
        # creates and registers a Future for this token without awaiting
        return self.services.wait_registry.register(token)

    def clock(self):
        if not self.services.clock:
            raise RuntimeError("Clock service not available")
        return self.services.clock

    def svc(self, name: str) -> Any:
        # generic accessor for external context services
        raw = get_ext_context_service(name)
        if raw is None:
            raise KeyError(f"Service '{name}' not registered")
        # bind the service to the context
        bind = getattr(raw, "bind", None)
        if callable(bind):
            return raw.bind(context=self)
        return raw

    def __getattr__(self, name: str) -> Any:
        # Try to resolve as an external context service
        try:
            bound = self.svc(name)
        except KeyError:
            # Fall back to normal attribute error for anything else
            raise AttributeError(f"NodeContext has no attribute '{name}'") from None
        # Return a callable handle that behaves like the bound service
        return _ServiceHandle(name, bound)

    def _now(self):
        if self.services.clock:
            return self.services.clock.now()
        else:
            from datetime import datetime

            return datetime.utcnow()

    # ---- continuation helpers ----
    async def create_continuation(
        self,
        *,
        kind: str,
        payload: dict | None,
        channel: str | None,
        deadline_s: int | None = None,
        poll: dict | None = None,
        attempts: int = 0,
    ) -> Continuation:
        """Create and store a continuation for this node in the continuation store."""
        token = await self.services.continuation_store.mint_token(
            self.run_id, self.node_id, attempts=attempts
        )
        deadline = None
        if deadline_s:
            deadline = self._now() + timedelta(seconds=deadline_s)

        continuation = Continuation(
            run_id=self.run_id,
            node_id=self.node_id,
            kind=kind,
            token=token,
            prompt=payload.get("prompt") if payload else None,
            resume_schema=payload.get("resume_schema") if payload else None,
            channel=channel,
            deadline=deadline,
            poll=poll,
            next_wakeup_at=deadline,
            created_at=self._now(),
            attempts=attempts,
            payload=payload,
            session_id=getattr(self, "session_id", None),
            agent_id=getattr(self, "agent_id", None),
            app_id=getattr(self, "app_id", None),
            graph_id=getattr(self, "graph_id", None),
        )
        await self.services.continuation_store.save(continuation)
        return continuation

    async def wait_for_resume(self, token: str) -> dict:
        """Wait for a continuation to be resumed, and return the payload.
        This will register the wait in the wait registry, and suspend until resumed.
        Useful for nodes that need to pause and wait for short-term external events.
        For long-term waits, use DualStage Tools instead.
        """
        waits = self.services.wait_registry
        if not waits:
            raise RuntimeError("WaitRegistry missing on context/runtime")
        fut = waits.register(token)
        payload = await fut
        return payload
