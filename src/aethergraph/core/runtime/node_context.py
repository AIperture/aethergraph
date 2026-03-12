from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import warnings

from aethergraph.contracts.services.execution import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ExecutionService,
    Language,
)
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunVisibility,
)
from aethergraph.core.runtime.runtime_services import get_ext_context_service
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.channel.session import ChannelSession
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.knowledge.node_kb import NodeKB
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.providers import Provider
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.planning.node_planner import NodePlanner
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.services.skills.skill_registry import SkillRegistry
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.websearch.facade import WebSearchFacade

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

    _planner_facade: NodePlanner | None = None  # lazy init

    # --- accessors (compatible names) ---
    def runtime(self) -> NodeServices:
        return self.services

    async def execute(
        self,
        code: str,
        *,
        language: Language = "python",
        timeout_s: float = 30.0,
        args: list[str] | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        """ """
        exe_svs: ExecutionService | None = getattr(self.services, "execution", None)
        if exe_svs is None:
            raise RuntimeError("NodeContext.services.execution is not configured")

        req = CodeExecutionRequest(
            language=language,
            code=code,
            args=args or [],
            timeout_s=timeout_s,
            workdir=workdir,
            env=env,
        )
        return await exe_svs.execute(req)

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
        Deprecated wrapper for `context.runner().spawn_run(...)`.

        This method is kept for backward compatibility and forwards all arguments
        to the centralized runner facade.

        Examples:
            Spawn through legacy context API:
            ```python
            run_id = await context.spawn_run("my-graph-id", inputs={"x": 1})
            ```

            Preferred equivalent:
            ```python
            run_id = await context.runner().spawn_run("my-graph-id", inputs={"x": 1})
            ```

        Args:
            graph_id: Registered graph identifier to execute.
            inputs: Input payload passed to the child graph.
            session_id: Optional session id override.
            tags: Optional run tags.
            visibility: Optional visibility override.
            origin: Optional origin override.
            importance: Optional importance override.
            agent_id: Optional agent id override.
            app_id: Optional app id override.
            run_id: Optional explicit run id.

        Returns:
            str: Created child run id.

        Notes:
            Prefer `context.runner().spawn_run(...)` for new code.
        """
        warnings.warn(
            "NodeContext.spawn_run() is deprecated; use context.runner().spawn_run().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.runner().spawn_run(
            graph_id,
            inputs=inputs,
            session_id=session_id,
            tags=tags,
            visibility=visibility,
            origin=origin,
            importance=importance,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )

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
        Deprecated wrapper for `context.runner().run_and_wait(...)`.

        This method is kept for backward compatibility and delegates to the
        centralized runner facade.

        Examples:
            Run synchronously through legacy API:
            ```python
            run_id, outputs, has_waits, continuations = await context.run_and_wait(
                "my-graph-id",
                inputs={"x": 1},
            )
            ```

            Preferred equivalent:
            ```python
            run_id, outputs, has_waits, continuations = await context.runner().run_and_wait(
                "my-graph-id",
                inputs={"x": 1},
            )
            ```

        Args:
            graph_id: Registered graph identifier to execute.
            inputs: Input payload passed to the child graph.
            session_id: Optional session id override.
            tags: Optional run tags.
            visibility: Optional visibility override.
            origin: Optional origin override.
            importance: Optional importance override.
            agent_id: Optional agent id override.
            app_id: Optional app id override.
            run_id: Optional explicit run id.

        Returns:
            tuple[str, dict[str, Any] | None, bool, list[dict[str, Any]]]:
                `(run_id, outputs, has_waits, continuations)`.

        Notes:
            Prefer `context.runner().run_and_wait(...)` for new code.
        """
        warnings.warn(
            "NodeContext.run_and_wait() is deprecated; use context.runner().run_and_wait().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.runner().run_and_wait(
            graph_id,
            inputs=inputs,
            session_id=session_id,
            tags=tags,
            visibility=visibility,
            origin=origin,
            importance=importance,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )

    async def wait_run(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
        return_outputs: bool = False,
    ) -> RunRecord | tuple[RunRecord, dict[str, Any] | None]:
        """
        Deprecated wrapper for `context.runner().wait_run(...)`.

        This method is kept for backward compatibility and delegates to the
        centralized runner facade.

        Examples:
            Wait through legacy API:
            ```python
            record = await context.wait_run(run_id)
            ```

            Preferred equivalent:
            ```python
            record, outputs = await context.runner().wait_run(run_id, return_outputs=True)
            ```

        Args:
            run_id: Run identifier to wait on.
            timeout_s: Optional timeout in seconds.
            return_outputs: If true, return `(record, outputs)`.

        Returns:
            RunRecord | tuple[RunRecord, dict[str, Any] | None]:
                Final run record, or tuple when `return_outputs=True`.

        Notes:
            Prefer `context.runner().wait_run(...)` for new code.
        """
        warnings.warn(
            "NodeContext.wait_run() is deprecated; use context.runner().wait_run().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.runner().wait_run(
            run_id,
            timeout_s=timeout_s,
            return_outputs=return_outputs,
        )

    async def cancel_run(self, run_id: str) -> None:
        """
        Deprecated wrapper for `context.runner().cancel_run(...)`.

        This method is kept for backward compatibility and delegates to the
        centralized runner facade.

        Examples:
            Cancel through legacy API:
            ```python
            await context.cancel_run(run_id)
            ```

            Preferred equivalent:
            ```python
            await context.runner().cancel_run(run_id)
            ```

        Args:
            run_id: Run identifier to cancel.

        Returns:
            None: Cancellation is requested asynchronously.

        Notes:
            Prefer `context.runner().cancel_run(...)` for new code.
        """
        warnings.warn(
            "NodeContext.cancel_run() is deprecated; use context.runner().cancel_run().",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.runner().cancel_run(run_id)

    def planner(self) -> "NodePlanner":
        if self._planner_facade is None:
            if self.services.planner_service is None:
                raise RuntimeError("NodeContext.services.planner_service is not configured")
            self._planner_facade = NodePlanner(
                service=self.services.planner_service,
                node_ctx=self,
            )
        return self._planner_facade

    def logger(self):
        if not self.services.logger:
            raise RuntimeError("Logger service not available")
        return self.services.logger.for_node_ctx(
            run_id=self.run_id, node_id=self.node_id, graph_id=self.graph_id
        )

    async def emit_agent_event(
        self,
        *,
        event_type: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        status: str = "info",
        tags: list[str] | None = None,
        producer_name: str = "node_context",
        producer_version: str | None = None,
        payload_schema_name: str | None = None,
        payload_schema_version: int | None = 1,
        parent_event_id: str | None = None,
        caused_by_event_id: str | None = None,
    ) -> dict[str, Any]:
        from aethergraph.services.inspect import emit_agent_event

        return await emit_agent_event(
            event_type=event_type,
            summary=summary,
            payload=payload,
            status=status,
            tags=tags,
            producer_name=producer_name,
            producer_version=producer_version,
            payload_schema_name=payload_schema_name,
            payload_schema_version=payload_schema_version,
            parent_event_id=parent_event_id,
            caused_by_event_id=caused_by_event_id,
        )

    def ui_session_channel(self) -> "ChannelSession":
        """
        Creates a new ChannelSession for the current node context with session key as
        `ui:session/<session_id>`.

        This method is a convenience helper for the AG UI to get the default session channel.

        Returns:
            ChannelSession: The channel session associated with the current session.
        """
        warnings.warn(
            "NodeContext.ui_session_channel() is deprecated; use context.channel('ui:session').",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.channel("ui:session")

    def ui_run_channel(self) -> "ChannelSession":
        """
        Creates a new ChannelSession for the current node context with session key as
        `ui:run/<run_id>`.

        This method is a convenience helper for the AG UI to get the default run channel.

        Returns:
            ChannelSession: The channel session associated with the current run.
        """
        warnings.warn(
            "NodeContext.ui_run_channel() is deprecated; use context.channel('ui:run').",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.channel("ui:run")

    def triggers(self) -> TriggerFacade:
        if not self.services.triggers:
            raise RuntimeError("NodeContext.services.triggers is not configured")
        return self.services.triggers

    def skills(self) -> SkillRegistry:
        if not self.services.skills:
            raise RuntimeError("NodeContext.services.skills is not configured")
        return self.services.skills

    def registry(self) -> RegistryFacade:
        if not self.services.registry:
            raise RuntimeError("NodeContext.services.registry is not configured")
        return self.services.registry

    def channel(self, channel_key: str | None = None):
        """
        Set up a new ChannelSession for the current node context.

        Args:
            channel_key (str | None): An optional key to specify a particular channel.
            If not provided, the default channel will be used.
            Special shorthand values are supported:
            - `ui:session` -> `ui:session/<current_session_id>`
            - `ui:run` -> `ui:run/<current_run_id>`

        Returns:
            ChannelSession: An instance representing the session for the specified channel.

        Notes:
            Supported channel key formats include:

            | Channel Type         | Format Example                                 | Notes                                 |
            |----------------------|-----------------------------------------------|---------------------------------------|
            | Console              | `console:stdin`                               | Console input/output                  |
            | Slack                | `slack:team/{team_id}:chan/{channel_id}`      | Needs additional configuration        |
            | Telegram             | `tg:chat/{chat_id}`                           | Needs additional configuration        |
            | UI Session           | `ui:session/{session_id}`                     | Requires AG web UI                    |
            | UI Session (current) | `ui:session`                                  | Expands to current `session_id`       |
            | UI Run               | `ui:run/{run_id}`                             | Requires AG web UI                    |
            | UI Run (current)     | `ui:run`                                      | Expands to current `run_id`           |
            | Webhook              | `webhook:{unique_identifier}`                 | For Slack, Discord, Zapier, etc.      |
            | File-based channel   | `file:path/to/directory`                      | File system based channels            |
        """
        resolved_key = channel_key
        if channel_key == "ui:session":
            resolved_key = f"ui:session/{self.session_id}"
        elif channel_key == "ui:run":
            resolved_key = f"ui:run/{self.run_id}"

        return ChannelSession(self, resolved_key)

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

    def kb(self) -> NodeKB:
        if not self.services.kb:
            raise RuntimeError("NodeKB service not available")
        return self.services.kb

    def web_search(self) -> WebSearchFacade:
        if not self.services.web_search:
            raise RuntimeError("Web search service not available")
        return self.services.web_search

    def runner(self) -> RunFacade:
        """
        Get the centralized run management facade for this node context.

        This accessor returns the bound `RunFacade` created during runtime wiring.

        Examples:
            Spawn a child run:
            ```python
            run_id = await context.runner().spawn_run("my-graph-id", inputs={"x": 1})
            ```

            Wait for completion:
            ```python
            record = await context.runner().wait_run(run_id)
            ```

        Args:
            None: This accessor takes no arguments.

        Returns:
            RunFacade: Run management facade for spawn/wait/cancel operations.

        Notes:
            This is the preferred API over legacy `context.spawn_run()` style
            helpers.
        """
        if not self.services.runner:
            raise RuntimeError("Run facade service not available")
        return self.services.runner

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
    ) -> GenericLLMClient:
        """
        Retrieve or configure an LLM client for this context.

        This method allows you to access a language model client by profile name,
        or dynamically override its configuration at runtime.

        Examples:
            Get the default LLM client:
            ```python
            llm = context.llm()
            response = await llm.complete("Hello, world!")
            ```

            Use a custom profile:
            ```python
            llm = context.llm(profile="my-profile")
            ```

            Override provider and model for a one-off call:
            ```python
            llm = context.llm(
                provider=Provider.OpenAI,
                model="gpt-4-turbo",
                api_key="sk-...",
            )
            ```

        Args:
            profile: The profile name to use (default: "default"). Set up in `.env` or `register_llm_client()` method.
            provider: Optionally override the provider (e.g., `Provider.OpenAI`).
            model: Optionally override the model name.
            base_url: Optionally override the base URL for the LLM API.
            api_key: Optionally override the API key for authentication.
            azure_deployment: Optionally specify an Azure deployment name.
            timeout: Optionally set a request timeout (in seconds).

        Returns:
            LLMClientProtocol: The configured LLM client instance for this context.
        """
        svc = self.services.llm

        if svc is None:
            raise RuntimeError("LLM service not available")

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
        Quickly configure or override the LLM provider, model, and API key for a given profile.

        This method allows you to update the credentials and model configuration for a specific
        LLM profile at runtime. It is useful for dynamically switching providers or rotating keys
        without restarting the application.

        Examples:
            Set the OpenAI API key for the default profile:
            ```python
            context.llm_set_key(
                provider="openai",
                model="gpt-4-turbo",
                api_key="sk-...",
            )
            ```

            Configure a custom profile for Anthropic:
            ```python
            context.llm_set_key(
                provider="anthropic",
                model="claude-3-opus",
                api_key="sk-ant-...",
                profile="anthropic-profile"
            )
            ```

        Args:
            provider: The LLM provider name (e.g., "openai", "anthropic").
            model: The model name or identifier to use.
            api_key: The API key or credential for the provider.
            profile: The profile name to update (default: "default").

        Returns:
            None. The profile is updated in-place and will be used for subsequent calls
            to `context.llm(profile=...)`.
        """
        svc = self.services.llm
        if svc is None:
            raise RuntimeError("LLM service not available")
        svc.set_key(provider=provider, model=model, api_key=api_key, profile=profile)

    def mcp(self, name):
        if not self.services.mcp:
            raise RuntimeError("MCPService not available")
        return self.services.mcp.get(name)

    def indices(self) -> ScopedIndices:
        if not self.services.indices:
            raise RuntimeError("ScopedIndices not available")
        return self.services.indices

    # def run_manager(self):
    #     # Deprecated legacy accessor; use context.runner() instead.
    #     return self.runner()

    def continuations(self):
        return self.services.continuation_store

    def prepare_wait_for_resume(self, token: str):
        # creates and registers a Future for this token without awaiting
        if not self.services.wait_registry:
            raise RuntimeError("WaitRegistry missing on context/runtime")
        return self.services.wait_registry.register(token)

    def clock(self):
        if not self.services.clock:
            raise RuntimeError("Clock service not available")
        return self.services.clock

    def svc(self, name: str) -> Any:
        """
        Retrieve and bind an external context service by name. This method is equivalent to `context.<service_name>()`.
        User can use either `context.svc("service_name")` or `context.service_name()` to access the service.

        This method accesses a registered external service, optionally binding it to the current
        node context if the service supports context binding via a `bind` method.

        Examples:
            Basic usage to access a service:
            ```python
            db = context.svc("database")
            ```

            Accessing a service that requires context binding:
            ```python
            logger = context.svc("logger")
            logger.info("Node started.")
            ```

        Args:
            name: The unique string identifier of the external service to retrieve.

        Returns:
            Any: The external service instance, bound to the current context if applicable.

        Raises:
            KeyError: If the requested service is not registered in the external context.
        """
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
        """
        Retrieve and bind an external context service by name. This allows accessing services as attributes on the context object.

        This method overrides attribute access to dynamically resolve external services registered in the context.
        If a service with the requested name exists, it is retrieved and wrapped in a `_ServiceHandle` for ergonomic access.
        The returned handle allows attribute access, direct retrieval, and call forwarding if the service is callable.

        Examples:
            ```python
            # Retrieve a database service and run a query
            db = context.database()
            db.query("SELECT * FROM users")

            # Access a logger service and log a message
            context.logger.info("Hello from node!")

            # Forward arguments to a callable service
            result = context.some_tool("input text")
            ```

        Args:
            name: The name of the service to resolve as an attribute.

        Returns:
            _ServiceHandle: A callable handle to the resolved service.

        Raises:
            AttributeError: If no service with the given name exists in the context.

        Usage:
            - You can access external services directly as attributes on the context object.
            For example, if you have registered a service named "my_service", you can use:

                ```python
                # Get the service instance
                svc = context.my_service()

                # Call the service if it's callable
                result = context.my_service(arg1, arg2)

                # Access service attributes
                value = context.my_service.some_attribute
                ```

            - In your Service, you can use `self.ctx` to access the node context if needed. For example:
                ```python
                class MyService:
                    ...
                    def my_method(self, ...):
                        context = self.ctx  # Access the NodeContext
                        # Use context information as needed
                        context.channel.send("Hello from MyService!")
                ```

        Notes:
            - If the service is not registered, an AttributeError is raised.
            - If the service is callable, calling `context.service_name(args)` will forward the call.
            - If you call `context.service_name()` with no arguments, you get the underlying service instance.
            - Attribute access (e.g., `context.service_name.some_attr`) is delegated to the service.


        """
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
