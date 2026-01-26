from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.execution import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ExecutionService,
)
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.indices.scoped_indices import ScopedIndices

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

    async def execute(
        self,
        code: str,
        *,
        language: str = "python",
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
        print("Executing code via NodeContext.execute()...")
        print(req)
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
        Launch a new run from within the current node or graph context.

         This method creates and schedules a new run for the specified graph, using the provided inputs and optional metadata.
        It does not wait for the run to complete; instead, it returns immediately with the new run's ID.
        The run is managed asynchronously in the background, and is tracked and persisted via the configured RunManager.

        Examples:
            Basic usage to spawn a run for a graph:
            ```python
            run_id = await context.spawn_run(
                "my-graph-id",
                inputs={"x": 1, "y": 2}
            )
            ```

            Spawning a run with custom tags and agent context:
            ```python
            from aethergraph.runtime import RunVisibility
            run_id = await context.spawn_run(
                "my-graph-id",
                inputs={"foo": "bar"},
                tags=["experiment", "priority"],
                agent_id="agent-123",    # associate with an agent if applicable
                visibility=RunVisibility.inline, # not shown in UI
            )
            ```

        Args:
            graph_id: The unique identifier of the graph to execute. i.e. the `name` field of a registered graph.
            inputs: Dictionary of input values to pass to the graph.
            session_id: Optional session identifier. Defaults to the current context's session if not provided.
            tags: Optional list of string tags for categorization and tracking.
            visibility: Optional visibility setting for the run (e.g., public, private, normal).
            origin: Optional indicator of the run's origin (e.g., agent, app). Defaults based on agent_id.
            importance: Optional importance level for the run (e.g., normal, high).
            agent_id: Optional agent identifier if the run is associated with an agent.
            app_id: Optional application identifier if the run is associated with an app.
            run_id: Optional explicit run identifier. If not provided, one is generated.

        Returns:
            str: The unique run_id of the newly created run.

        Raises:
            RuntimeError: If the RunManager service is not configured in the context.

        Notes:
            - The spawned run inherits the context's identity for provenance tracking.
            - Metadata `tags`, `visibility`, `origin`, `importance`, `agent_id`, `app_id`, help manage and monitor the run in AG UI,
                but do not affect the execution logic of the graph itself. If you are not using AG UI, these fields can be omitted.
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
        Run a child graph as a first-class RunManager run and wait for completion.

        This method launches a new run for the specified graph, waits for it to finish,
        and returns its outputs and metadata. The run is tracked and visualized in the UI,
        and all status updates are persisted via the RunManager.

        Examples:
            Basic usage to run and wait for a graph:
            ```python
            run_id, outputs, has_waits, continuations = await context.run_and_wait(
                "my-graph-id",
                inputs={"x": 1, "y": 2}
            )
            ```

            Running with custom tags and agent context:
            ```python
            run_id, outputs, has_waits, continuations = await context.run_and_wait(
                "my-graph-id",
                inputs={"foo": "bar"},
                tags=["experiment", "priority"],
                agent_id="agent-123",
                visibility=RunVisibility.inline,
            )
            ```

        Args:
            graph_id: The unique identifier of the graph to execute.
            inputs: Dictionary of input values to pass to the graph.
            session_id: Optional session identifier. Defaults to the current context's session.
            tags: Optional list of string tags for categorization and tracking.
            visibility: Optional visibility setting for the run (e.g., public, private, normal).
            origin: Optional indicator of the run's origin (e.g., agent, app).
            importance: Optional importance level for the run (e.g., normal, high).
            agent_id: Optional agent identifier if the run is associated with an agent.
            app_id: Optional application identifier if the run is associated with an app.
            run_id: Optional explicit run identifier. If not provided, one is generated.

        Returns:
            run_id (str): The unique run ID of the completed run.
            outputs (dict | None): The outputs returned by the graph.
            has_waits (bool): True if the run contained any wait nodes. [Not currently used]
            continuations (list[dict]): List of continuation metadata, if any. [Not currently used]

        Raises:
            RuntimeError: If the RunManager service is not configured in the context.

        Notes:
            - The run is fully tracked and visualized in the AG UI.
            - Use this method for orchestration patterns where you need to await child runs.
            - Metadata fields help with monitoring and provenance, but do not affect graph logic.

        Warning:
            - This method blocks until the child run completes.
            - This method will not honor the concurrency limits of the parent run, and may lead to deadlocks if the parent run is waiting on resources held by the child run.
            - Avoid using this method in high-concurrency scenarios to prevent deadlocks.
              For such cases, consider using `spawn_run` followed by `wait_run` instead.
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
        return_outputs: bool = False,
    ) -> RunRecord:
        """
        Wait for a run to complete and retrieve its final record.

        This method waits the RunManager for the specified run until it finishes,
        then returns the completed RunRecord. Optionally, a timeout (in seconds)
        can be set to limit how long to wait.

        Examples:
            Basic usage to wait for a run:
            ```python
            run_id = await context.spawn_run("my-graph-id", inputs={"x": 1})
            record = await context.wait_run(run_id)
            ```

            Waiting with a timeout:
            ```python
            record, outputs = await context.wait_run(run_id, timeout_s=30, return_outputs=True)
            ```

        Args:
            run_id: The unique identifier of the run to wait for.
            timeout_s: Optional timeout in seconds. If set, the method will raise
                a TimeoutError if the run does not complete in time.
            return_outputs: If True, also return the run's outputs along with the record.

        Returns:
            RunRecord: The final record of the completed run.
            Output: If `return_outputs` is True, returns a tuple of (RunRecord, outputs dict).

        Raises:
            RuntimeError: If the RunManager service is not configured in the context.
            TimeoutError: If the run does not complete within the specified timeout.

        Notes:
            - This method is useful for orchestration patterns where you need to
              synchronize on the completion of child runs.
            - For high-concurrency scenarios, prefer using `spawn_run` and `wait_run`
              in combination rather than `run_and_wait`.
        """
        rm: RunManager | None = getattr(self.services, "run_manager", None)
        if rm is None:
            raise RuntimeError("NodeContext.services.run_manager is not configured")
        return await rm.wait_run(run_id, timeout_s=timeout_s, return_outputs=return_outputs)

    async def cancel_run(self, run_id: str) -> None:
        """
        Cancel a scheduled or running child run by its unique ID.

        This method requests cancellation of a run managed by the RunManager.
        The cancellation is propagated to the run's execution context, and any
        in-progress tasks will be interrupted if possible.

        Examples:
            Basic usage to cancel a spawned run:
            ```python
            run_id = await context.spawn_run("my-graph-id", inputs={"x": 1})
            await context.cancel_run(run_id)
            ```

            Cancel a run after waiting for a condition:
            ```python
            if should_abort:
                await context.cancel_run(run_id)
            ```

        Args:
            run_id: The unique identifier of the run to cancel.

        Returns:
            None. The cancellation request is dispatched to the RunManager.

        Raises:
            RuntimeError: If the RunManager service is not configured in the context.

        Notes:
            - Cancellation is best-effort and may not immediately terminate all tasks.
            - Use this method for orchestration patterns where you need to abort child runs.
            - The run's status will be updated to "cancelled" in the UI and persistence layer.
        """
        rm: RunManager | None = getattr(self.services, "run_manager", None)
        if rm is None:
            raise RuntimeError("NodeContext.services.run_manager is not configured")
        await rm.cancel_run(run_id)

    def logger(self):
        return self.services.logger.for_node_ctx(
            run_id=self.run_id, node_id=self.node_id, graph_id=self.graph_id
        )

    def ui_session_channel(self) -> "ChannelSession":
        """
        Creates a new ChannelSession for the current node context with session key as
        `ui:session/<session_id>`.

        This method is a convenience helper for the AG UI to get the default session channel.

        Returns:
            ChannelSession: The channel session associated with the current session.
        """
        if not self.session_id:
            raise RuntimeError("NodeContext.session_id is not set")
        return ChannelSession(self, f"ui:session/{self.session_id}")

    def ui_run_channel(self) -> "ChannelSession":
        """
        Creates a new ChannelSession for the current node context with session key as
        `ui:run/<run_id>`.

        This method is a convenience helper for the AG UI to get the default run channel.

        Returns:
            ChannelSession: The channel session associated with the current run.
        """
        return ChannelSession(self, f"ui:run/{self.run_id}")

    def channel(self, channel_key: str | None = None):
        """
        Set up a new ChannelSession for the current node context.

        Args:
            channel_key (str | None): An optional key to specify a particular channel.
            If not provided, the default channel will be used.

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
            | UI Run               | `ui:run/{run_id}`                             | Requires AG web UI                    |
            | Webhook              | `webhook:{unique_identifier}`                 | For Slack, Discord, Zapier, etc.      |
            | File-based channel   | `file:path/to/directory`                      | File system based channels            |
        """
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
        svc.set_key(provider=provider, model=model, api_key=api_key, profile=profile)

    def rag(self):
        if not self.services.rag:
            raise RuntimeError("RAGService not available")
        return self.services.rag

    def mcp(self, name):
        if not self.services.mcp:
            raise RuntimeError("MCPService not available")
        return self.services.mcp.get(name)

    def indices(self) -> ScopedIndices:
        if not self.services.indices:
            raise RuntimeError("ScopedIndices not available")
        return self.services.indices

    # def run_manager(self) -> RunManager:
    #     if not self.services.run_manager:
    #         raise RuntimeError("RunManager not available")
    #     return self.services.run_manager

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
