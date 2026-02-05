from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility

from .plan_executor import BackgroundExecutionHandle, ExecutionEvent, ExecutionResult
from .plan_types import CandidatePlan, PlanningContext, PlanningEvent
from .planner_service import PlannerService, PlanResult

PlanEventsCallback = Callable[[PlanningEvent], None]
ExecEventsCallback = Callable[[ExecutionEvent], None]


@dataclass
class NodePlanner:
    """
    Node-bound facade over PlannerService.

    - Knows about NodeContext identity/session/app/agent.
    - Delegates to PlannerService but fills execution context automatically.
    """

    service: PlannerService
    node_ctx: Any  # forward reference to NodeContext

    # ---------- Planning ----------
    async def plan(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = True,
        preferred_external_keys: list[str] | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        on_event: PlanEventsCallback | None = None,
    ) -> PlanResult:
        """
        Generate a plan based on the provided goal and context.

        This method delegates the planning process to the underlying PlannerService,
        using the provided inputs and optional callbacks for event handling.

        Examples:
            Basic usage to generate a plan:
            ```python
            result = await node_planner.plan(goal="Optimize workflow")
            ```

            Providing additional inputs and handling events:
            ```python
            result = await node_planner.plan(
                goal="Generate report",
                user_inputs={"date": "2023-10-01"},
                on_event=lambda event: print(f"Event: {event}")
            )
            ```

        Args:
            goal: The primary objective or goal for the planning process.
            user_inputs: Optional dictionary of user-provided inputs for the plan.
            external_slots: Optional dictionary of external slot values to consider.
            flow_ids: Optional list of flow identifiers to constrain the planning scope.
            instruction: Optional instruction or guidance for the planner.
            allow_partial: Whether to allow partial plans if the goal cannot be fully satisfied (default: True).
            preferred_external_keys: Optional list of preferred external keys to prioritize.
            memory_snippets: Optional list of memory snippets to include in the planning context.
            artifact_snippets: Optional list of artifact snippets to include in the planning context.
            on_event: Optional callback function to handle planning events.
                  The callback should be of type `PlanningEventCallback = Callable[[PlanningEvent], None] | Callable[[PlanningEvent], Awaitable[None]]`.

        Returns:
            PlanResult: The result of the planning process, including the generated plan and metadata.

        Notes:
        You can use `on_event` to monitor the planning process and react to intermediate events.
        """
        return await self.service.plan(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            on_event=on_event,
        )

    async def plan_with_context(
        self,
        ctx: PlanningContext,
        *,
        on_event: PlanEventsCallback | None = None,
    ) -> PlanResult:
        """
        Plan a task using the provided planning context.

        This method delegates the planning process to the underlying service, allowing
        for the generation of a plan based on the given context and optional event callback.

        Examples:
            Basic usage to generate a plan:
            ```python
            result = await planner.plan_with_context(context)
            ```

            Using an event callback to monitor planning progress:
            ```python
            async def on_event(event):
                print(f"Received event: {event}")

            result = await planner.plan_with_context(context, on_event=on_event)
            ```

        Args:
            ctx: The `PlanningContext` object containing the necessary information for planning.
            on_event: Optional callback function to handle planning events. Defaults to None.

        Returns:
            A `PlanResult` object containing the outcome of the planning process.

        Notes:
            - The `on_event` callback can be used to receive updates or intermediate results
            during the planning process.
        """
        return await self.service.plan_with_context(ctx, on_event=on_event)

    async def parse_inputs(
        self,
        *,
        message: str,
        missing_keys: list[str],
        instruction: str | None = None,
    ):
        """
        Parse input data and handle missing keys.

        This method processes the provided input message, identifies any missing keys,
        and optionally uses an instruction to guide the parsing process. It delegates
        the actual parsing logic to the `service.parse_inputs` method.

        Examples:
            Basic usage to parse inputs:
            ```python
            result = await node_planner.parse_inputs(
                message="Input data with key1 = 0.1 and key2 = dummy string inputs",
                missing_keys=["key1", "key2"]
            )
            ```

            Parsing with an additional instruction:
            ```python
            result = await node_planner.parse_inputs(
                message="Input data with key1 equals 0.1 and key2 is some dummy string inputs",
                missing_keys=["key1", "key2"],
                instruction="key1 is a float and key2 is a string. Make sure all values are included in a dictionary."
            )
            ```

        Args:
            message: The input message to be parsed.
            missing_keys: Field names whose values we want to extract.
            instruction: Optional instruction to guide the parsing process.

        Returns:
            The result of the parsing operation as returned by the `service.parse_inputs` method.

        Notes:
        This method is asynchronous and relies on the `service.parse_inputs` implementation
        for the actual parsing logic.
        """
        return await self.service.parse_inputs(
            message=message,
            missing_keys=missing_keys,
            instruction=instruction,
        )

    # ---------- Execution (auto-fill context) ----------

    async def execute(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> ExecutionResult:
        """
        Execute a plan with the provided context and parameters.

        This method uses the NodeContext to auto-fill execution metadata such as
        identity, session, agent, and application IDs. Callers can override these
        defaults by providing explicit keyword arguments.

        Examples:
            Basic usage to execute a plan:
            ```python
            result = await node_planner.execute(plan)
            ```

            Executing with additional user inputs and event handling:
            ```python
            result = await node_planner.execute(
            plan,
            user_inputs={"key1": "value1"},
            on_event=lambda event: print(f"Execution event: {event}")
            )
            ```

        Args:
            plan: The `CandidatePlan` object to execute.
            user_inputs: Optional dictionary of user-provided inputs for the execution. Values referenced as "${user.<key>}"
            on_event: Optional callback function to handle execution events.
            identity: Optional `RequestIdentity` to override the default identity.
            visibility: Visibility level for the execution (default: `RunVisibility.normal`).
            importance: Importance level for the execution (default: `RunImportance.normal`).
            session_id: Optional session ID to override the default session.
            agent_id: Optional agent ID to override the default agent.
            app_id: Optional application ID to override the default application.
            tags: Optional list of tags to associate with the execution.
            origin: Optional `RunOrigin` to specify the origin of the execution.

        Returns:
            ExecutionResult: The result of the execution process, including status and metadata.

        Notes:
        Use the `on_event` callback to monitor execution progress and handle intermediate events.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        # auto-add some helpful tags
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        if eff_origin is None:
            # tiny heuristic; tune later
            eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        return await self.service.execute_plan(
            plan,
            user_inputs=user_inputs,
            on_event=on_event,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
        )

    async def plan_and_execute(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = False,
        preferred_external_keys: list[str] | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        planning_events_cb: PlanEventsCallback | None = None,
        execution_events_cb: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> tuple[PlanResult, ExecutionResult | None]:
        """
        Plan a task and execute the resulting plan with node-bound context.

        This method combines the planning and execution phases into a single operation.
        It generates a plan based on the provided goal and context, then executes the
        accepted plan using the NodeContext to auto-fill execution metadata.

        Examples:
            Basic usage to plan and execute:
            ```python
            plan_result, exec_result = await node_planner.plan_and_execute(goal="Optimize workflow")
            ```

            Providing additional inputs and handling events:
            ```python
            plan_result, exec_result = await node_planner.plan_and_execute(
                goal="Generate report",
                user_inputs={"date": "2023-10-01"},
                planning_events_cb=lambda event: print(f"Planning event: {event}"),
                execution_events_cb=lambda event: print(f"Execution event: {event}")
            )
            ```

        Args:
            goal: The primary objective or goal for the planning process.
            user_inputs: Optional dictionary of user-provided inputs for the plan and execution.
            external_slots: Optional dictionary of external slot values to consider during planning.
            flow_ids: Optional list of flow identifiers to constrain the planning scope.
            instruction: Optional instruction or guidance for the planner.
            allow_partial: Whether to allow partial plans if the goal cannot be fully satisfied (default: False).
            preferred_external_keys: Optional list of preferred external keys to prioritize during planning.
            memory_snippets: Optional list of memory snippets to include in the planning context.
            artifact_snippets: Optional list of artifact snippets to include in the planning context.
            planning_events_cb: Optional callback function to handle planning events.
            execution_events_cb: Optional callback function to handle execution events.
            identity: Optional `RequestIdentity` to override the default identity.
            visibility: Visibility level for the execution (default: `RunVisibility.normal`).
            importance: Importance level for the execution (default: `RunImportance.normal`).
            session_id: Optional session ID to override the default session.
            agent_id: Optional agent ID to override the default agent.
            app_id: Optional application ID to override the default application.
            tags: Optional list of tags to associate with the execution.
            origin: Optional `RunOrigin` to specify the origin of the execution.

        Returns:
            tuple[PlanResult, ExecutionResult | None]: A tuple containing the result of the planning process
            and the result of the execution process (if applicable).

        Notes:
            - Use the `planning_events_cb` and `execution_events_cb` callbacks to monitor progress and handle
            intermediate events during the planning and execution phases.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        # if eff_origin is None:
        #     eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        return await self.service.plan_and_execute(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            planning_events_cb=planning_events_cb,
            execution_events_cb=execution_events_cb,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
        )

    # ---------- Background execution (auto-fill context) ----------

    async def execute_background(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
        on_complete: Callable[[ExecutionResult], Any] | None = None,
        exec_id: str | None = None,
    ) -> BackgroundExecutionHandle:
        """
        Execute a candidate plan in the background, bound to this node.
        This method initiates a fire-and-forget execution of the provided plan,
        leveraging the node's context and metadata. Progress updates can be
        reported via the `on_event` callback, and completion is signaled through
        the `on_complete` callback.

        Examples:
            Basic usage to execute a plan in the background:
            ```python
            handle = await node_planner.execute_background(plan)
            ```

            Executing with additional metadata and callbacks:
            ```python
            handle = await node_planner.execute_background(
                user_inputs={"param1": "value1"},
                on_event=event_callback,
                on_complete=completion_callback,
                tags=["custom:tag"],
            )
            ```

        Args:
            plan: The candidate plan to execute.
            user_inputs: Optional dictionary of user-provided inputs for the execution.
            on_event: Optional callback to handle execution progress events.
            identity: Optional identity to associate with the execution; defaults to the node's context identity.
            visibility: The visibility level of the execution (default: RunVisibility.normal).
            importance: The importance level of the execution (default: RunImportance.normal).
            session_id: Optional session ID to associate with the execution; defaults to the node's context session ID.
            agent_id: Optional agent ID to associate with the execution; defaults to the node's context agent ID.
            app_id: Optional application ID to associate with the execution; defaults to the node's context app ID.
            tags: Optional list of tags to associate with the execution; additional tags are derived from the node's context.
            origin: Optional origin of the execution; defaults to `RunOrigin.agent` if an agent ID is present, otherwise `RunOrigin.user`.
            on_complete: Optional callback to handle execution completion.
            exec_id: Optional explicit execution ID to associate with the execution.

        Returns:
            BackgroundExecutionHandle: A handle to monitor or interact with the background execution.

        Notes:
            - Tags automatically include identifiers for the graph, node, and run (if available) from the node's context.
            - The `on_event` callback is invoked with progress updates, while the `on_complete` callback is invoked upon completion.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        if eff_origin is None:
            eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        handle = await self.service.execute_background(
            plan,
            user_inputs=user_inputs,
            on_event=on_event,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
            on_complete=on_complete,
            exec_id=exec_id,
        )
        return handle

    async def plan_and_execute_background(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = False,
        preferred_external_keys: list[str] | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        planning_events_cb: PlanEventsCallback | None = None,
        execution_events_cb: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
        on_complete: Callable[[ExecutionResult], Any] | None = None,
        exec_id: str | None = None,
    ) -> tuple[PlanResult, BackgroundExecutionHandle | None]:
        """
        Plan and execute a goal asynchronously in the background.
        This method delegates the planning and execution of a goal to the service layer,
        while auto-filling the node context and other metadata. It supports callbacks
        for planning and execution events, and allows for customization of execution
        parameters such as visibility, importance, and tags.

        Examples:
            Basic usage to plan and execute a goal:
            ```python
            plan_result, handle = await node_planner.plan_and_execute_background(
                goal="Achieve X",
                user_inputs={"key": "value"}
            )
            ```

            Using callbacks and additional metadata:
            ```python
            plan_result, handle = await node_planner.plan_and_execute_background(
                goal="Achieve Y",
                planning_events_cb=on_planning_event,
                execution_events_cb=on_execution_event,
                tags=["custom-tag"],
                visibility=RunVisibility.inline,
            )
            ```

        Args:
            goal: The goal to achieve during the planning and execution process.
            user_inputs: Optional dictionary of user-provided inputs for planning.
            external_slots: Optional dictionary of external slots to use during planning.
            flow_ids: Optional list of flow IDs to consider during execution.
            instruction: Optional instruction to guide the planning process.
            allow_partial: Whether to allow partial execution if the full goal cannot be achieved.
            preferred_external_keys: Optional list of preferred external keys for planning.
            memory_snippets: Optional list of memory snippets to include in planning.
            artifact_snippets: Optional list of artifact snippets to include in planning.
            planning_events_cb: Optional callback for planning events.
            execution_events_cb: Optional callback for execution events.
            identity: Optional identity to use for the execution context.
            visibility: Visibility level for the execution (default: normal).
            importance: Importance level for the execution (default: normal).
            session_id: Optional session ID to associate with the execution.
            agent_id: Optional agent ID to associate with the execution.
            app_id: Optional application ID to associate with the execution.
            tags: Optional list of tags to associate with the execution.
            origin: Optional origin metadata for the execution.
            on_complete: Optional callback to invoke upon completion of execution.
            exec_id: Optional execution ID to associate with the process.

        Returns:
            A tuple containing the planning result and an optional background execution handle.

        Notes:
            - The method automatically appends node-specific tags such as `graph:<graph_id>`
              and `node:<node_id>` to the provided tags.
            - If a `run_id` is available in the node context, it is also appended as a tag.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        # keep same behavior as plan_and_execute (no default if None)

        plan_result, handle = await self.service.plan_and_execute_background(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            planning_events_cb=planning_events_cb,
            execution_events_cb=execution_events_cb,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
            on_complete=on_complete,
            exec_id=exec_id,
        )

        return plan_result, handle
