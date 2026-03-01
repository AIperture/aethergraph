from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunVisibility,
)

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager


@dataclass
class RunFacade:
    """
    Centralize node-facing child run management over `RunManager`.

    This facade is designed for `context.runner()` access and applies context
    defaults (identity, session, agent/app ids) so call sites can orchestrate
    child runs without repeating runtime plumbing.

    Examples:
        Spawn a child run and continue:
        ```python
        run_id = await context.runner().spawn_run(
            "my-graph",
            inputs={"task": "index"},
        )
        ```

        Spawn then wait for completion:
        ```python
        run_id = await context.runner().spawn_run("my-graph", inputs={"x": 1})
        record, outputs = await context.runner().wait_run(
            run_id,
            return_outputs=True,
        )
        ```

    Args:
        run_manager: Runtime run manager that persists and executes runs.
        identity: Optional default identity propagated to child runs.
        session_id: Optional default session id for child runs.
        agent_id: Optional default agent id for child runs.
        app_id: Optional default app id for child runs.

    Returns:
        RunFacade: Bound facade for child run orchestration APIs.

    Notes:
        This facade only delegates; run execution semantics are owned by
        `RunManager`.
    """

    run_manager: RunManager
    identity: RequestIdentity | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None

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
        Submit a child run and return immediately with its run id.

        This method calls `run_manager.submit_run(...)` and applies defaults from
        this facade when optional arguments are not provided.

        Examples:
            Spawn with default context identity/session:
            ```python
            run_id = await runner().spawn_run(
                "my-graph",
                inputs={"prompt": "hello"},
            )
            ```

            Spawn with explicit metadata overrides:
            ```python
            run_id = await runner().spawn_run(
                "my-graph",
                inputs={"payload": {"a": 1}},
                tags=["batch", "priority"],
                visibility=RunVisibility.inline,
                agent_id="agent-123",
            )
            ```

        Args:
            graph_id: Registered graph identifier to execute.
            inputs: Graph input payload for the child run.
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
            If `origin` is not provided, it defaults to `agent` when an effective
            agent id exists; otherwise `app`.
        """
        effective_session_id = session_id or self.session_id
        effective_agent_id = agent_id if agent_id is not None else self.agent_id
        effective_app_id = app_id if app_id is not None else self.app_id

        record = await self.run_manager.submit_run(
            graph_id=graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=effective_session_id,
            tags=tags,
            visibility=visibility or RunVisibility.normal,
            origin=origin or (RunOrigin.agent if effective_agent_id is not None else RunOrigin.app),
            importance=importance or RunImportance.normal,
            agent_id=effective_agent_id,
            app_id=effective_app_id,
            identity=self.identity,
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
        Run a child graph as a tracked run and wait for completion.

        This method delegates to `run_manager.run_and_wait(...)` and returns the
        completed run id plus execution outputs and wait metadata.

        Examples:
            Wait for a child graph:
            ```python
            run_id, outputs, has_waits, continuations = await runner().run_and_wait(
                "my-graph",
                inputs={"x": 1},
            )
            ```

            Wait with explicit run metadata:
            ```python
            run_id, outputs, has_waits, continuations = await runner().run_and_wait(
                "my-graph",
                inputs={"x": 1},
                tags=["child"],
                run_id="run-custom-001",
            )
            ```

        Args:
            graph_id: Registered graph identifier to execute.
            inputs: Graph input payload for the child run.
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
                `(run_id, outputs, has_waits, continuations)` from the completed
                child run.

        Notes:
            This method uses `count_slot=False` to avoid nested deadlock behavior
            in orchestration paths.
        """
        effective_session_id = session_id or self.session_id
        effective_agent_id = agent_id if agent_id is not None else self.agent_id
        effective_app_id = app_id if app_id is not None else self.app_id

        record, outputs, has_waits, continuations = await self.run_manager.run_and_wait(
            graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=effective_session_id,
            tags=tags,
            visibility=visibility or RunVisibility.normal,
            origin=origin or (RunOrigin.agent if effective_agent_id is not None else RunOrigin.app),
            importance=importance or RunImportance.normal,
            agent_id=effective_agent_id,
            app_id=effective_app_id,
            identity=self.identity,
            count_slot=False,
        )
        return record.run_id, outputs, has_waits, continuations

    async def wait_run(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
        return_outputs: bool = False,
    ) -> RunRecord | tuple[RunRecord, dict[str, Any] | None]:
        """
        Wait for a run to reach a terminal state.

        This method delegates to `run_manager.wait_run(...)`.

        Examples:
            Wait for a run record:
            ```python
            record = await runner().wait_run(run_id)
            ```

            Wait and also collect outputs:
            ```python
            record, outputs = await runner().wait_run(
                run_id,
                timeout_s=30,
                return_outputs=True,
            )
            ```

        Args:
            run_id: Run identifier to wait on.
            timeout_s: Optional timeout in seconds.
            return_outputs: If true, return `(record, outputs)` tuple.

        Returns:
            RunRecord | tuple[RunRecord, dict[str, Any] | None]:
                Final run record, or `(record, outputs)` when requested.

        Notes:
            Output availability depends on in-process execution context.
        """
        return await self.run_manager.wait_run(
            run_id,
            timeout_s=timeout_s,
            return_outputs=return_outputs,
        )

    async def cancel_run(self, run_id: str) -> None:
        """
        Request best-effort cancellation for a run id.

        This method delegates to `run_manager.cancel_run(...)`.

        Examples:
            Cancel a spawned run:
            ```python
            await runner().cancel_run(run_id)
            ```

            Cancel based on condition:
            ```python
            if should_abort:
                await runner().cancel_run(run_id)
            ```

        Args:
            run_id: Run identifier to cancel.

        Returns:
            None: Cancellation is requested asynchronously.

        Notes:
            Cancellation may not be immediate; scheduler termination is
            best-effort.
        """
        await self.run_manager.cancel_run(run_id)
