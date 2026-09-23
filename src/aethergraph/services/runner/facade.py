from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Event
from typing import TYPE_CHECKING, Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import OriginBinding
from aethergraph.core.runtime.run_cancellation import get_run_cancellation_registry
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunParent,
    RunRecord,
    RunVisibility,
)
from aethergraph.observability.operations import resolve_operation_observer

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
        origin_binding: Optional immutable run origin propagated to child runs.

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
    current_run_id: str | None = None
    origin_binding: OriginBinding | None = None

    def _child_run_config(self, *, session_id: str | None = None) -> dict[str, Any]:
        """Build inherited runtime configuration for a child run.

        The returned mapping carries trusted parent provenance and the child-session
        origin without mutating shared Channel services or the parent binding.

        Examples:
            Build configuration with a run origin:
            ```python
            binding = OriginBinding(
                integration_id="endpoint",
                route_id="route.endpoint",
                session_id="s-1",
                channel_key="endpoint:sessions/s-1",
                external_conversation_id="s-1",
                capability_profile_id="agent-endpoint/v1",
            )
            facade = RunFacade(manager, origin_binding=binding)
            config = facade._child_run_config()
            ```

            Build configuration without a scoped channel:
            ```python
            facade = RunFacade(manager)
            config = facade._child_run_config()
            ```

        Args:
            session_id: Effective child session; omission retains the facade session.

        Returns:
            dict[str, Any]: Available parent provenance and serialized origin binding.

        Notes:
            The mapping is passed through the existing `run_config` path.
        """
        config: dict[str, Any] = {}
        if self.current_run_id and self.session_id:
            config["parent_run"] = asdict(RunParent(self.current_run_id, self.session_id))
        if self.origin_binding is not None:
            binding = self.origin_binding
            if session_id is not None and binding.session_id != session_id:
                binding = binding.model_copy(update={"session_id": session_id})
            config["origin_binding"] = binding.model_dump(mode="json")
        return config

    async def inspect_run(self, run_id: str) -> RunRecord:
        """Read current run metadata without waiting or affecting execution.

        Reads the canonical run store and restricts the result to this facade's
        session or a direct child owned by that session, within the same tenant.
        Missing and out-of-scope identities both fail lookup.

        Examples:
            Inspect a submitted child:
            ```python
            record = await context.runner().inspect_run(child_run_id)
            print(record.status.value)
            ```

            Inspect the current run:
            ```python
            record = await context.runner().inspect_run(context.run_id)
            print(record.result_available)
            ```

        Args:
            run_id: Exact canonical run identity.

        Returns:
            RunRecord: Current metadata, including completion and result availability.

        Notes:
            This observation neither requests cancellation nor waits for completion.
            Result payloads remain available through `wait_run(return_outputs=True)`.
            Parent ownership comes from retained admission metadata, never tags,
            input previews, or a matching graph name. Later turns in the parent
            session retain the same authority after the initiating run completes.
        """
        if not run_id:
            raise ValueError("Run inspection requires an exact run identity")
        record = await self.run_manager.get_record(run_id)
        if record is None or (
            self.session_id is not None
            and record.session_id != self.session_id
            and (record.parent is None or record.parent.session_id != self.session_id)
        ):
            raise LookupError("Run does not exist in the caller scope")
        if self.identity is not None and (
            record.org_id != self.identity.org_id or record.user_id != self.identity.user_id
        ):
            raise LookupError("Run does not exist in the caller scope")
        return record

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
        observer = resolve_operation_observer()
        span = await observer.start_span(
            service="runner",
            operation="spawn_run",
            request={
                "graph_id": graph_id,
                "inputs": inputs,
                "session_id": effective_session_id,
                "tags": tags,
                "run_id": run_id,
            },
            tags=["runner", "spawn"],
            metadata={"target_run_id": run_id},
        )
        try:
            record = await self.run_manager.submit_run(
                graph_id=graph_id,
                inputs=inputs,
                run_id=run_id,
                session_id=effective_session_id,
                tags=tags,
                visibility=visibility or RunVisibility.normal,
                origin=origin
                or (RunOrigin.agent if effective_agent_id is not None else RunOrigin.app),
                importance=importance or RunImportance.normal,
                agent_id=effective_agent_id,
                app_id=effective_app_id,
                identity=self.identity,
                run_config=self._child_run_config(session_id=effective_session_id),
            )
            await span.finish(
                response={"run_id": record.run_id},
                metadata={"target_run_id": record.run_id},
            )
            return record.run_id
        except Exception as exc:
            await span.fail(exc, metadata={"target_run_id": run_id})
            raise

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
        observer = resolve_operation_observer()
        span = await observer.start_span(
            service="runner",
            operation="run_and_wait",
            request={
                "graph_id": graph_id,
                "inputs": inputs,
                "session_id": effective_session_id,
                "tags": tags,
                "run_id": run_id,
            },
            tags=["runner", "wait"],
            metadata={"target_run_id": run_id},
        )
        try:
            record, outputs, has_waits, continuations = await self.run_manager.run_and_wait(
                graph_id,
                inputs=inputs,
                run_id=run_id,
                session_id=effective_session_id,
                tags=tags,
                visibility=visibility or RunVisibility.normal,
                origin=origin
                or (RunOrigin.agent if effective_agent_id is not None else RunOrigin.app),
                importance=importance or RunImportance.normal,
                agent_id=effective_agent_id,
                app_id=effective_app_id,
                identity=self.identity,
                count_slot=False,
                run_config=self._child_run_config(session_id=effective_session_id),
            )
            if has_waits:
                await span.wait(
                    metadata={
                        "target_run_id": record.run_id,
                        "continuations": continuations,
                    },
                    request={"graph_id": graph_id},
                )
            await span.finish(
                response={"outputs": outputs, "has_waits": has_waits},
                metadata={
                    "target_run_id": record.run_id,
                    "continuations": continuations,
                },
            )
            return record.run_id, outputs, has_waits, continuations
        except Exception as exc:
            await span.fail(exc, metadata={"target_run_id": run_id})
            raise

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
            When `return_outputs=True`, succeeded runs now resolve durable outputs
            even after process boundaries when persisted results are available.
        """
        observer = resolve_operation_observer()
        span = await observer.start_span(
            service="runner",
            operation="wait_run",
            request={"run_id": run_id, "timeout_s": timeout_s, "return_outputs": return_outputs},
            tags=["runner", "wait"],
            metadata={"target_run_id": run_id},
        )
        try:
            await self.inspect_run(run_id)
            result = await self.run_manager.wait_run(
                run_id,
                timeout_s=timeout_s,
                return_outputs=return_outputs,
            )
            await span.finish(response=result, metadata={"target_run_id": run_id})
            return result
        except Exception as exc:
            await span.fail(exc, metadata={"target_run_id": run_id})
            raise

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "user_requested",
    ) -> None:
        """Request best-effort cancellation with an exact semantic cause.

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

            Cancel a child because its parent stopped:
            ```python
            await runner().cancel_run(run_id, reason="parent_cancelled")
            ```

        Args:
            run_id: Run identifier to cancel.
            reason: Exact cancellation cause transported to the target run.

        Returns:
            None: Cancellation is requested asynchronously.

        Notes:
            Supported causes are enforced by the run manager. Cancellation may
            not be immediate; scheduler termination is best-effort.
        """
        observer = resolve_operation_observer()
        span = await observer.start_span(
            service="runner",
            operation="cancel_run",
            request={"run_id": run_id, "reason": reason},
            tags=["runner", "cancel"],
            metadata={"target_run_id": run_id},
        )
        try:
            await self.inspect_run(run_id)
            await self.run_manager.cancel_run(run_id, reason=reason)
            await span.finish(response={"cancelled": True}, metadata={"target_run_id": run_id})
        except Exception as exc:
            await span.fail(exc, metadata={"target_run_id": run_id})
            raise

    async def cancellation_reason(self) -> str:
        """Return the exact cancellation cause for the current bound run.

        Examples:
            Read the cause after cancellation:
            ```python
            reason = await runner().cancellation_reason()
            ```

            Read an unbound facade:
            ```python
            assert await unbound_runner.cancellation_reason() == ""
            ```

        Args:
            None.

        Returns:
            str: Exact stored cause, or an empty string when unavailable.

        Notes:
            This method does not request cancellation or mutate its handle.
        """

        if not self.current_run_id:
            return ""
        handle = await get_run_cancellation_registry().get(self.current_run_id)
        return "" if handle is None else str(handle.cancel_reason or "")

    async def is_cancel_requested(self) -> bool:
        if not self.current_run_id:
            return False
        handle = await get_run_cancellation_registry().get(self.current_run_id)
        return bool(handle and handle.is_cancel_requested())

    async def raise_if_cancel_requested(self) -> None:
        if not self.current_run_id:
            return
        handle = await get_run_cancellation_registry().get(self.current_run_id)
        if handle is not None:
            handle.raise_if_cancel_requested()

    async def thread_cancel_event(self) -> Event:
        if not self.current_run_id:
            raise RuntimeError("RunFacade.thread_cancel_event() requires a bound current run id.")
        handle = await get_run_cancellation_registry().create(self.current_run_id)
        return handle.thread_cancel_event()
