from __future__ import annotations

import asyncio
from collections.abc import Iterable
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

from aethergraph.services.llm.correlation import complete_llm_call_correlation

from .models import (
    LLMObservationRecord,
    ObservationFilter,
    ObservationRecord,
    PurgeResult,
    StorageStats,
)
from .policy import ObservationPolicy
from .sqlite_store import SQLiteObservationStore
from .studio_translation import (
    ObservabilityIdentity,
    ObservabilityNotFoundError,
    ObservabilityUnavailableError,
    ObservabilityWorkspaceError,
    StudioTranslationPresenter,
)

if TYPE_CHECKING:
    from aethergraph.api.v1.schemas.inspect import (
        AgentEventListResponse,
        InspectLogListResponse,
        LLMCallListResponse,
        LLMCallRecord,
        TraceEventListResponse,
    )


class ActiveObservabilityScopeError(RuntimeError):
    """Reject destructive observation cleanup for active or resumable runs."""

    def __init__(self, scope_key: str, run_ids: Iterable[str]) -> None:
        self.scope_key = scope_key
        self.run_ids = tuple(sorted({str(run_id) for run_id in run_ids if run_id}))
        joined = ", ".join(self.run_ids)
        super().__init__(f"Observability scope {scope_key} has active runs: {joined}")


class ObservabilityFacade:
    """Coordinate the one supported AG observation read/write boundary.

    Intro:
        Exposes canonical observation, LLM, lifecycle, and retention operations.

    Examples:
        Append a structured observation:
        ```python
        await facade.append_observation(record)
        ```

        Purge one run after reviewing a dry run:
        ```python
        preview = await facade.delete_run_observations("run-1", dry_run=True)
        if preview.matching_observations:
            await facade.delete_run_observations("run-1")
        ```

    Args:
        store: Concrete SQLite observation store owned by this facade.
        event_log: Global AG event log for metering and custom agent events.
        engine_event_log: Canonical memory-event log for `agent_engine.*` events.
        run_store: Authoritative AG run store used for grouping and status.
        identity: Optional read identity applied by the translation presenter.
        run_statuses: Optional retained status overrides for historical reads.
        owns_event_log: Whether `close` releases the injected event log.
        owns_engine_event_log: Whether `close` releases the engine event log.
        owns_store: Whether `close` releases the observation store.

    Returns:
        ObservabilityFacade: A facade over canonical observations and prompts.

    Notes:
        This facade does not read legacy JSONL or engine trace stores.
    """

    def __init__(
        self,
        store: SQLiteObservationStore,
        *,
        event_log: Any | None = None,
        engine_event_log: Any | None = None,
        run_store: Any | None = None,
        identity: ObservabilityIdentity | None = None,
        run_statuses: dict[str, str] | None = None,
        owns_event_log: bool = False,
        owns_engine_event_log: bool = False,
        owns_store: bool = True,
    ) -> None:
        self.store = store
        self.event_log = event_log
        self.engine_event_log = engine_event_log
        self.run_store = run_store
        self.identity = identity or ObservabilityIdentity()
        self._run_statuses = dict(run_statuses or {})
        self._owns_event_log = owns_event_log
        self._owns_engine_event_log = owns_engine_event_log
        self._owns_store = owns_store

    async def close(self) -> None:
        """Release stores owned by this facade.

        Intro:
            Closes only resources whose ownership was explicitly assigned.

        Examples:
            `await facade.close()`
            `await facade.close()`

        Args:
            None.

        Returns:
            None: Owned stores are closed before completion.

        Notes:
            Repeated calls are safe.
        """
        if self._owns_event_log and self.event_log is not None:
            await self.event_log.close()
            self._owns_event_log = False
        if self._owns_engine_event_log and self.engine_event_log is not None:
            await self.engine_event_log.close()
            self._owns_engine_event_log = False
        if self._owns_store:
            await self.store.close()
            self._owns_store = False

    def for_identity(self, identity: ObservabilityIdentity) -> ObservabilityFacade:
        """Create a non-owning identity-scoped query facade.

        Intro:
            Reuses the active stores while applying one request identity.

        Examples:
            `scoped = facade.for_identity(ObservabilityIdentity(mode="local"))`
            `cloud = facade.for_identity(identity)`

        Args:
            identity: Exact identity constraints for subsequent reads.

        Returns:
            ObservabilityFacade: Non-owning scoped facade.

        Notes:
            Closing the returned facade does not close shared stores.
        """
        return ObservabilityFacade(
            self.store,
            event_log=self.event_log,
            engine_event_log=self.engine_event_log,
            run_store=self.run_store,
            identity=identity,
            run_statuses=self._run_statuses,
            owns_store=False,
        )

    async def emit(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        if capture_mode != self.store.policy.capture_mode:
            raise ValueError("LLM client capture mode does not match the observation store policy")
        await self.store.append_llm_call(record)
        complete_llm_call_correlation(
            record.llm_call_id,
            prompt_manifest_id=record.prompt_manifest_id,
        )

    async def append_observation(
        self,
        record: ObservationRecord,
        *,
        resource_links: Iterable[dict[str, Any]] = (),
    ) -> str:
        return await self.store.append_observation(record, resource_links=resource_links)

    async def get_observation(self, observation_id: str) -> dict[str, Any] | None:
        return await self.store.get_observation(observation_id)

    async def list_observations(
        self, filters: ObservationFilter | None = None, *, offset: int = 0
    ) -> list[dict[str, Any]]:
        return await self.store.list_observations(filters, offset=offset)

    async def get_llm_call(self, llm_call_id: str) -> dict[str, Any] | None:
        return await self.store.get_llm_call(llm_call_id)

    async def list_llm_calls(self, **filters: Any) -> list[dict[str, Any]]:
        return await self.store.query_llm_calls(**filters)

    async def list_inspect_traces(self, **filters: Any) -> TraceEventListResponse:
        """Project explicit service observations into the current Inspect DTO.

        Intro:
            Preserves the temporary browser contract without legacy storage.

        Examples:
            `page = await facade.list_inspect_traces(run_id="run-1")`
            `page = await facade.list_inspect_traces(status="error")`

        Args:
            **filters: Current Inspect trace filters and pagination values.

        Returns:
            TraceEventListResponse: Translated service-operation page.

        Notes:
            Generic tracing is off by default, so an empty page is valid.
        """
        return await self._presenter().list_traces(**filters)

    async def list_inspect_llm_calls(self, **filters: Any) -> LLMCallListResponse:
        """Project LLM observations into the current Inspect list DTO.

        Intro:
            Returns metadata-safe call rows through the one v2 presenter.

        Examples:
            `page = await facade.list_inspect_llm_calls(run_id="run-1")`
            `page = await facade.list_inspect_llm_calls(provider="openai")`

        Args:
            **filters: Current LLM list filters and pagination values.

        Returns:
            LLMCallListResponse: Translated LLM call page.

        Notes:
            Prompt bodies are omitted from list results.
        """
        return await self._presenter().list_llm_calls(**filters)

    async def get_inspect_llm_call(
        self, call_id: str, *, required_run_id: str | None = None
    ) -> LLMCallRecord:
        """Project one full LLM observation into the current Inspect DTO.

        Intro:
            Applies identity and optional run ownership before hydration.

        Examples:
            `item = await facade.get_inspect_llm_call("call-1")`
            `item = await facade.get_inspect_llm_call("call-1", required_run_id="run-1")`

        Args:
            call_id: Exact LLM call identity.
            required_run_id: Optional required owner run.

        Returns:
            LLMCallRecord: Capture-policy-aware call detail.

        Notes:
            Missing content is represented by capture metadata, not fabrication.
        """
        return await self._presenter().get_llm_call(call_id, required_run_id=required_run_id)

    async def list_inspect_logs(self, **filters: Any) -> InspectLogListResponse:
        """Project structured log observations into the current Logs DTO.

        Intro:
            Keeps chronological log inspection on the v2 observation store.

        Examples:
            `page = await facade.list_inspect_logs(run_id="run-1")`
            `page = await facade.list_inspect_logs(level="error")`

        Args:
            **filters: Current log filters and pagination values.

        Returns:
            InspectLogListResponse: Translated structured-log page.

        Notes:
            Ordinary logs are never converted into semantic trace spans.
        """
        return await self._presenter().list_logs(**filters)

    async def list_inspect_agent_events(self, **filters: Any) -> AgentEventListResponse:
        """Project canonical and explicit custom agent events for Inspect.

        Intro:
            Reads canonical engine events and explicit AG agent events together.

        Examples:
            `page = await facade.list_inspect_agent_events(run_id="run-1")`
            `page = await facade.list_inspect_agent_events(event_type="agent_engine.decision")`

        Args:
            **filters: Current agent-event filters and pagination values.

        Returns:
            AgentEventListResponse: Normalized event page.

        Notes:
            Resource links stay in the canonical event payload.
        """
        return await self._presenter().list_agent_events(**filters)

    async def list_trace_sessions(
        self, *, limit: int = 50, cursor: str | None = None
    ) -> dict[str, Any]:
        """List current Trace Explorer session groups from v2 data.

        Intro:
            Groups authoritative runs by session without an engine trace store.

        Examples:
            `page = await facade.list_trace_sessions()`
            `page = await facade.list_trace_sessions(limit=25, cursor="25")`

        Args:
            limit: Maximum session groups returned.
            cursor: Optional decimal offset cursor.

        Returns:
            dict[str, Any]: Current session-group page shape.

        Notes:
            Run IDs are the semantic trace IDs during the UI transition.
        """
        return await self._presenter()._list_trace_sessions(limit=limit, cursor=cursor)

    async def inspect_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Build one current Trace Explorer tree from v2 records.

        Intro:
            Composes run, event, plan, graph, and context availability at read time.

        Examples:
            `tree = await facade.inspect_trace("run-1")`
            `missing = await facade.inspect_trace("unknown")`

        Args:
            trace_id: Exact AG run identity.

        Returns:
            dict[str, Any] | None: Current trace tree or `None`.

        Notes:
            No translated tree is persisted or cached.
        """
        return await self._presenter()._inspect_trace(trace_id)

    async def get_trace_graph(self, trace_id: str) -> dict[str, Any] | None:
        """Build the observed agent/dispatch graph for one run.

        Intro:
            Includes only nodes and edges supported by canonical events.

        Examples:
            `graph = await facade.get_trace_graph("run-1")`
            `missing = await facade.get_trace_graph("unknown")`

        Args:
            trace_id: Exact AG run identity.

        Returns:
            dict[str, Any] | None: Current graph DTO or `None`.

        Notes:
            Unknown topology is omitted rather than inferred.
        """
        return await self._presenter()._get_trace_graph(trace_id)

    async def get_trace_spans(
        self,
        trace_id: str,
        *,
        kind: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Project canonical events into current Trace Explorer spans.

        Intro:
            Pairs tools and dispatches by explicit causal identities.

        Examples:
            `page = await facade.get_trace_spans("run-1")`
            `tools = await facade.get_trace_spans("run-1", kind="tool_call")`

        Args:
            trace_id: Exact AG run identity.
            kind: Optional translated span-kind filter.
            agent_id: Optional exact agent-instance filter.

        Returns:
            dict[str, Any] | None: Mapping with translated `items`, or `None`.

        Notes:
            Unrepresented legacy span kinds are honestly absent.
        """
        return await self._presenter()._get_trace_spans(trace_id, kind=kind, agent_id=agent_id)

    async def get_trace_plans(self, trace_id: str) -> dict[str, Any] | None:
        """Reconstruct retained plan versions for one run.

        Intro:
            Reads canonical plan lifecycle events in persisted order.

        Examples:
            `plans = await facade.get_trace_plans("run-1")`
            `missing = await facade.get_trace_plans("unknown")`

        Args:
            trace_id: Exact AG run identity.

        Returns:
            dict[str, Any] | None: Mapping with plan snapshots, or `None`.

        Notes:
            Only explicitly retained plan payloads are returned.
        """
        return await self._presenter()._get_trace_plans(trace_id)

    async def get_trace_context_snapshot(
        self, trace_id: str, snapshot_id: str
    ) -> dict[str, Any] | None:
        """Hydrate one prompt-manifest context snapshot for the current UI.

        Intro:
            Converts captured manifest metadata/fragments without inventing content.

        Examples:
            `snapshot = await facade.get_trace_context_snapshot("run-1", "manifest-1")`
            `missing = await facade.get_trace_context_snapshot("run-1", "unknown")`

        Args:
            trace_id: Exact AG run identity.
            snapshot_id: Exact prompt manifest identity.

        Returns:
            dict[str, Any] | None: Current context DTO or `None`.

        Notes:
            Metadata capture returns a valid empty body with capture information.
        """
        return await self._presenter()._get_context_snapshot(trace_id, snapshot_id)

    async def get_trace_agent_states(self, trace_id: str, agent_id: str) -> dict[str, Any] | None:
        """Return explicitly retained agent state history for one run.

        Intro:
            Preserves the current collection contract without synthetic snapshots.

        Examples:
            `states = await facade.get_trace_agent_states("run-1", "planner")`
            `missing = await facade.get_trace_agent_states("unknown", "planner")`

        Args:
            trace_id: Exact AG run identity.
            agent_id: Exact agent-instance identity.

        Returns:
            dict[str, Any] | None: State-history page or `None`.

        Notes:
            The result is empty when no canonical state event was retained.
        """
        return await self._presenter()._get_agent_states(trace_id, agent_id)

    async def get_usage(self, *, run_id: str | None = None) -> dict[str, int]:
        """Aggregate product-agent LLM and cache usage.

        Intro:
            Sums usage stored on canonical LLM observations.

        Examples:
            `usage = await facade.get_usage(run_id="run-1")`
            `workspace_usage = await facade.get_usage()`

        Args:
            run_id: Optional exact run scope.

        Returns:
            dict[str, int]: LLM call and token/cache totals.

        Notes:
            This is agent-scoped product usage, not engine trace metering.
        """
        if self.event_log is None:
            raise RuntimeError("AG event log is required for usage reads")
        rows = await self.event_log.query(kinds=["meter.llm"], run_id=run_id, limit=None)
        if self.engine_event_log is None:
            raise RuntimeError("Canonical engine event log is required for usage reads")
        tool_rows = await self.engine_event_log.query(
            kinds=["agent_engine.tool_call"],
            tags=["agent_engine"],
            run_id=run_id,
            limit=None,
        )
        totals = {
            "llm_calls": len(rows),
            "tool_calls": len(tool_rows),
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "uncached_input_tokens": 0,
        }
        for row in rows:
            totals["input_tokens"] += int(row.get("input_tokens") or row.get("prompt_tokens") or 0)
            totals["output_tokens"] += int(
                row.get("output_tokens") or row.get("completion_tokens") or 0
            )
            for name in (
                "cache_read_tokens",
                "cache_write_tokens",
                "uncached_input_tokens",
            ):
                totals[name] += int(row.get(name) or 0)
        return totals

    async def list_resource_events(
        self, resource_key: str, *, relation: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Query all v2 evidence linked to one resource identity.

        Intro:
            Composes indexed AG observation links and canonical engine-event tags.

        Examples:
            `events = await facade.list_resource_events("artifact:a-1")`
            `outputs = await facade.list_resource_events("artifact:a-1", relation="output")`

        Args:
            resource_key: Exact namespaced resource identity.
            relation: Optional canonical relation filter.

        Returns:
            dict[str, list[dict[str, Any]]]: Observation and engine-event evidence.

        Notes:
            Tool result bodies are never parsed to discover resource references.
        """
        if self.engine_event_log is None:
            raise RuntimeError("Canonical engine event log is required for resource-event reads")
        tags = ["agent_engine", f"resource:{resource_key}"]
        if relation is not None:
            tags.append(f"resource_relation:{relation}")
        engine_events = await self.engine_event_log.query(
            tags=tags,
            limit=None,
            order_dir="desc",
        )
        return {
            "observations": await self.store.list_resource_observations(
                resource_key, relation=relation
            ),
            "engine_events": engine_events,
        }

    async def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        return await self.store.list_observations(ObservationFilter(trace_id=trace_id))

    async def list_traces(self, filters: ObservationFilter | None = None) -> list[str]:
        rows = await self.store.list_observations(filters)
        return sorted({str(row["trace_id"]) for row in rows if row.get("trace_id")})

    async def update_trace_management(self, scope_key: str, **changes: Any) -> dict[str, Any]:
        return await self.store.update_trace_management(scope_key, **changes)

    async def delete_observation(self, observation_id: str) -> PurgeResult:
        return await self.store.delete_observation(observation_id)

    async def delete_trace(self, trace_id: str, *, dry_run: bool = False) -> PurgeResult:
        """Delete AG-owned observations for one semantic trace/run.

        Intro:
            Purges observation payloads while retaining canonical runtime history.

        Examples:
            `preview = await facade.delete_trace("run-1", dry_run=True)`
            `result = await facade.delete_trace("run-1")`

        Args:
            trace_id: Semantic trace identity, equal to the AG run ID in v2.
            dry_run: Whether to report impact without deleting data.

        Returns:
            PurgeResult: Estimated or completed deletion accounting.

        Notes:
            Active or resumable runs are refused for non-dry-run deletion.
        """
        if not dry_run:
            await self._ensure_runs_inactive(
                f"trace:{trace_id}",
                await self._runs_for_run_id(trace_id),
            )
        return await self.store.delete_trace(trace_id, dry_run=dry_run)

    async def delete_run_observations(self, run_id: str, *, dry_run: bool = False) -> PurgeResult:
        """Delete AG-owned observations for one run.

        Intro:
            Removes capture data without deleting the authoritative run record.

        Examples:
            `preview = await facade.delete_run_observations("run-1", dry_run=True)`
            `result = await facade.delete_run_observations("run-1")`

        Args:
            run_id: Exact authoritative AG run identity.
            dry_run: Whether to report impact without deleting data.

        Returns:
            PurgeResult: Estimated or completed deletion accounting.

        Notes:
            Active or resumable runs are refused for non-dry-run deletion.
        """
        if not dry_run:
            await self._ensure_runs_inactive(
                f"run:{run_id}",
                await self._runs_for_run_id(run_id),
            )
        return await self.store.delete_run_observations(run_id, dry_run=dry_run)

    async def delete_session_observations(
        self, session_id: str, *, dry_run: bool = False
    ) -> PurgeResult:
        """Delete AG-owned observations for one completed session.

        Intro:
            Purges observation payloads and hides the session from observability views.

        Examples:
            `preview = await facade.delete_session_observations("session-1", dry_run=True)`
            `result = await facade.delete_session_observations("session-1")`

        Args:
            session_id: Exact authoritative AG session identity.
            dry_run: Whether to report impact without deleting data.

        Returns:
            PurgeResult: Estimated or completed deletion accounting.

        Notes:
            Any active or resumable run causes non-dry-run deletion to fail atomically.
        """
        if not dry_run:
            await self._ensure_runs_inactive(
                f"session:{session_id}",
                await self._runs_for_session(session_id),
            )
        return await self.store.delete_session_observations(session_id, dry_run=dry_run)

    async def delete_sessions_observations(
        self,
        session_ids: Iterable[str],
    ) -> list[PurgeResult]:
        """Delete AG-owned observations for completed sessions atomically by eligibility.

        Intro:
            Validates every requested session before performing the first deletion.

        Examples:
            `results = await facade.delete_sessions_observations(["s-1", "s-2"])`
            `results = await facade.delete_sessions_observations([])`

        Args:
            session_ids: Session identities to validate and purge.

        Returns:
            list[PurgeResult]: One completed deletion result per unique session.

        Notes:
            Canonical run, session, event, and artifact history is not deleted.
        """
        normalized = tuple(
            dict.fromkeys(
                str(session_id).strip() for session_id in session_ids if str(session_id).strip()
            )
        )
        runs_by_session = {
            session_id: await self._runs_for_session(session_id) for session_id in normalized
        }
        for session_id, runs in runs_by_session.items():
            await self._ensure_runs_inactive(f"session:{session_id}", runs)
        return [
            await self.store.delete_session_observations(session_id) for session_id in normalized
        ]

    async def purge_observations(
        self, filters: ObservationFilter, *, dry_run: bool = True
    ) -> PurgeResult:
        return await self.store.purge_observations(filters, dry_run=dry_run)

    async def get_storage_stats(self):
        return await self.store.get_storage_stats()

    async def garbage_collect_fragments(self) -> int:
        return await self.store.garbage_collect_fragments()

    async def compact_storage(self) -> StorageStats:
        """Compact the observation database after destructive maintenance.

        Intro:
            Returns unused main/WAL capacity to the filesystem on demand.

        Examples:
            `stats = await facade.compact_storage()`
            `physical_bytes = stats.physical_bytes`

        Args:
            None.

        Returns:
            StorageStats: Storage accounting after SQLite compaction.

        Notes:
            Call from an administrative maintenance window, not an active loop.
        """
        return await self.store.compact_storage()

    async def _runs_for_run_id(self, run_id: str) -> list[Any]:
        if self.run_store is None:
            raise ObservabilityUnavailableError(
                "Run store is required for destructive observability operations"
            )
        run = await self.run_store.get(run_id)
        if run is None:
            return []
        if not self._run_is_visible(run):
            raise ObservabilityNotFoundError("Observability run was not found")
        return [run]

    async def _runs_for_session(self, session_id: str) -> list[Any]:
        if self.run_store is None:
            raise ObservabilityUnavailableError(
                "Run store is required for destructive observability operations"
            )
        runs = list(
            await self.run_store.list(
                session_id=session_id,
                limit=10_000,
                offset=0,
            )
        )
        visible = [run for run in runs if self._run_is_visible(run)]
        if runs and len(visible) != len(runs):
            raise ObservabilityNotFoundError("Observability session was not found")
        return visible

    async def _ensure_runs_inactive(
        self,
        scope_key: str,
        runs: Iterable[Any],
    ) -> None:
        active = [
            self._run_value(run, "run_id")
            for run in runs
            if self._run_status(run) not in {"succeeded", "failed", "canceled"}
        ]
        if active:
            raise ActiveObservabilityScopeError(scope_key, active)

    def _run_is_visible(self, run: Any) -> bool:
        if self.identity.mode not in {"cloud", "demo"}:
            return True
        if not self.identity.user_id:
            return False
        if self._run_value(run, "user_id") != self.identity.user_id:
            return False
        return not self.identity.org_id or (self._run_value(run, "org_id") == self.identity.org_id)

    @classmethod
    def _run_status(cls, run: Any) -> str:
        value = cls._run_value(run, "status")
        return str(getattr(value, "value", value) or "")

    @staticmethod
    def _run_value(run: Any, key: str) -> Any:
        return run.get(key) if isinstance(run, dict) else getattr(run, key, None)

    def _presenter(self) -> StudioTranslationPresenter:
        async def resolve_run_statuses(run_ids: set[str]) -> dict[str, str]:
            statuses = {
                run_id: self._run_statuses[run_id]
                for run_id in run_ids
                if run_id in self._run_statuses
            }
            if self.run_store is None:
                return statuses
            for run_id in run_ids - statuses.keys():
                run = await self.run_store.get(run_id)
                if run is None:
                    continue
                value = run.get("status") if isinstance(run, dict) else getattr(run, "status", None)
                statuses[run_id] = str(getattr(value, "value", value) or "")
            return statuses

        return StudioTranslationPresenter(
            event_log=self.event_log,
            engine_event_log=self.engine_event_log,
            store=self.store,
            run_store=self.run_store,
            identity=self.identity,
            run_status_resolver=resolve_run_statuses,
        )


class _ReadOnlySQLiteRunStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def get(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get, run_id)

    async def list(self, *, limit: int = 100, offset: int = 0, **_: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list, limit, offset)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    def _get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _list(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT data_json FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


def open_active_observability_facade(
    store: SQLiteObservationStore,
    *,
    event_log: Any,
    engine_event_log: Any | None,
    run_store: Any,
) -> ObservabilityFacade:
    """Compose the active-container observability query boundary.

    Intro:
        Binds the live observation, global event, canonical memory-event, and
        run stores without copying semantic events.

    Examples:
        Compose a fully observable runtime:
        ```python
        facade = open_active_observability_facade(
            store,
            event_log=global_log,
            engine_event_log=memory_log,
            run_store=runs,
        )
        ```

        Compose a runtime whose canonical engine stream is unavailable:
        ```python
        facade = open_active_observability_facade(
            store,
            event_log=global_log,
            engine_event_log=None,
            run_store=runs,
        )
        ```

    Args:
        store: Active canonical observation store.
        event_log: Active global AG event log.
        engine_event_log: Active canonical memory-event log, when configured.
        run_store: Active authoritative run store.

    Returns:
        ObservabilityFacade: One active read/write boundary.

    Notes:
        The returned facade owns the observation store, not either shared event log.
    """
    return ObservabilityFacade(
        store,
        event_log=event_log,
        engine_event_log=engine_event_log,
        run_store=run_store,
    )


def open_observability_facade(
    workspace_root: str | Path,
    *,
    read_only: bool = True,
    policy: ObservationPolicy | None = None,
    identity: ObservabilityIdentity | None = None,
    run_statuses: dict[str, str] | None = None,
) -> ObservabilityFacade:
    """Open the canonical observation store for one workspace.

    Intro:
        Resolves the four required v2 workspace stores without creating a
        legacy read path or copying canonical engine events.

    Examples:
        Open historical data read-only:
        ```python
        facade = open_observability_facade(".runtime/build-1")
        ```

        Open a writable local workspace:
        ```python
        facade = open_observability_facade(".runtime/local", read_only=False)
        ```

    Args:
        workspace_root: Existing AetherGraph workspace root.
        read_only: Whether writes must be rejected.
        policy: Optional capture policy for writable use.
        identity: Optional identity scope for translated reads.
        run_statuses: Optional retained run-status overrides.

    Returns:
        ObservabilityFacade: Facade over observations, canonical events, and runs.

    Notes:
        Missing global-event, memory-event, observation, or run databases fail
        directly; there is no legacy fallback.
    """
    root = Path(workspace_root).expanduser().resolve()
    event_path = root / "events" / "events.db"
    engine_event_path = root / "memory_events" / "events.db"
    observation_path = root / "events" / "observability.db"
    run_path = root / "runs" / "runs.db"
    if not root.is_dir() or not all(
        path.is_file() for path in (event_path, engine_event_path, observation_path, run_path)
    ):
        raise ObservabilityWorkspaceError("AetherGraph v2 observability workspace was not found")
    from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

    event_log = SqliteEventLog(str(event_path), read_only=read_only)
    engine_event_log = SqliteEventLog(str(engine_event_path), read_only=read_only)
    return ObservabilityFacade(
        SQLiteObservationStore(
            observation_path,
            read_only=read_only,
            policy=policy,
        ),
        event_log=event_log,
        engine_event_log=engine_event_log,
        run_store=_ReadOnlySQLiteRunStore(run_path),
        identity=identity,
        run_statuses=run_statuses,
        owns_event_log=True,
        owns_engine_event_log=True,
    )
