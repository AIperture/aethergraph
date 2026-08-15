"""Public host boundary for an in-process AetherGraph runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import ExternalIdentity, HostManifest
from aethergraph.contracts.services.memory import ExternalResourceChangedEvent
from aethergraph.core.runtime.inspection import RuntimeInspectionService
from aethergraph.core.runtime.run_types import RunOrigin, RunRecord
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.observability.runtime_output import (
    RuntimeOutputCaptureHost,
    enable_runtime_output_capture,
)
from aethergraph.services.channel.choices import normalize_choice_reply
from aethergraph.services.channel.resources import ArtifactIngressScope, ResourceStager
from aethergraph.services.container.default_container import (
    DefaultContainer,
    build_default_container,
)
from aethergraph.services.integration import (
    EventLogSemanticEventStore,
    InteractionResolutionError,
    InteractionResolver,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
    install_integration_ingress,
)
from aethergraph.services.scope.scope import Scope

from .contracts import (
    RuntimeArtifactRecord,
    RuntimeArtifactScope,
    RuntimeGraphRegistration,
    RuntimeIdentity,
    RuntimeModelProfile,
    RuntimeOpenRequest,
    RuntimeRegistrationSnapshot,
    RuntimeRunRecord,
    RuntimeRunRequest,
    RuntimeRunStatus,
    RuntimeSemanticEvent,
    RuntimeStagedArtifact,
)
from .errors import RuntimeGraphLoadError, RuntimeInteractionError, RuntimeNotReadyError


def _public_record(record: RunRecord) -> RuntimeRunRecord:
    return RuntimeRunRecord(
        run_id=record.run_id,
        graph_id=record.graph_id,
        session_id=record.session_id,
        status=getattr(record.status, "value", str(record.status)),
        error=record.error,
        started_at=record.started_at,
        finished_at=record.finished_at,
        tags=tuple(record.tags),
        metadata=dict(record.meta or {}),
    )


class EmbeddedRuntime:
    """Own one in-process runtime without exposing its dependency container."""

    def __init__(self, container: DefaultContainer) -> None:
        """Bind the public host boundary to one internally constructed container.

        Examples:
            Construct through the supported factory:
            ```python
            runtime = open_embedded_runtime(request)
            ```

            Close the runtime after use:
            ```python
            await runtime.close()
            ```

        Args:
            container: Internal service container owned exclusively by this runtime.

        Returns:
            None.

        Notes:
            Hosts construct instances through `open_embedded_runtime`; the container
            argument is an implementation seam, not a public integration object.
        """
        self._container = container
        self._output_capture: RuntimeOutputCaptureHost | None = None
        self._semantic_stores: dict[str, EventLogSemanticEventStore] = {}
        self._integration: RuntimeIntegration | None = None
        self._active_run_ids: set[str] = set()
        self._closed = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Activate this runtime's services for scoped framework operations.

        Examples:
            Import a graph under the runtime context:
            ```python
            with runtime.activate():
                importlib.import_module("generated.entrypoint")
            ```

            Execute a custom framework operation:
            ```python
            with runtime.activate():
                perform_operation()
            ```

        Args:
            None.

        Returns:
            Iterator[None]: Context manager that restores the prior services on exit.

        Notes:
            Normal Hosts should prefer the higher-level methods on this class.
        """
        self._ensure_open()
        with use_services(self._container):
            yield

    def load_graph(
        self,
        *,
        module_name: str,
        symbol_name: str,
        graph_id: str,
        module_search_path: Path | None = None,
    ) -> RuntimeGraphRegistration:
        """Import and validate one declared graph inside this runtime.

        Examples:
            Load an installed graph module:
            ```python
            registration = runtime.load_graph(
                module_name="my_agent.entrypoint",
                symbol_name="run_agent",
                graph_id="run_agent",
            )
            ```

            Load a compiled graph from an immutable source directory:
            ```python
            registration = runtime.load_graph(
                module_name="generated.entrypoint",
                symbol_name="main",
                graph_id="compiled_graph",
                module_search_path=Path("build/src"),
            )
            ```

        Args:
            module_name: Importable module containing the graph declaration.
            symbol_name: Callable symbol expected in the imported module.
            graph_id: Exact graph-function identity expected in the registry.
            module_search_path: Optional immutable source directory to prepend once.

        Returns:
            RuntimeGraphRegistration: Validated public registration identity.

        Notes:
            The callable and registry implementation remain private to the runtime.
        """
        self._ensure_open()
        if module_search_path is not None:
            path = str(module_search_path.resolve(strict=True))
            if path not in sys.path:
                sys.path.insert(0, path)
        with self.activate():
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name, None)
            graph_fn = current_registry().get_graphfn(name=graph_id)
        if not callable(symbol) or graph_fn is None:
            raise RuntimeGraphLoadError(
                f"Module {module_name!r} did not register declared graph {graph_id!r}."
            )
        return RuntimeGraphRegistration(
            module_name=module_name,
            symbol_name=symbol_name,
            graph_id=graph_id,
        )

    def install_semantic_channel(self, *, name: str, deployment_id: str) -> None:
        """Install one canonical semantic-event Channel adapter.

        Examples:
            Install Studio test delivery:
            ```python
            runtime.install_semantic_channel(name="studio", deployment_id="studio-test")
            ```

            Install another Host-owned semantic route:
            ```python
            runtime.install_semantic_channel(name="host", deployment_id="deployment-1")
            ```

        Args:
            name: Channel adapter name used by graph code.
            deployment_id: Stable Host deployment identity written to semantic events.

        Returns:
            None.

        Notes:
            Semantic and runtime events share the runtime's canonical event log.
        """
        self._ensure_open()
        store = EventLogSemanticEventStore(self._container.eventlog)
        self._container.channels.register_adapter(
            name,
            SemanticEventChannelAdapter(
                emitter=SemanticEventEmitter(deployment_id=deployment_id, store=store)
            ),
        )
        self._semantic_stores[deployment_id] = store

    def install_integration(self, manifest: HostManifest) -> RuntimeIntegration:
        """Install one immutable canonical integration ingress boundary.

        Examples:
            Install an AG UI manifest:
            ```python
            integration = runtime.install_integration(manifest)
            ```

            Reuse the returned capability for session binding:
            ```python
            session_id = await integration.bind_session(...)
            ```

        Args:
            manifest: Immutable Host deployment and route authority.

        Returns:
            RuntimeIntegration: Public accept, binding, and event capability.

        Notes:
            Reinstallation is rejected and coordinator stores remain private.
        """
        self._ensure_open()
        if self._integration is not None:
            raise RuntimeError("Integration ingress is already installed.")
        coordinator = install_integration_ingress(
            container=self._container,
            manifest=manifest,
        )
        self._semantic_stores[manifest.deployment_id] = coordinator.semantic_events
        self._integration = RuntimeIntegration(self, coordinator)
        return self._integration

    def enable_output_capture(self, *, tags: Sequence[str] = ()) -> None:
        """Enable runtime-scoped stdout and stderr capture.

        Examples:
            Enable default capture:
            ```python
            runtime.enable_output_capture()
            ```

            Tag captured Studio test output:
            ```python
            runtime.enable_output_capture(tags=("studio-test",))
            ```

        Args:
            tags: Stable tags appended to captured runtime-output events.

        Returns:
            None.

        Notes:
            Capture is process-global but reference counted and closed by this runtime.
        """
        self._ensure_open()
        if self._output_capture is not None:
            raise RuntimeError("Runtime output capture is already enabled.")
        self._output_capture = enable_runtime_output_capture(self._container, tags=tuple(tags))

    async def submit(self, request: RuntimeRunRequest) -> RuntimeRunRecord:
        """Persist and schedule one graph run through the public boundary.

        Examples:
            Submit a local run:
            ```python
            record = await runtime.submit(
                RuntimeRunRequest(graph_id="agent", inputs={"message": "Hello"})
            )
            ```

            Submit a session-bound Host run:
            ```python
            record = await runtime.submit(
                RuntimeRunRequest(
                    graph_id="agent",
                    inputs={"message": "Continue"},
                    session_id="session-1",
                )
            )
            ```

        Args:
            request: Closed run request with identity and scheduling metadata.

        Returns:
            RuntimeRunRecord: Stable public view of the admitted run.

        Notes:
            Internal run-store records and the run manager never cross this boundary.
        """
        manager = self._require_run_manager()
        try:
            origin = RunOrigin(request.origin)
        except ValueError as exc:
            raise ValueError(f"Unsupported run origin: {request.origin}") from exc
        with self.activate():
            record = await manager.submit_run(
                graph_id=request.graph_id,
                inputs=dict(request.inputs),
                run_id=request.run_id,
                session_id=request.session_id,
                tags=list(request.tags),
                identity=RequestIdentity(
                    user_id=request.identity.user_id,
                    org_id=request.identity.org_id,
                    mode=request.identity.mode,
                ),
                origin=origin,
                agent_id=request.agent_id,
                app_id=request.app_id,
                app_name=request.app_name,
                run_config=dict(request.run_config),
            )
        self._active_run_ids.add(record.run_id)
        return _public_record(record)

    async def cancel(
        self, run_id: str, *, reason: str = "user_requested"
    ) -> RuntimeRunRecord | None:
        """Request cancellation for one exact run.

        Examples:
            Cancel an active run:
            ```python
            record = await runtime.cancel("run-1")
            ```

            Preserve parent-cancellation semantics:
            ```python
            record = await runtime.cancel("run-child", reason="parent_cancelled")
            ```

        Args:
            run_id: Exact runtime run identity.
            reason: Supported semantic cancellation cause.

        Returns:
            RuntimeRunRecord | None: Current public record when the run exists.

        Notes:
            Cancellation is best effort and physical termination remains scheduler-owned.
        """
        with self.activate():
            record = await self._require_run_manager().cancel_run(run_id, reason=reason)
        return None if record is None else _public_record(record)

    async def wait(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
    ) -> RuntimeRunStatus:
        """Wait for one run and return its durable public result.

        Examples:
            Wait without a timeout:
            ```python
            result = await runtime.wait("run-1")
            ```

            Bound a Host wait:
            ```python
            result = await runtime.wait("run-1", timeout_s=30.0)
            ```

        Args:
            run_id: Exact runtime run identity.
            timeout_s: Optional maximum wait duration in seconds.

        Returns:
            RuntimeRunStatus: Terminal public status and durable output.

        Notes:
            Unknown run identities retain the run manager's existing wait semantics.
        """
        with self.activate():
            await self._require_run_manager().wait_run(
                run_id,
                timeout_s=timeout_s,
                return_outputs=True,
            )
        result = await self.status(run_id)
        if result is None:
            raise RuntimeError(f"Run disappeared after completion: {run_id}")
        return result

    async def status(self, run_id: str) -> RuntimeRunStatus | None:
        """Read canonical run status, durable output, and runtime diagnostics.

        Examples:
            Poll an active run:
            ```python
            status = await runtime.status("run-1")
            ```

            Read durable output after process-independent completion:
            ```python
            output = (await runtime.status("run-1")).output
            ```

        Args:
            run_id: Exact runtime run identity.

        Returns:
            RuntimeRunStatus | None: Public status or `None` for an unknown run.

        Notes:
            Terminal reads flush pending captured output before returning.
        """
        self._ensure_open()
        manager = self._require_run_manager()
        record = await manager.get_record(run_id)
        if record is None:
            return None
        status = getattr(record.status, "value", str(record.status))
        if status in {"succeeded", "failed", "canceled"}:
            if self._output_capture is not None:
                await self._output_capture.flush_run(run_id)
            self._active_run_ids.discard(run_id)
        output: Mapping[str, Any] | None = None
        result_store = self._container.run_result_store
        if status == "succeeded" and result_store is not None:
            result = await result_store.get(run_id)
            output = None if result is None else dict(result.outputs)
        inspection = await RuntimeInspectionService(
            run_manager=manager,
            state_store=self._container.state_store,
        ).inspect(run_id)
        diagnostics: tuple[Mapping[str, Any], ...] = ()
        run_error_info: Mapping[str, Any] | None = None
        if inspection is not None:
            run_error_info = inspection.run_error_info
            diagnostics = tuple(
                {
                    "node_id": node.node_id,
                    "tool_name": node.tool_name,
                    "status": node.status,
                    "error": node.error,
                    "error_info": node.error_info,
                }
                for node in inspection.node_diagnostics
            )
        return RuntimeRunStatus(
            record=_public_record(record),
            output=output,
            run_error_info=run_error_info,
            node_diagnostics=diagnostics,
        )

    async def respond_to_interaction(
        self,
        *,
        session_id: str,
        interaction_id: str,
        response_kind: str,
        text: str | None = None,
        choice: Any = None,
    ) -> bool:
        """Resolve and resume one exact public interaction.

        Examples:
            Resume a text request:
            ```python
            applied = await runtime.respond_to_interaction(
                session_id="session-1",
                interaction_id="interaction-1",
                response_kind="text",
                text="Continue",
            )
            ```

            Resume a choice request:
            ```python
            applied = await runtime.respond_to_interaction(
                session_id="session-1",
                interaction_id="interaction-2",
                response_kind="choice",
                choice="approve",
            )
            ```

        Args:
            session_id: Exact AG session that owns the open interaction.
            interaction_id: Public interaction identity emitted to the Host.
            response_kind: Either `text` or `choice`.
            text: Optional user-authored response text.
            choice: Optional raw choice value for choice normalization.

        Returns:
            bool: `True` after an exact interaction was resumed.

        Notes:
            Missing or mismatched interactions raise `RuntimeInteractionError`;
            continuation tokens never leave this boundary.
        """
        self._ensure_open()
        expected_kinds = (
            {"approval", "choice"}
            if response_kind == "choice"
            else {"user_input", "user_input_or_files"}
        )
        if response_kind not in {"text", "choice"}:
            raise ValueError(f"Unsupported interaction response kind: {response_kind}")
        try:
            resolved = await InteractionResolver(self._container.cont_store).resolve_exact(
                session_id=session_id,
                interaction_id=interaction_id,
                expected_kinds=expected_kinds,
            )
        except InteractionResolutionError as exc:
            raise RuntimeInteractionError(code=exc.code, message=str(exc)) from exc
        continuation = resolved.continuation
        payload: dict[str, Any] = {
            "text": text,
            "attachments": [],
            "interaction_id": resolved.interaction_id,
        }
        if response_kind == "choice":
            payload.update(
                normalize_choice_reply(
                    prompt=continuation.prompt,
                    raw_choice=choice,
                    raw_text=text,
                )
            )
        with self.activate():
            await self._container.resume_router.resume(
                run_id=continuation.run_id,
                node_id=continuation.node_id,
                token=continuation.token,
                payload=payload,
            )
        return True

    async def query_events(
        self,
        *,
        run_ids: Sequence[str],
        order_dir: str = "asc",
        limit: int | None = None,
        engine: bool = False,
        **filters: Any,
    ) -> tuple[Mapping[str, Any], ...]:
        """Query canonical runtime or Engine events for exact run membership.

        Examples:
            Read runtime output for one run:
            ```python
            rows = await runtime.query_events(
                run_ids=("run-1",), kinds=("runtime.console.output",)
            )
            ```

            Read Engine tool activity for a complete turn:
            ```python
            rows = await runtime.query_events(
                run_ids=turn_run_ids,
                engine=True,
                kinds=("tool.call.started", "tool.call.finished"),
            )
            ```

        Args:
            run_ids: Exact physical run identities included in the query.
            order_dir: Ascending or descending canonical cursor order.
            limit: Optional maximum merged result count.
            engine: Whether to query the Engine event log instead of runtime events.
            **filters: Event-log filters such as scope, cursor, and kinds.

        Returns:
            tuple[Mapping[str, Any], ...]: Cursor-ordered event rows.

        Notes:
            The Host never receives an event-log store or constructs storage paths.
        """
        self._ensure_open()
        if order_dir not in {"asc", "desc"}:
            raise ValueError("order_dir must be 'asc' or 'desc'.")
        event_log = self._container.eventlog
        if engine:
            observability = self._container.observability
            event_log = None if observability is None else observability.engine_event_log
            if event_log is None:
                raise RuntimeNotReadyError("Canonical Engine event log is unavailable.")
        ranked: list[tuple[int, int, int, Mapping[str, Any]]] = []
        for run_rank, run_id in enumerate(run_ids):
            rows = await event_log.query(
                **filters,
                run_id=run_id,
                order_dir=order_dir,
                limit=limit,
            )
            for row_rank, row in enumerate(rows):
                ranked.append((int(row["_row_id"]), run_rank, row_rank, row))
        descending = order_dir == "desc"
        ranked.sort(
            key=lambda item: (
                -item[0] if descending else item[0],
                item[1],
                item[2],
            )
        )
        result = tuple(item[3] for item in ranked)
        return result if limit is None else result[:limit]

    async def list_semantic_events(
        self,
        *,
        deployment_id: str,
        session_id: str,
        after_cursor: int | None = None,
        limit: int | None = None,
    ) -> tuple[RuntimeSemanticEvent, ...]:
        """Read semantic events installed for one Host deployment and session.

        Examples:
            Read complete semantic history:
            ```python
            events = await runtime.list_semantic_events(
                deployment_id="studio-test", session_id="session-1"
            )
            ```

            Continue after a durable cursor:
            ```python
            events = await runtime.list_semantic_events(
                deployment_id="studio-test",
                session_id="session-1",
                after_cursor=42,
            )
            ```

        Args:
            deployment_id: Previously installed semantic deployment identity.
            session_id: Exact AG session identity.
            after_cursor: Optional exclusive shared-log cursor.
            limit: Optional maximum result count.

        Returns:
            tuple[RuntimeSemanticEvent, ...]: Cursor-ordered public semantic events.

        Notes:
            A deployment must be installed through `install_semantic_channel` first.
        """
        self._ensure_open()
        store = self._semantic_stores.get(deployment_id)
        if store is None:
            raise RuntimeNotReadyError(f"Semantic deployment {deployment_id!r} is not installed.")
        history = await store.list_session(
            deployment_id=deployment_id,
            session_id=session_id,
            after_cursor=after_cursor,
            limit=limit,
        )
        return tuple(
            RuntimeSemanticEvent(
                cursor=item.cursor,
                kind=item.event.kind.value,
                event=item.event.model_dump(mode="json"),
            )
            for item in history
        )

    def observability_reader(self) -> Any:
        """Return the supported historical-observability reader capability.

        Examples:
            Create an Engine projection adapter:
            ```python
            reader = runtime.observability_reader()
            ```

            Query a Host-specific presenter through the reader:
            ```python
            sessions = await runtime.observability_reader().list_traces(limit=20)
            ```

        Args:
            None.

        Returns:
            Any: Public observability facade configured for this runtime workspace.

        Notes:
            This capability is read-only; raw observation and event stores remain private.
        """
        self._ensure_open()
        if self._container.observability is None:
            raise RuntimeNotReadyError("Observability is unavailable.")
        return self._container.observability

    def model_profile(self, name: str) -> RuntimeModelProfile:
        """Read immutable values for one configured model profile.

        Examples:
            Read the default profile:
            ```python
            profile = runtime.model_profile("default")
            ```

            Verify a named summarizer profile:
            ```python
            assert runtime.model_profile("summarizer").name == "summarizer"
            ```

        Args:
            name: Exact configured profile name.

        Returns:
            RuntimeModelProfile: Immutable name, provider, and model values.

        Notes:
            The mutable settings object and LLM service remain private.
        """
        self._ensure_open()
        settings = self._container.settings
        if settings is None:
            raise RuntimeNotReadyError("Runtime settings are unavailable.")
        profiles = {"default": settings.llm.default, **dict(settings.llm.profiles or {})}
        profile = profiles.get(name)
        if profile is None:
            raise KeyError(f"Unknown runtime model profile: {name}")
        return RuntimeModelProfile(
            name=name,
            provider=str(profile.provider or "") or None,
            model=str(profile.model or "") or None,
        )

    def registration_snapshot(
        self,
        *,
        agent_ids: Sequence[str],
        graph_ids: Sequence[str],
    ) -> RuntimeRegistrationSnapshot:
        """Read immutable registry facts needed for Host startup validation.

        Examples:
            Validate one Agent and graph:
            ```python
            snapshot = runtime.registration_snapshot(
                agent_ids=("general",), graph_ids=("general_graph",)
            )
            ```

            Detect a missing graph:
            ```python
            assert "missing" not in snapshot.registered_graph_ids
            ```

        Args:
            agent_ids: Exact Agent identities to inspect.
            graph_ids: Exact graph-function identities to inspect.

        Returns:
            RuntimeRegistrationSnapshot: Immutable metadata and registered graph IDs.

        Notes:
            The mutable global registry never crosses this boundary.
        """
        self._ensure_open()
        registry = self._container.registry
        agents = {
            agent_id: registry.get_meta(nspace="agent", name=agent_id) for agent_id in agent_ids
        }
        registered: set[str] = set()
        for graph_id in graph_ids:
            try:
                graph = registry.get_graphfn(graph_id)
            except KeyError:
                graph = None
            if graph is not None:
                registered.add(graph_id)
        return RuntimeRegistrationSnapshot(
            agent_metadata={
                key: None if value is None else dict(value) for key, value in agents.items()
            },
            registered_graph_ids=frozenset(registered),
        )

    async def stage_artifact(
        self,
        *,
        identity: RuntimeIdentity,
        data: bytes,
        name: str,
        mime: str | None,
        file_id: str | None,
        scope: RuntimeArtifactScope,
        labels: Mapping[str, Any] | None = None,
    ) -> RuntimeStagedArtifact:
        """Stage Host-authenticated bytes in the runtime artifact owner.

        Examples:
            Stage a current source buffer:
            ```python
            artifact = await runtime.stage_artifact(
                identity=identity, data=b"print('hi')", name="main.py",
                mime="text/x-python", file_id="buffer-1", scope=scope,
            )
            ```

            Stage a produced text suggestion:
            ```python
            artifact = await runtime.stage_artifact(
                identity=identity, data=text.encode(), name="suggestion.md",
                mime="text/markdown", file_id=None, scope=scope,
            )
            ```

        Args:
            identity: Trusted Host identity used for artifact scope.
            data: Exact authenticated bytes.
            name: Safe display filename.
            mime: Optional declared media type.
            file_id: Optional stable inbound identity.
            scope: Provider-neutral artifact ingress scope.
            labels: Optional bounded provenance labels.

        Returns:
            RuntimeStagedArtifact: Stable artifact identity, size, and URI.

        Notes:
            Artifact stores and facades remain private to the runtime.
        """
        self._ensure_open()
        resource = await ResourceStager(
            container=self._container,
            identity=RequestIdentity(
                user_id=identity.user_id,
                org_id=identity.org_id,
                mode=identity.mode,
            ),
        ).stage_bytes(
            data,
            name=name,
            mime=mime,
            file_id=file_id,
            scope=ArtifactIngressScope(
                source=scope.source,
                session_id=scope.session_id,
                run_id=scope.run_id,
                channel_key=scope.channel_key,
                conversation_id=scope.conversation_id,
                graph_id=scope.graph_id,
                node_id=scope.node_id,
                tool_name=scope.tool_name,
                tool_version=scope.tool_version,
            ),
            labels=dict(labels or {}),
        )
        if not resource.artifact_id:
            raise RuntimeError("Artifact staging produced no artifact identity.")
        return RuntimeStagedArtifact(
            artifact_id=resource.artifact_id,
            size_bytes=int(resource.size or len(data)),
            uri=resource.uri,
        )

    async def artifact(self, artifact_id: str) -> RuntimeArtifactRecord | None:
        """Read immutable metadata for one artifact identity.

        Examples:
            Read an existing artifact:
            ```python
            artifact = await runtime.artifact("artifact-1")
            ```

            Detect a missing artifact:
            ```python
            assert await runtime.artifact("missing") is None
            ```

        Args:
            artifact_id: Exact runtime artifact identity.

        Returns:
            RuntimeArtifactRecord | None: Immutable metadata when present.

        Notes:
            Authorization remains the embedding Host's responsibility.
        """
        self._ensure_open()
        artifact = await self._container.artifact_index.get(artifact_id)
        if artifact is None:
            return None
        return RuntimeArtifactRecord(
            artifact_id=str(getattr(artifact, "artifact_id", artifact_id)),
            uri=str(getattr(artifact, "uri", "") or "") or None,
            name=str(getattr(artifact, "name", "") or "") or None,
            mime=str(getattr(artifact, "mime", None) or getattr(artifact, "mimetype", None) or "")
            or None,
            size_bytes=int(
                getattr(artifact, "bytes", None) or getattr(artifact, "size", None) or 0
            ),
            sha256=str(getattr(artifact, "sha256", "") or "") or None,
            org_id=str(getattr(artifact, "org_id", "") or "") or None,
            labels=dict(getattr(artifact, "labels", None) or {}),
        )

    async def load_artifact_text(self, uri: str) -> str:
        """Load UTF-8 artifact content from one authorized runtime URI.

        Examples:
            Load a suggestion:
            ```python
            text = await runtime.load_artifact_text("artifact://suggestion")
            ```

            Load a staged upload after metadata authorization:
            ```python
            text = await runtime.load_artifact_text(artifact.uri)
            ```

        Args:
            uri: Exact artifact-store URI from an authorized metadata record.

        Returns:
            str: Stored UTF-8 text.

        Notes:
            Callers must authorize metadata before invoking this content read.
        """
        self._ensure_open()
        return await self._container.artifacts.load_text(uri)

    async def append_external_resource_change(
        self,
        *,
        identity: RuntimeIdentity,
        session_id: str,
        change: ExternalResourceChangedEvent,
    ) -> None:
        """Idempotently append one external-resource event to session memory.

        Examples:
            Append a workspace change:
            ```python
            await runtime.append_external_resource_change(
                identity=identity, session_id="session-1", change=change
            )
            ```

            Retry the same event safely:
            ```python
            await runtime.append_external_resource_change(
                identity=identity, session_id="session-1", change=change
            )
            ```

        Args:
            identity: Trusted Host identity for memory isolation.
            session_id: Exact durable AG session identity.
            change: Strict external-resource change contract.

        Returns:
            None.

        Notes:
            The event ID is checked before append; memory stores remain private.
        """
        self._ensure_open()
        scope = Scope(
            org_id=identity.org_id,
            user_id=identity.user_id,
            session_id=session_id,
            graph_id="agstudio.workspace_changes",
            node_id="workspace_event_bridge",
            memory_level="session",
        )
        memory = self._container.memory_factory.for_session(
            run_id=f"external:{change.event_id}",
            graph_id="agstudio.workspace_changes",
            node_id="workspace_event_bridge",
            session_id=session_id,
            scope=scope,
        )
        if await memory.get_event(change.event_id) is None:
            await memory.append_external_resource_change(change)

    def observability_capture_mode(self) -> str:
        """Read the immutable prompt-capture mode selected at runtime open.

        Examples:
            Verify Studio test manifest capture:
            ```python
            assert runtime.observability_capture_mode() == "manifest"
            ```

            Record a Host diagnostic without reading settings:
            ```python
            mode = runtime.observability_capture_mode()
            ```

        Args:
            None.

        Returns:
            str: Configured observability capture mode.

        Notes:
            This returns one immutable scalar rather than the mutable settings tree.
        """
        self._ensure_open()
        settings = self._container.settings
        if settings is None:
            raise RuntimeNotReadyError("Runtime settings are unavailable.")
        return str(settings.llm.observability.capture_mode)

    async def close(self) -> None:
        """Cancel owned work and release runtime-owned process resources.

        Examples:
            Close during ordinary Host shutdown:
            ```python
            await runtime.close()
            ```

            Call close again safely after partial startup failure:
            ```python
            await runtime.close()
            await runtime.close()
            ```

        Args:
            None.

        Returns:
            None.

        Notes:
            The operation is idempotent and never closes resources owned by another runtime.
        """
        if self._closed:
            return
        failures: list[str] = []
        for run_id in tuple(self._active_run_ids):
            try:
                await self._require_run_manager().cancel_run(run_id)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"cancel {run_id}: {exc}")
        self._active_run_ids.clear()
        if self._output_capture is not None:
            try:
                await self._output_capture.close()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"runtime output: {exc}")
            self._output_capture = None
        seen: set[int] = set()
        for name in ("eventlog", "run_store", "session_store", "state_store"):
            value = getattr(self._container, name, None)
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            close = getattr(value, "close", None)
            if not callable(close):
                continue
            try:
                closed = close()
                if inspect.isawaitable(closed):
                    await closed
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {exc}")
        self._closed = True
        if failures:
            raise RuntimeError("Embedded runtime close failed: " + "; ".join(failures))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Embedded runtime is closed.")

    def _require_run_manager(self) -> Any:
        self._ensure_open()
        if self._container.run_manager is None:
            raise RuntimeNotReadyError("Run manager is unavailable.")
        return self._container.run_manager


def open_embedded_runtime(request: RuntimeOpenRequest) -> EmbeddedRuntime:
    """Construct one owned in-process runtime from a closed Host request.

    Examples:
        Open a runtime with no transport adapters:
        ```python
        runtime = open_embedded_runtime(
            RuntimeOpenRequest(root=Path("workspace"), settings=settings)
        )
        ```

        Install Host extensions before importing graphs:
        ```python
        runtime = open_embedded_runtime(
            RuntimeOpenRequest(
                root=Path("workspace"),
                settings=settings,
                extensions={"studio.resource_provider": provider},
            )
        )
        ```

    Args:
        request: Closed workspace, settings, adapter, and extension configuration.

    Returns:
        EmbeddedRuntime: Owned public runtime boundary.

    Notes:
        Extensions are installed before any Host graph module is imported.
    """
    container = build_default_container(
        root=str(request.root),
        cfg=request.settings,
        channel_adapters=dict(request.channel_adapters),
    )
    container.ext_services.update(dict(request.extensions))
    required = (
        "channels",
        "cont_store",
        "eventlog",
        "observability",
        "resume_router",
        "run_manager",
        "state_store",
    )
    missing = [name for name in required if getattr(container, name, None) is None]
    if missing:
        raise RuntimeNotReadyError(
            "Embedded runtime is missing required services: " + ", ".join(missing)
        )
    return EmbeddedRuntime(container)


class RuntimeIntegration:
    """Public capability wrapper for one installed integration coordinator."""

    def __init__(self, runtime: EmbeddedRuntime, coordinator: Any) -> None:
        """Bind a private canonical coordinator to deliberate Host operations.

        Examples:
            Obtain a wrapper through runtime installation:
            ```python
            integration = runtime.install_integration(manifest)
            ```

            Keep the coordinator private:
            ```python
            assert not hasattr(integration, "binding_store")
            ```

        Args:
            runtime: Owning public embedded runtime.
            coordinator: Internal installed integration coordinator.

        Returns:
            None.

        Notes:
            Hosts never construct this class directly.
        """
        self._runtime = runtime
        self._coordinator = coordinator

    async def accept(
        self,
        *,
        verified: Any,
        envelope: Any,
        root_admission_callback: Any = None,
    ) -> Any:
        """Accept one canonical verified ingress envelope.

        Examples:
            Accept a root turn:
            ```python
            receipt = await integration.accept(verified=verified, envelope=envelope)
            ```

            Persist Host admission before scheduling:
            ```python
            receipt = await integration.accept(
                verified=verified, envelope=envelope,
                root_admission_callback=persist_run,
            )
            ```

        Args:
            verified: Authenticated integration authority.
            envelope: Closed canonical ingress envelope.
            root_admission_callback: Optional pre-scheduling Host callback.

        Returns:
            Any: Canonical public ingress receipt.

        Notes:
            Idempotency, dispatch, and interaction routing remain coordinator-owned.
        """
        with self._runtime.activate():
            return await self._coordinator.accept(
                verified=verified,
                envelope=envelope,
                root_admission_callback=root_admission_callback,
            )

    async def bind_session(
        self,
        *,
        route_id: str,
        external_identity: ExternalIdentity,
        binding_id: str,
        session_id: str,
        now: datetime,
    ) -> str:
        """Resolve or create one exact route-scoped integration session.

        Examples:
            Bind a Studio thread:
            ```python
            session_id = await integration.bind_session(
                route_id="studio.general", external_identity=identity,
                binding_id="binding-1", session_id="session-1", now=now,
            )
            ```

            Resolve the same identity idempotently:
            ```python
            assert await integration.bind_session(**same_scope) == session_id
            ```

        Args:
            route_id: Exact installed manifest route identity.
            external_identity: Canonical external conversation identity.
            binding_id: Candidate durable binding identity.
            session_id: Candidate durable AG session identity.
            now: Authoritative binding timestamp.

        Returns:
            str: Persisted AG session identity.

        Notes:
            Route resolution never falls back to another profile.
        """
        route = next(
            (
                item
                for item in self._coordinator.manifest.integration_routes
                if item.route_id == route_id
            ),
            None,
        )
        if route is None:
            raise RuntimeError(f"Integration route is not installed: {route_id}")
        resolution = await self._coordinator.binding_store.get_or_create(
            route=route,
            external_identity=external_identity,
            build_id=self._coordinator.manifest.build_id,
            binding_id=binding_id,
            ag_session_id=session_id,
            now=now,
        )
        return resolution.binding.ag_session_id


__all__ = ["EmbeddedRuntime", "RuntimeIntegration", "open_embedded_runtime"]
