"""Provider-neutral historical observability opening for manifested workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

from aethergraph.config.storage_provider import StorageProviderSettings
from aethergraph.observability.canonical_inspection import (
    CanonicalInspectionReader,
    RunStatusResolver,
)
from aethergraph.observability.canonical_service import (
    CanonicalObservationService,
    ProviderObservationService,
)
from aethergraph.observability.inspection import (
    ObservabilityIdentity,
    ObservabilityUnavailableError,
    ObservabilityWorkspaceError,
)
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.observability.prompt_store import content_hash
from aethergraph.server.security.redaction import canonical_json
from aethergraph.services.canonical_storage_scope import merge_storage_scope
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.control.canonical_stores import project_canonical_run_record
from aethergraph.storage.composition import StorageComposition
from aethergraph.storage.contracts import (
    EventQuery,
    LLMCallQuery,
    ObservationCaptureMode,
    ObservationScopeManagementQuery,
    PageRequest,
    RunQuery,
    SortDirection,
    StorageBundle,
    StorageCapability,
    StorageError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageScope,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry
from aethergraph.storage.providers.local_sqlite import (
    LocalStorageProvider,
    read_local_workspace_manifest,
)
from aethergraph.storage.providers.local_sqlite.manifest import LOCAL_PROVIDER_NAME

_HISTORICAL_PAGE_SIZE = 1_000
_HISTORICAL_RESULT_LIMIT = 10_000
_READ_ONLY_CONTINUATION_SECRET = hashlib.sha256(
    b"aethergraph.read-only-observability.workspace.v1"
).digest()


class _UnavailableHistoricalSecrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise ObservabilityUnavailableError(
            f"Historical provider secret resolution is unavailable for {reference!r}"
        )


class _CanonicalObservabilityFacade:
    def __init__(
        self,
        *,
        composition: StorageComposition | None,
        owner_scope: StorageScope,
        identity: ObservabilityIdentity,
        run_statuses: Mapping[str, str],
        runtime_bundle: StorageBundle | None = None,
        reader: CanonicalInspectionReader | None = None,
    ) -> None:
        if (composition is None) == (runtime_bundle is None):
            raise ValueError("exactly one observability bundle owner is required")
        self._composition = composition
        self._runtime_bundle = runtime_bundle
        self._owner_scope = owner_scope
        self._identity = identity
        self._run_status_overrides = dict(run_statuses)
        self._reader = reader

    async def close(self) -> None:
        """Close an owned historical provider composition.

        Intro:
            Releases the one prepared or ready read-only bundle. A live runtime facade
            borrows its already-open bundle, so close deliberately leaves runtime
            lifecycle with `EmbeddedRuntime`.

        Examples:
            Close after a Studio read:
                ```python
                await facade.close()
                ```

            Close an unused opener result:
                ```python
                facade = open_observability_workspace(workspace)
                await facade.close()
                ```

            Leave a live runtime bundle owned by its runtime:
                ```python
                await runtime.observability_reader().close()
                await runtime.close()
                ```

        Args:
            None.

        Returns:
            None: Owned history is closed; a borrowed live bundle is unchanged.

        Notes:
            Close never selects, opens, retries through, or closes a borrowed provider.
        """
        if self._composition is not None:
            await self._composition.close()

    async def list_inspect_traces(self, **filters: Any):
        """List canonical trace observations through the stable Studio boundary.

        Intro:
            Lazily admits the read-only provider and delegates one bounded query to
            the canonical inspection reader.

        Examples:
            List one run:
                ```python
                page = await facade.list_inspect_traces(run_id="run-1")
                ```

            Continue a provider page:
                ```python
                page = await facade.list_inspect_traces(cursor=page.next_cursor)
                ```

        Args:
            **filters: Canonical trace filters and provider cursor pagination values.

        Returns:
            TraceEventListResponse: Stable generic Inspect trace page.

        Notes:
            Deprecated `app_id` remains bounded compatibility metadata only.
        """
        return await (await self._inspection()).list_traces(**filters)

    async def list_inspect_llm_calls(self, **filters: Any):
        """List metadata-only canonical LLM calls for Studio Inspect.

        Intro:
            Uses the selected provider's promoted filters and opaque cursor without
            hydrating retained prompt or response bodies.

        Examples:
            List one run:
                ```python
                page = await facade.list_inspect_llm_calls(run_id="run-1")
                ```

            Filter a model:
                ```python
                page = await facade.list_inspect_llm_calls(model="gpt-test")
                ```

        Args:
            **filters: Canonical LLM list filters and provider cursor values.

        Returns:
            LLMCallListResponse: Stable metadata-only LLM page.

        Notes:
            Full retained content remains available only through exact call detail.
        """
        return await (await self._inspection()).list_llm_calls(**filters)

    async def get_inspect_llm_call(
        self,
        call_id: str,
        *,
        required_run_id: str | None = None,
    ):
        """Hydrate one exact canonical LLM call for Studio Inspect.

        Intro:
            Applies owner, identity, and optional run constraints before provider
            detail hydration.

        Examples:
            Read one call:
                ```python
                call = await facade.get_inspect_llm_call("call-1")
                ```

            Require run ownership:
                ```python
                call = await facade.get_inspect_llm_call(
                    "call-1", required_run_id="run-1"
                )
                ```

        Args:
            call_id: Exact canonical LLM call identity.
            required_run_id: Optional exact run that must own the call.

        Returns:
            LLMCallRecord: Stable capture-policy-aware call detail.

        Notes:
            Missing or identity-hidden calls use the canonical not-found error.
        """
        return await (await self._inspection()).get_llm_call(
            call_id,
            required_run_id=required_run_id,
        )

    async def list_inspect_logs(self, **filters: Any):
        """List canonical structured logs through the stable Studio boundary.

        Intro:
            Applies promoted provider filters and page-bounded status enrichment.

        Examples:
            List error logs:
                ```python
                page = await facade.list_inspect_logs(level="error")
                ```

            List one run:
                ```python
                page = await facade.list_inspect_logs(run_id="run-1")
                ```

        Args:
            **filters: Canonical log filters and provider cursor pagination values.

        Returns:
            InspectLogListResponse: Stable generic Inspect log page.

        Notes:
            Catalog run-status overrides affect enrichment only, not provider scope.
        """
        return await (await self._inspection()).list_logs(**filters)

    async def list_inspect_agent_events(self, **filters: Any):
        """List canonical Agent events through the stable Studio boundary.

        Intro:
            Delegates exact scope, event-type, time, and cursor filters to canonical
            observation storage.

        Examples:
            List one run:
                ```python
                page = await facade.list_inspect_agent_events(run_id="run-1")
                ```

            Filter one event type:
                ```python
                page = await facade.list_inspect_agent_events(
                    event_type="planning.started"
                )
                ```

        Args:
            **filters: Canonical Agent-event filters and provider cursor values.

        Returns:
            AgentEventListResponse: Stable generic Inspect Agent-event page.

        Notes:
            This is generic AG inspection; Engine semantics remain in Engine.
        """
        return await (await self._inspection()).list_agent_events(**filters)

    async def list_suppressed_scopes(self) -> dict[str, set[str]]:
        """List explicitly hidden or deleted canonical observation scopes.

        Intro:
            Drains bounded provider pages up to the historical reader ceiling and
            projects only session, run, and trace identities used by Engine.

        Examples:
            Read all suppression markers:
                ```python
                suppressed = await facade.list_suppressed_scopes()
                ```

            Check one run:
                ```python
                hidden = "run-1" in suppressed["run_id"]
                ```

        Args:
            None.

        Returns:
            dict[str, set[str]]: Suppressed IDs grouped by stable scope name.

        Notes:
            The operation never reads or deletes authoritative run/session history.
        """
        bundle = await self._bundle()
        scope = self._query_scope()
        result = {"session_id": set(), "run_id": set(), "trace_id": set()}
        if scope is None:
            return result
        cursor = None
        count = 0
        while True:
            page = await bundle.observations.query_scope_management(
                ObservationScopeManagementQuery(
                    scope=scope,
                    page=PageRequest(limit=_HISTORICAL_PAGE_SIZE, cursor=cursor),
                )
            )
            count += len(page.items)
            if count > _HISTORICAL_RESULT_LIMIT:
                raise ObservabilityUnavailableError(
                    "Historical observation management exceeds the bounded read ceiling"
                )
            for record in page.items:
                if not (record.hidden or record.deleted):
                    continue
                if record.scope.session_id:
                    result["session_id"].add(record.scope.session_id)
                if record.scope.run_id:
                    result["run_id"].add(record.scope.run_id)
                if record.trace_id:
                    result["trace_id"].add(record.trace_id)
            if page.next_cursor is None:
                return result
            cursor = page.next_cursor

    async def list_runs(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List authoritative canonical runs visible to the workspace identity.

        Intro:
            Converts provider run pages into the stable runtime mapping consumed by
            Engine while enforcing a finite historical read ceiling.

        Examples:
            Read the first page:
                ```python
                runs = await facade.list_runs(limit=100, offset=0)
                ```

            Read Engine's bounded historical window:
                ```python
                runs = await facade.list_runs(limit=10_000, offset=0)
                ```

        Args:
            limit: Positive maximum number of stable run mappings to return.
            offset: Non-negative number of provider-ordered rows to skip.

        Returns:
            list[dict[str, Any]]: Visible stable run mappings in provider order.

        Notes:
            Offset plus limit may not exceed 10,000; provider access remains cursor-based.
        """
        _validate_window(limit, offset)
        bundle = await self._bundle()
        scope = self._query_scope()
        if scope is None:
            return []
        wanted = limit + offset
        records = []
        cursor = None
        while len(records) < wanted:
            page = await bundle.runs.query(
                RunQuery(
                    scope=scope,
                    page=PageRequest(
                        limit=min(_HISTORICAL_PAGE_SIZE, wanted - len(records)),
                        cursor=cursor,
                    ),
                )
            )
            records.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return [
            asdict(project_canonical_run_record(record))
            for record in records[offset : offset + limit]
        ]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Read one authoritative canonical run visible to the identity.

        Intro:
            Performs an exact provider lookup using owner, identity, and run scope,
            then projects the stable runtime mapping used by Engine.

        Examples:
            Read one run:
                ```python
                run = await facade.get_run("run-1")
                ```

            Handle absence:
                ```python
                assert await facade.get_run("missing") is None
                ```

        Args:
            run_id: Exact canonical run identity.

        Returns:
            dict[str, Any] | None: Stable run mapping, or `None` when absent or hidden.

        Notes:
            Deprecated App identity is projected only from explicit compatibility metadata.
        """
        bundle = await self._bundle()
        scope = self._query_scope(run_id=run_id)
        if scope is None:
            return None
        record = await bundle.runs.get(scope, run_id)
        return None if record is None else asdict(project_canonical_run_record(record))

    async def list_engine_events(self, *, run_id: str) -> list[dict[str, Any]]:
        """List canonical Engine events for one run in causal storage order.

        Intro:
            Reads only the canonical memory event stream with exact owner/run scope
            and the `agent_engine` tag, draining finite provider cursor pages.

        Examples:
            Read one run's Engine events:
                ```python
                events = await facade.list_engine_events(run_id="run-1")
                ```

            Inspect the first kind:
                ```python
                first_kind = events[0]["kind"] if events else None
                ```

        Args:
            run_id: Exact canonical run identity.

        Returns:
            list[dict[str, Any]]: Stable Engine event mappings in ascending order.

        Notes:
            Engine interpretation remains outside AG; more than 10,000 rows fails closed.
        """
        bundle = await self._bundle()
        scope = self._query_scope(run_id=run_id)
        if scope is None:
            return []
        records = []
        cursor = None
        while True:
            page = await bundle.memory_events.query(
                EventQuery(
                    scope=scope,
                    tags=("agent_engine",),
                    order=SortDirection.ASCENDING,
                    page=PageRequest(limit=_HISTORICAL_PAGE_SIZE, cursor=cursor),
                )
            )
            records.extend(page.items)
            if len(records) > _HISTORICAL_RESULT_LIMIT:
                raise ObservabilityUnavailableError(
                    "Historical Engine events exceed the bounded read ceiling"
                )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return [_event_mapping(record) for record in records]

    async def hydrate_prompt_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        """Project one prompt manifest from canonical LLM correlation and detail.

        Intro:
            Resolves the indexed manifest-to-call correlation with a bounded metadata
            query, then uses the exact scoped LLM detail operation to present Engine's
            stable manifest shape.

        Examples:
            Read a captured context snapshot source:
                ```python
                manifest = await facade.hydrate_prompt_manifest("manifest-1")
                ```

            Handle an absent capture:
                ```python
                assert await facade.hydrate_prompt_manifest("missing") is None
                ```

        Args:
            manifest_id: Exact canonical prompt-manifest correlation identity.

        Returns:
            dict[str, Any] | None: Stable Engine manifest projection or `None`.

        Notes:
            No provider-private manifest, fragment, table, or path API is exposed.
        """
        bundle = await self._bundle()
        scope = self._query_scope()
        if scope is None:
            return None
        page = await bundle.observations.query_llm_calls(
            LLMCallQuery(
                scope=scope,
                prompt_manifest_ids=(manifest_id,),
                page=PageRequest(limit=2),
            )
        )
        if not page.items:
            return None
        if len(page.items) != 1 or page.next_cursor is not None:
            raise ObservabilityUnavailableError(
                "Prompt manifest identity is not uniquely correlated"
            )
        record = page.items[0]
        detail = await bundle.observations.get_llm_call(scope, record.llm_call_id)
        if detail is None or detail.record.prompt_manifest_id != manifest_id:
            raise ObservabilityUnavailableError("Prompt manifest correlation is inconsistent")
        return _prompt_manifest_mapping(detail)

    async def _inspection(self) -> CanonicalInspectionReader:
        await self._bundle()
        assert self._reader is not None
        return self._reader

    async def _bundle(self):
        if self._runtime_bundle is not None:
            return self._runtime_bundle
        assert self._composition is not None
        try:
            bundle = await self._composition.start()
        except StorageError as exc:
            raise ObservabilityUnavailableError(
                "AetherGraph historical storage is unavailable"
            ) from exc
        if self._reader is None:
            service = ProviderObservationService(
                repository=bundle.observations,
                owner_scope=self._owner_scope,
                policy=ObservationPolicy(),
            )
            self._reader = CanonicalInspectionReader(
                service,
                identity=self._identity,
                run_status_resolver=self._resolve_run_statuses,
            )
        return bundle

    async def _resolve_run_statuses(self, run_ids: set[str]) -> dict[str, str]:
        statuses = {
            run_id: self._run_status_overrides[run_id]
            for run_id in run_ids
            if run_id in self._run_status_overrides
        }
        bundle = await self._bundle()
        for run_id in run_ids - statuses.keys():
            scope = self._query_scope(run_id=run_id)
            if scope is None:
                continue
            record = await bundle.runs.get(scope, run_id)
            if record is not None:
                statuses[run_id] = record.status.value
        return statuses

    def _query_scope(self, **dimensions: str) -> StorageScope | None:
        values = dict(dimensions)
        if self._identity.mode in {"cloud", "demo"}:
            if self._identity.user_id is None:
                return None
            values["user_id"] = self._identity.user_id
            if self._identity.org_id is not None:
                values["org_id"] = self._identity.org_id
        try:
            return merge_storage_scope(self._owner_scope, **values)
        except ValueError:
            return None


def _bind_runtime_observability(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    service: CanonicalObservationService,
    identity: ObservabilityIdentity | None = None,
    run_status_resolver: RunStatusResolver | None = None,
) -> _CanonicalObservabilityFacade:
    """Bind the full observability facade to one borrowed live runtime bundle."""
    exact_identity = identity or ObservabilityIdentity()
    facade = _CanonicalObservabilityFacade(
        composition=None,
        runtime_bundle=bundle,
        owner_scope=owner_scope,
        identity=exact_identity,
        run_statuses={},
    )
    facade._reader = CanonicalInspectionReader(
        service,
        identity=exact_identity,
        run_status_resolver=run_status_resolver or facade._resolve_run_statuses,
    )
    return facade


def open_observability_workspace(
    workspace_root: str | Path,
    *,
    identity: ObservabilityIdentity | None = None,
    run_statuses: Mapping[str, str] | None = None,
) -> _CanonicalObservabilityFacade:
    """Prepare the exact manifested provider for historical observability reads.

    Intro:
        Resolves and validates one authorized workspace manifest synchronously, opens
        exactly its built-in local provider in read-only mode, and defers asynchronous
        health admission to the first facade operation.

    Examples:
        Open local historical inspection:
            ```python
            facade = open_observability_workspace(workspace_root)
            ```

        Apply a catalog status overlay:
            ```python
            facade = open_observability_workspace(
                workspace_root,
                identity=ObservabilityIdentity(mode="local"),
                run_statuses={"run-1": "failed"},
            )
            ```

    Args:
        workspace_root: Already-authorized opaque AG runtime workspace root.
        identity: Optional request identity applied to every canonical read.
        run_statuses: Optional catalog-owned status overlay for Inspect enrichment.

    Returns:
        ObservabilityFacade: Stable async read facade owning one provider.

    Notes:
        Unmanifested, malformed, unsupported, or non-local workspaces fail directly.
        No legacy layout probe, migration, alternate provider, or writable open occurs.
    """
    root = Path(workspace_root).expanduser().resolve()
    try:
        manifest = read_local_workspace_manifest(root)
        selection = StorageProviderSettings(provider=LOCAL_PROVIDER_NAME).to_selection()
        reference = selection.config["continuation_token_secret_ref"]
        assert isinstance(reference, str)
        registry = StorageProviderRegistry(
            {
                LOCAL_PROVIDER_NAME: lambda: LocalStorageProvider(
                    continuation_token_secret_ref=reference,
                    continuation_token_secret=_READ_ONLY_CONTINUATION_SECRET,
                )
            }
        )
        composition = StorageComposition(
            registry,
            frozenset(
                {
                    StorageCapability.READ_ONLY_OPEN,
                    StorageCapability.HEALTH,
                }
            ),
        )
        clock = SystemClock()
        composition.prepare(
            StorageOpenRequest(
                workspace_id=manifest.workspace_id,
                workspace_root=root,
                owner_scope=manifest.owner_scope,
                selection=selection,
                mode=StorageOpenMode.READ_ONLY,
                expected_format_version=manifest.format_version,
                clock=clock,
                secrets=_UnavailableHistoricalSecrets(),
            )
        )
    except (OSError, StorageError, ValueError) as exc:
        raise ObservabilityWorkspaceError(
            "AetherGraph manifested observability workspace could not be opened"
        ) from exc
    return _CanonicalObservabilityFacade(
        composition=composition,
        owner_scope=manifest.owner_scope,
        identity=identity or ObservabilityIdentity(),
        run_statuses=run_statuses or {},
    )


def _validate_window(limit: int, offset: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit + offset > _HISTORICAL_RESULT_LIMIT:
        raise ValueError(f"offset plus limit must not exceed {_HISTORICAL_RESULT_LIMIT}")


def _event_mapping(record: Any) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "id": record.event_id,
        "ts": record.occurred_at.timestamp(),
        "run_id": record.scope.run_id,
        "session_id": record.scope.session_id,
        "kind": record.kind,
        "text": record.text or "",
        "tags": list(record.tags),
        "data": _engine_event_data(record.payload),
    }


def _engine_event_data(payload: Any) -> dict[str, Any]:
    plain = _plain_json(payload)
    if not isinstance(plain, dict) or set(plain) != {"data"}:
        raise ObservabilityWorkspaceError(
            "Canonical Engine event payload must contain exactly one authored data envelope"
        )
    data = plain["data"]
    if not isinstance(data, dict) or "data" in data:
        raise ObservabilityWorkspaceError(
            "Canonical Engine event authored data must be one flat mapping"
        )
    return data


def _prompt_manifest_mapping(detail: Any) -> dict[str, Any]:
    record = detail.record
    attributes = _plain_json(record.observation.attributes)
    request = _plain_json(detail.captured_request)
    parts: list[dict[str, Any]] = []
    if isinstance(request, dict):
        if record.capture_mode is ObservationCaptureMode.FULL:
            body = canonical_json(request)
            parts.append(
                _manifest_part(
                    ordinal=0,
                    semantic_kind="direct_message",
                    role=None,
                    content_kind="provider_request",
                    body=body,
                )
            )
        elif record.capture_mode is ObservationCaptureMode.MANIFEST:
            for ordinal, message in enumerate(request.get("messages") or []):
                body = canonical_json(message)
                role = message.get("role") if isinstance(message, dict) else None
                semantic_kind = message.get("semantic_kind") if isinstance(message, dict) else None
                parts.append(
                    _manifest_part(
                        ordinal=ordinal,
                        semantic_kind=str(semantic_kind or "direct_message"),
                        role=str(role) if role is not None else None,
                        content_kind="prompt_message",
                        body=body,
                    )
                )
            config = request.get("provider_request_args") or {}
            parts.append(
                _manifest_part(
                    ordinal=len(parts),
                    semantic_kind="direct_message",
                    role="provider_config",
                    content_kind="provider_request_config",
                    body=canonical_json(config),
                )
            )
    return {
        "manifest_id": record.prompt_manifest_id,
        "capture_mode": record.capture_mode.value,
        "assembled_request_hash": attributes.get("assembled_request_hash"),
        "total_chars": attributes.get("prompt_chars", 0),
        "total_bytes": attributes.get("prompt_bytes", 0),
        "roles": list(attributes.get("prompt_roles") or ()),
        "parts": parts,
        "provider_request": request if isinstance(request, dict) else None,
    }


def _manifest_part(
    *,
    ordinal: int,
    semantic_kind: str,
    role: str | None,
    content_kind: str,
    body: str,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "semantic_kind": semantic_kind,
        "role": role,
        "fragment_id": content_hash(content_kind=content_kind, body=body),
        "content_kind": content_kind,
        "byte_count": len(body.encode("utf-8")),
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


# Stable Plan 1 annotation/import boundary. Historical readers own one read-only
# composition; live readers borrow one runtime-owned canonical bundle. The retired
# SQLite-owning facade module is not restored.
ObservabilityFacade = _CanonicalObservabilityFacade


__all__ = ["ObservabilityFacade", "open_observability_workspace"]
