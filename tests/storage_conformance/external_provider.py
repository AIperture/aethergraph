"""Deterministic filesystem-free external provider used by shared conformance tests."""

from __future__ import annotations

from dataclasses import dataclass

from test_storage_focused_protocols import (
    _ArtifactRepository,
    _BlobStore,
    _EventStore,
    _SearchBackend,
    _StateStore,
)

from aethergraph.storage.contracts import (
    ObservationLLMSummaryQuery,
    ObservationLLMSummaryRecord,
    ObservationTraceSummaryQuery,
    ObservationTraceSummaryRecord,
    StorageBundle,
    StorageCapabilities,
    StorageCapability,
    StorageConfigurationError,
    StorageHealth,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
)

EXTERNAL_PROVIDER_NAME = "test.external"


@dataclass(frozen=True, slots=True)
class StoreHandle:
    name: str
    resource: object


@dataclass(slots=True)
class SharedResource:
    close_calls: int = 0
    closed: bool = False


@dataclass(slots=True)
class DeterministicExternalObservationRepository:
    """Filesystem-free typed aggregate surface for external-provider conformance."""

    trace_queries: list[ObservationTraceSummaryQuery]
    llm_queries: list[ObservationLLMSummaryQuery]

    def __init__(self) -> None:
        self.trace_queries = []
        self.llm_queries = []

    async def summarize_traces(
        self,
        query: ObservationTraceSummaryQuery,
    ) -> ObservationTraceSummaryRecord:
        self.trace_queries.append(query)
        return ObservationTraceSummaryRecord(
            span_count=2,
            error_count=1,
            total_duration_ms=12,
            trace_id_count=2,
            trace_ids=("trace-a",),
            trace_ids_truncated=True,
            top_failing_services={"runner": 1},
            latest_error_at=query.occurred_at_or_after,
        )

    async def summarize_llm_calls(
        self,
        query: ObservationLLMSummaryQuery,
    ) -> ObservationLLMSummaryRecord:
        self.llm_queries.append(query)
        return ObservationLLMSummaryRecord(
            total_calls=2,
            total_prompt_tokens=5,
            total_completion_tokens=3,
            total_tokens=8,
            error_count=1,
            model_count=2,
            by_model={"model-a": 1},
            by_model_truncated=True,
        )


class DeterministicExternalBundle:
    provider_name = EXTERNAL_PROVIDER_NAME
    capabilities = StorageCapabilities.of(
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
        StorageCapability.ATOMIC_COMPARE_AND_SET,
        StorageCapability.ORDERED_APPEND,
        StorageCapability.MONOTONIC_CURSORS,
        StorageCapability.BLOB_STREAMING,
        StorageCapability.BLOB_RANGE_READ,
        StorageCapability.SEARCH_STRUCTURAL,
        StorageCapability.SEARCH_SEMANTIC,
        StorageCapability.SEARCH_LEXICAL,
        StorageCapability.SEARCH_HYBRID,
        StorageCapability.READ_ONLY_OPEN,
        StorageCapability.HEALTH,
    )
    format_version = 1

    def __init__(
        self,
        mode: StorageOpenMode,
        *,
        ready: bool,
        close_failures: int,
    ) -> None:
        self.mode = mode
        self.resource = SharedResource()
        self.ready = ready
        self.close_failures = close_failures
        self.events = _EventStore()
        self.memory_events = _EventStore()
        self.state = _StateStore()
        self.blobs = _BlobStore()
        self.artifacts = _ArtifactRepository(self.blobs)
        self.search = _SearchBackend()
        self.observations = DeterministicExternalObservationRepository()
        for name in (
            "kv",
            "documents",
            "auth_grants",
            "auth_invites",
            "registry_manifests",
            "runs",
            "run_results",
            "sessions",
            "continuations",
            "continuation_leases",
            "triggers",
            "ingress_idempotency",
            "external_session_bindings",
            "inbound_events",
            "semantic_events",
            "runtime_output",
        ):
            setattr(self, name, StoreHandle(name=name, resource=self.resource))

    async def health(self) -> StorageHealth:
        if self.resource.closed:
            return StorageHealth(ready=False, detail="closed")
        return StorageHealth(ready=self.ready, detail="ready" if self.ready else "unavailable")

    async def close(self) -> None:
        if self.resource.closed:
            return
        self.resource.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise StorageHealthError("injected external flush failure")
        self.resource.closed = True


class DeterministicExternalProvider:
    name = EXTERNAL_PROVIDER_NAME

    def __init__(self, *, ready: bool = True, close_failures: int = 0) -> None:
        self.ready = ready
        self.close_failures = close_failures
        self.open_calls = 0
        self.bundles: list[DeterministicExternalBundle] = []

    def validate_config(self, selection: StorageProviderSelection) -> None:
        if selection.provider != self.name:
            raise StorageConfigurationError("external selection does not match provider")
        unknown = set(selection.config) - {"endpoint", "credential_ref"}
        if unknown:
            raise StorageConfigurationError(f"unknown external provider keys: {sorted(unknown)}")
        if selection.config.get("endpoint") != "memory://external-conformance":
            raise StorageConfigurationError("external endpoint is invalid")
        credential_ref = selection.config.get("credential_ref")
        if not isinstance(credential_ref, str) or not credential_ref.startswith("secret://"):
            raise StorageConfigurationError("external credential_ref is invalid")

    def open(self, request: StorageOpenRequest) -> StorageBundle:
        self.validate_config(request.selection)
        if request.expected_format_version != self.format_version:
            raise StorageConfigurationError("external format version is unsupported")
        self.open_calls += 1
        bundle = DeterministicExternalBundle(
            request.mode,
            ready=self.ready,
            close_failures=self.close_failures,
        )
        self.bundles.append(bundle)
        return bundle  # type: ignore[return-value]

    @property
    def format_version(self) -> int:
        return DeterministicExternalBundle.format_version
