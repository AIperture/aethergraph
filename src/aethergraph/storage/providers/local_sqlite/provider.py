"""Coherent provider and lifecycle-owned bundle for current local workspaces."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from aethergraph.contracts.services.llm import EmbeddingClientProtocol

from ...contracts import (
    StorageCapabilities,
    StorageCapability,
    StorageClock,
    StorageConfigurationError,
    StorageHealth,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
)
from .artifact_repository import LocalArtifactRepository
from .blob_store import LocalBlobStore
from .continuation_repositories import (
    LocalContinuationLeaseRepository,
    LocalContinuationRepository,
)
from .control_repositories import (
    LocalRunRepository,
    LocalRunResultRepository,
    LocalSessionRepository,
)
from .database import LocalCheckpoint, LocalDatabaseRole, LocalSQLiteDatabase
from .event_store import LocalEventStore
from .integration_repositories import (
    LocalExternalSessionBindingRepository,
    LocalIngressIdempotencyRepository,
)
from .manifest import (
    LOCAL_PROVIDER_NAME,
    LOCAL_STORAGE_FORMAT_VERSION,
    open_local_workspace_manifest,
    update_local_workspace_lifecycle,
)
from .observation_repository import LocalObservationRepository
from .search_backend import LocalSearchBackend
from .state_store import LocalStateStore
from .stream_repositories import (
    LocalInboundEventRepository,
    LocalRuntimeOutputSink,
    LocalSemanticEventRepository,
)
from .supporting_stores import LocalDocumentStore, LocalKeyValueStore
from .trigger_repository import LocalTriggerRepository

_CONFIG_KEYS = frozenset(
    {
        "busy_timeout_ms",
        "continuation_token_secret_ref",
        "durability",
        "runtime_output_max_pending_frames",
        "search_max_candidates",
    }
)


@dataclass(frozen=True, slots=True)
class _LocalProviderConfig:
    busy_timeout_ms: int
    continuation_token_secret_ref: str
    durability: str
    runtime_output_max_pending_frames: int
    search_max_candidates: int


@dataclass(slots=True)
class _LocalBundleLifecycle:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    databases_closed: bool = False
    closed: bool = False


@dataclass(frozen=True, slots=True)
class LocalStorageBundle:
    """Immutable typed store collection owning one local workspace lifecycle."""

    capabilities: StorageCapabilities
    mode: StorageOpenMode
    events: LocalEventStore
    memory_events: LocalEventStore
    state: LocalStateStore
    blobs: LocalBlobStore
    artifacts: LocalArtifactRepository
    search: LocalSearchBackend
    kv: LocalKeyValueStore
    documents: LocalDocumentStore
    auth_grants: LocalKeyValueStore
    auth_invites: LocalKeyValueStore
    registry_manifests: LocalDocumentStore
    runs: LocalRunRepository
    run_results: LocalRunResultRepository
    sessions: LocalSessionRepository
    continuations: LocalContinuationRepository
    continuation_leases: LocalContinuationLeaseRepository
    triggers: LocalTriggerRepository
    observations: LocalObservationRepository
    ingress_idempotency: LocalIngressIdempotencyRepository
    external_session_bindings: LocalExternalSessionBindingRepository
    inbound_events: LocalInboundEventRepository
    semantic_events: LocalSemanticEventRepository
    runtime_output: LocalRuntimeOutputSink
    _databases: tuple[LocalSQLiteDatabase, ...] = field(repr=False, compare=False)
    _workspace_root: Path = field(repr=False, compare=False)
    _clock: StorageClock = field(repr=False, compare=False)
    _lifecycle: _LocalBundleLifecycle = field(
        default_factory=_LocalBundleLifecycle,
        repr=False,
        compare=False,
    )
    provider_name: str = field(default=LOCAL_PROVIDER_NAME, init=False)
    format_version: int = field(default=LOCAL_STORAGE_FORMAT_VERSION, init=False)

    async def health(self) -> StorageHealth:
        """Return readiness for every shared local database role.

        The check is serialized with maintenance and close. It queries only the three
        already-open provider-owned databases and never selects or opens another store.

        Examples:
            Check a ready local bundle:
                ```python
                assert (await bundle.health()).ready
                ```

            Inspect a closed local bundle:
                ```python
                await bundle.close()
                assert not (await bundle.health()).ready
                ```

        Args:
            None.

        Returns:
            StorageHealth: Ready only when all database quick checks report `ok`.

        Notes:
            Closed bundles return a bounded non-ready result instead of reopening
            provider resources.
        """
        async with self._lifecycle.lock:
            if self._lifecycle.closed or self._lifecycle.databases_closed:
                return StorageHealth(ready=False, detail="closed")
            for database in self._databases:
                status = await database.health()
                if not status.ready:
                    return StorageHealth(
                        ready=False,
                        detail=f"{database.role.value}: {status.detail}",
                    )
            return StorageHealth(ready=True, detail="ready")

    async def checkpoint(self) -> Mapping[LocalDatabaseRole, LocalCheckpoint]:
        """Checkpoint every writable SQLite role and record maintenance time.

        Checkpoints run in stable control, events, and search order while bundle close
        is excluded. The lifecycle manifest is updated only after every role succeeds.

        Examples:
            Run explicit local maintenance:
                ```python
                results = await bundle.checkpoint()
                ```

            Inspect the events-role result:
                ```python
                events = results[LocalDatabaseRole.EVENTS]
                assert events.log_pages >= events.checkpointed_pages
                ```

        Args:
            None.

        Returns:
            Mapping[LocalDatabaseRole, LocalCheckpoint]: Immutable results by role.

        Notes:
            Read-only bundles fail with `StorageReadOnlyError` through the centralized
            database policy. Partial maintenance never advances the manifest timestamp.
        """
        async with self._lifecycle.lock:
            if self._lifecycle.closed or self._lifecycle.databases_closed:
                raise StorageHealthError("Local storage bundle is closed")
            results: dict[LocalDatabaseRole, LocalCheckpoint] = {}
            for database in self._databases:
                results[database.role] = await database.checkpoint()
            await asyncio.to_thread(
                update_local_workspace_lifecycle,
                self._workspace_root,
                clean_shutdown=False,
                last_maintenance_at=self._clock.now(),
            )
            return MappingProxyType(results)

    async def close(self) -> None:
        """Flush accepted output and close every local resource exactly once.

        Writable close establishes the final output barrier, checkpoints every SQLite
        role, closes all database handles, and only then marks the manifest clean. A
        failure leaves the exact close retryable without selecting another provider.

        Examples:
            Close after runtime shutdown:
                ```python
                await bundle.close()
                ```

            Retry an interrupted close:
                ```python
                with suppress(StorageHealthError):
                    await bundle.close()
                await bundle.close()
                ```

        Args:
            None.

        Returns:
            None: All resources are closed and writable lifecycle state is durable.

        Notes:
            Read-only close never edits the workspace manifest. Services must not close
            individual databases or stores owned by this bundle.
        """
        async with self._lifecycle.lock:
            if self._lifecycle.closed:
                return
            if not self._lifecycle.databases_closed:
                await self.runtime_output._flush_all()
                if self.mode is StorageOpenMode.READ_WRITE:
                    for database in self._databases:
                        await database.checkpoint()
                errors: list[Exception] = []
                for database in reversed(self._databases):
                    try:
                        await database.close()
                    except Exception as exc:  # pragma: no cover - driver close is stable
                        errors.append(exc)
                if errors:
                    if len(errors) == 1:
                        raise errors[0]
                    raise ExceptionGroup("Local database close failures", errors)
                self._lifecycle.databases_closed = True
            if self.mode is StorageOpenMode.READ_WRITE:
                await asyncio.to_thread(
                    update_local_workspace_lifecycle,
                    self._workspace_root,
                    clean_shutdown=True,
                    last_maintenance_at=self._clock.now(),
                )
            self._lifecycle.closed = True


class LocalStorageProvider:
    """Open current-format local SQLite bundles with injected runtime dependencies."""

    name = LOCAL_PROVIDER_NAME

    def __init__(
        self,
        *,
        continuation_token_secret_ref: str,
        continuation_token_secret: str | bytes,
        embedder: EmbeddingClientProtocol | None = None,
    ) -> None:
        """Capture resolved composition dependencies without persisting credentials.

        Trusted composition obtains continuation material before registering this
        provider factory. The built-in AG path derives it synchronously from already
        resolved authentication material; async-native callers may resolve an opaque
        reference first. Synchronous provider open then validates that the captured
        reference exactly matches the immutable provider selection.

        Examples:
            Construct a lexical-only local provider:
                ```python
                provider = LocalStorageProvider(
                    continuation_token_secret_ref="secret://continuations",
                    continuation_token_secret=secret_bytes,
                )
                ```

            Construct a semantic-capable local provider:
                ```python
                provider = LocalStorageProvider(
                    continuation_token_secret_ref="secret://continuations",
                    continuation_token_secret=secret_bytes,
                    embedder=embedding_client,
                )
                ```

        Args:
            continuation_token_secret_ref: Exact configured reference resolved by composition.
            continuation_token_secret: Resolved HMAC material of at least 32 bytes.
            embedder: Optional async embedding dependency captured by the registry factory.

        Returns:
            None: A provider factory product ready for side-effect-free config validation.

        Notes:
            The provider never resolves or derives credentials. Secret bytes are
            retained in process only and never written to the manifest, SQLite
            databases, logs, or provider diagnostics.
        """
        if (
            not isinstance(continuation_token_secret_ref, str)
            or not continuation_token_secret_ref.strip()
            or continuation_token_secret_ref != continuation_token_secret_ref.strip()
        ):
            raise StorageConfigurationError(
                "continuation_token_secret_ref must be an exact non-empty string"
            )
        if isinstance(continuation_token_secret, str):
            continuation_token_secret = continuation_token_secret.encode()
        if not isinstance(continuation_token_secret, bytes) or len(continuation_token_secret) < 32:
            raise StorageConfigurationError("Continuation token secret must be at least 32 bytes")
        self._continuation_token_secret_ref = continuation_token_secret_ref
        self._continuation_token_secret = bytes(continuation_token_secret)
        self._embedder = embedder

    def validate_config(self, selection: StorageProviderSelection) -> None:
        """Validate the exact local options and captured secret-reference binding.

        Validation accepts only current typed options and performs no filesystem,
        database, secret-resolution, or embedding operation.

        Examples:
            Validate a default local selection:
                ```python
                provider.validate_config(selection)
                ```

            Reject a hidden fallback option:
                ```python
                with pytest.raises(StorageConfigurationError):
                    provider.validate_config(selection_with_fallback)
                ```

        Args:
            selection: Exact immutable provider name and local options.

        Returns:
            None: The selection can be opened by this exact provider instance.

        Notes:
            Unknown keys and a secret reference different from the factory-captured
            reference fail directly; values are never normalized into defaults.
        """
        config = _validate_local_config(selection)
        if config.continuation_token_secret_ref != self._continuation_token_secret_ref:
            raise StorageConfigurationError(
                "Local continuation secret reference does not match provider composition"
            )

    def open(self, request: StorageOpenRequest) -> LocalStorageBundle:
        """Open one coherent current-format local bundle synchronously.

        The manifest and all three database roles open before any store is published.
        Component schemas install only into a fresh/current writable workspace; partial
        construction closes every handle and leaves the manifest explicitly unclean.

        Examples:
            Open a writable local workspace:
                ```python
                bundle = provider.open(request)
                ```

            Open existing history read-only:
                ```python
                bundle = provider.open(read_only_request)
                ```

        Args:
            request: Complete trusted request selecting this exact local provider.

        Returns:
            LocalStorageBundle: Immutable typed stores sharing three owned databases.

        Notes:
            Open never blocks on asynchronous secret resolution, imports legacy paths,
            probes another provider, or falls back after an error.
        """
        self.validate_config(request.selection)
        config = _validate_local_config(request.selection)
        manifest = open_local_workspace_manifest(request)
        if request.mode is StorageOpenMode.READ_WRITE and manifest.clean_shutdown:
            update_local_workspace_lifecycle(
                request.workspace_root,
                clean_shutdown=False,
                last_maintenance_at=manifest.last_maintenance_at,
            )

        databases: list[LocalSQLiteDatabase] = []
        try:
            for role in (
                LocalDatabaseRole.CONTROL,
                LocalDatabaseRole.EVENTS,
                LocalDatabaseRole.SEARCH,
            ):
                databases.append(
                    LocalSQLiteDatabase.open(
                        workspace_root=request.workspace_root,
                        role=role,
                        mode=request.mode,
                        busy_timeout_ms=config.busy_timeout_ms,
                        durability=config.durability,
                    )
                )
            control, events_database, search_database = databases
            shared_kv = LocalKeyValueStore(database=control, clock=request.clock)
            shared_documents = LocalDocumentStore(database=control, clock=request.clock)
            runtime_output = LocalRuntimeOutputSink(
                database=events_database,
                max_pending_frames=config.runtime_output_max_pending_frames,
            )
            return LocalStorageBundle(
                capabilities=_local_capabilities(has_embedder=self._embedder is not None),
                mode=request.mode,
                events=LocalEventStore(database=events_database, stream="runtime"),
                memory_events=LocalEventStore(database=events_database, stream="memory"),
                state=LocalStateStore(database=control, clock=request.clock),
                blobs=LocalBlobStore(
                    database=control,
                    workspace_root=request.workspace_root,
                    clock=request.clock,
                ),
                artifacts=LocalArtifactRepository(database=control),
                search=LocalSearchBackend(
                    database=search_database,
                    embedder=self._embedder,
                    max_candidates=config.search_max_candidates,
                ),
                kv=shared_kv,
                documents=shared_documents,
                auth_grants=shared_kv,
                auth_invites=shared_kv,
                registry_manifests=shared_documents,
                runs=LocalRunRepository(database=control),
                run_results=LocalRunResultRepository(database=control),
                sessions=LocalSessionRepository(database=control),
                continuations=LocalContinuationRepository(
                    database=control,
                    token_secret=self._continuation_token_secret,
                ),
                continuation_leases=LocalContinuationLeaseRepository(database=control),
                triggers=LocalTriggerRepository(database=control),
                observations=LocalObservationRepository(database=control),
                ingress_idempotency=LocalIngressIdempotencyRepository(database=control),
                external_session_bindings=LocalExternalSessionBindingRepository(database=control),
                inbound_events=LocalInboundEventRepository(database=events_database),
                semantic_events=LocalSemanticEventRepository(database=events_database),
                runtime_output=runtime_output,
                _databases=tuple(databases),
                _workspace_root=request.workspace_root,
                _clock=request.clock,
            )
        except BaseException:
            for database in reversed(databases):
                database._close_during_open_failure()
            raise


def _validate_local_config(selection: StorageProviderSelection) -> _LocalProviderConfig:
    if selection.provider != LOCAL_PROVIDER_NAME:
        raise StorageConfigurationError(
            f"Local provider cannot validate selection {selection.provider!r}"
        )
    unknown = set(selection.config) - _CONFIG_KEYS
    if unknown:
        raise StorageConfigurationError(f"Unknown local storage options: {sorted(unknown)!r}")
    reference = selection.config.get("continuation_token_secret_ref")
    if not isinstance(reference, str) or not reference.strip() or reference != reference.strip():
        raise StorageConfigurationError(
            "continuation_token_secret_ref must be an exact non-empty string"
        )
    durability = selection.config.get("durability", "normal")
    if durability not in {"normal", "full"}:
        raise StorageConfigurationError("durability must be 'normal' or 'full'")
    return _LocalProviderConfig(
        busy_timeout_ms=_int_option(
            selection,
            "busy_timeout_ms",
            default=5_000,
            minimum=1,
            maximum=120_000,
        ),
        continuation_token_secret_ref=reference,
        durability=durability,
        runtime_output_max_pending_frames=_int_option(
            selection,
            "runtime_output_max_pending_frames",
            default=10_000,
            minimum=1,
            maximum=1_000_000,
        ),
        search_max_candidates=_int_option(
            selection,
            "search_max_candidates",
            default=10_000,
            minimum=1_000,
            maximum=100_000,
        ),
    )


def _int_option(
    selection: StorageProviderSelection,
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = selection.config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise StorageConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _local_capabilities(*, has_embedder: bool) -> StorageCapabilities:
    capabilities = {
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
        StorageCapability.ATOMIC_COMPARE_AND_SET,
        StorageCapability.ORDERED_APPEND,
        StorageCapability.MONOTONIC_CURSORS,
        StorageCapability.SHARED_DELIVERY_CURSOR,
        StorageCapability.TTL,
        StorageCapability.LEASES,
        StorageCapability.BLOB_STREAMING,
        StorageCapability.BLOB_RANGE_READ,
        StorageCapability.SEARCH_STRUCTURAL,
        StorageCapability.SEARCH_LEXICAL,
        StorageCapability.READ_ONLY_OPEN,
        StorageCapability.HEALTH,
    }
    if has_embedder:
        capabilities.update(
            {
                StorageCapability.SEARCH_SEMANTIC,
                StorageCapability.SEARCH_HYBRID,
            }
        )
    return StorageCapabilities(supported=frozenset(capabilities))
