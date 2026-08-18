"""Canonical memory-facade composition over one coherent storage bundle."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import math
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import StorageBundle, StorageScope

from .canonical_facade import CanonicalMemoryFacade
from .canonical_public import CanonicalPublicMemoryFacade

if TYPE_CHECKING:
    from aethergraph.contracts.services.llm import LLMClientProtocol

_MAX_HOT_EVENTS = 10_000


class CanonicalMemoryFacadeFactory:
    """Bind canonical memory facades without exposing provider stores to consumers."""

    def __init__(
        self,
        *,
        bundle: StorageBundle,
        owner_scope: StorageScope,
        hot_max_events: int = 500,
        hot_ttl_seconds: float = 900.0,
        default_signal_threshold: float = 0.0,
        llm: LLMClientProtocol | None = None,
        monotonic_clock: Callable[[], float] = monotonic,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        event_id_factory: Callable[[], str] = lambda: f"event-{uuid4().hex}",
    ) -> None:
        """Compose one memory factory from an already-open storage bundle.

        The factory retains a single coherent bundle and its trusted ownership scope.
        Construction performs no provider selection, open, health check, or I/O.

        Examples:
            Bind production composition inputs:
                ```python
                factory = CanonicalMemoryFacadeFactory(
                    bundle=bundle,
                    owner_scope=open_request.owner_scope,
                )
                ```

            Configure deterministic cache behavior:
                ```python
                factory = CanonicalMemoryFacadeFactory(
                    bundle=fake_bundle,
                    owner_scope=StorageScope(project_id="project-1"),
                    hot_max_events=100,
                    hot_ttl_seconds=60.0,
                    default_signal_threshold=0.25,
                    llm=llm,
                    monotonic_clock=clock,
                    clock=utc_clock.now,
                    event_id_factory=lambda: "event-1",
                )
                ```

        Args:
            bundle: One coherent already-open canonical storage bundle.
            owner_scope: Exact trusted provider ownership scope.
            hot_max_events: Positive maximum events retained by each facade cache.
            hot_ttl_seconds: Positive insertion-age lifetime for facade cache entries.
            default_signal_threshold: Finite default public distillation threshold.
            llm: Optional explicitly injected client for requested LLM distillation.
            monotonic_clock: Monotonic cache-expiry clock shared by bound facades.
            clock: Timezone-aware UTC event timestamp source for public facades.
            event_id_factory: Stable non-empty public event identity source.

        Returns:
            None: The provider-backed factory is ready without provider I/O.

        Notes:
            The owning `StorageComposition` retains lifecycle responsibility. App and
            client compatibility metadata are not accepted as owner dimensions.
        """
        validate_storage_owner_scope(owner_scope)
        if isinstance(hot_max_events, bool) or not isinstance(hot_max_events, int):
            raise TypeError("hot_max_events must be an integer")
        if hot_max_events < 1:
            raise ValueError("hot_max_events must be positive")
        if hot_max_events > _MAX_HOT_EVENTS:
            raise ValueError(f"hot_max_events must not exceed {_MAX_HOT_EVENTS}")
        if isinstance(hot_ttl_seconds, bool) or not isinstance(hot_ttl_seconds, int | float):
            raise TypeError("hot_ttl_seconds must be numeric")
        if hot_ttl_seconds <= 0:
            raise ValueError("hot_ttl_seconds must be positive")
        if (
            isinstance(default_signal_threshold, bool)
            or not isinstance(default_signal_threshold, int | float)
            or not math.isfinite(default_signal_threshold)
        ):
            raise ValueError("default_signal_threshold must be a finite number")
        self._bundle = bundle
        self.owner_scope = owner_scope
        self._hot_max_events = hot_max_events
        self._hot_ttl_seconds = float(hot_ttl_seconds)
        self._default_signal_threshold = float(default_signal_threshold)
        self._llm = llm
        self._monotonic_clock = monotonic_clock
        self._clock = clock
        self._event_id_factory = event_id_factory

    def for_execution(
        self,
        execution_scope: StorageScope,
        *,
        event_scope: StorageScope | None = None,
    ) -> CanonicalMemoryFacade:
        """Bind one memory facade to an exact partial execution scope.

        Provider-authoritative owner dimensions are merged with caller execution
        dimensions. Any conflicting populated dimension fails before construction.

        Examples:
            Bind run memory:
                ```python
                memory = factory.for_execution(StorageScope(run_id="run-1"))
                ```

            Bind session Agent memory:
                ```python
                memory = factory.for_execution(
                    StorageScope(session_id="session-1"),
                    event_scope=StorageScope(
                        session_id="session-1",
                        run_id="run-1",
                        agent_id="writer",
                    ),
                )
                ```

        Args:
            execution_scope: Partial canonical dimensions selected by runtime scope policy.
            event_scope: Optional full event provenance containing the bucket dimensions.

        Returns:
            CanonicalMemoryFacade: Bound facade over the same coherent provider bundle.

        Notes:
            Memory-level policy is resolved before this call. The factory does not
            derive scope from deprecated `app_id`, `client_id`, or a legacy bucket string.
        """
        scope = merge_storage_scope(self.owner_scope, **execution_scope.as_filter())
        resolved_event_scope = (
            scope
            if event_scope is None
            else merge_storage_scope(self.owner_scope, **event_scope.as_filter())
        )
        return CanonicalMemoryFacade(
            event_store=self._bundle.memory_events,
            state_store=self._bundle.state,
            search_backend=self._bundle.search,
            scope=scope,
            event_scope=resolved_event_scope,
            hot_max_events=self._hot_max_events,
            hot_ttl_seconds=self._hot_ttl_seconds,
            monotonic_clock=self._monotonic_clock,
        )

    def for_public_execution(
        self,
        execution_scope: StorageScope,
        *,
        logical_scope_id: str,
        provenance_scope: StorageScope | None = None,
        deprecated_app_id: str | None = None,
    ) -> CanonicalPublicMemoryFacade:
        """Bind stable public Memory behavior to one canonical execution scope.

        The low-level facade and public DTO projection share the exact same bundle
        stores. Logical bucket and deprecated App labels remain response metadata.

        Examples:
            Bind session Memory for `NodeContext`:
                ```python
                memory = factory.for_public_execution(
                    StorageScope(session_id="session-1"),
                    logical_scope_id="session:session-1",
                    provenance_scope=StorageScope(
                        session_id="session-1",
                        run_id="run-1",
                    ),
                )
                ```

            Bind optional deprecated App compatibility metadata:
                ```python
                memory = factory.for_public_execution(
                    StorageScope(run_id="run-1"),
                    logical_scope_id="run:run-1",
                    deprecated_app_id="app-1",
                )
                ```

        Args:
            execution_scope: Partial canonical dimensions selected by runtime scope policy.
            logical_scope_id: Stable public memory-bucket label, never provider scope.
            provenance_scope: Optional full execution provenance for public Event DTOs.
            deprecated_app_id: Optional explicitly deprecated response metadata.

        Returns:
            CanonicalPublicMemoryFacade: Stable public projection over one canonical facade.

        Notes:
            This method performs no provider lifecycle operation and deprecated App
            metadata never affects the scope passed to provider stores.
        """
        provenance_input = provenance_scope or execution_scope
        resolved_provenance = merge_storage_scope(
            self.owner_scope,
            **provenance_input.as_filter(),
        )
        return CanonicalPublicMemoryFacade(
            canonical=self.for_execution(
                execution_scope,
                event_scope=provenance_input,
            ),
            logical_scope_id=logical_scope_id,
            provenance_scope=resolved_provenance,
            deprecated_app_id=deprecated_app_id,
            llm=self._llm,
            default_signal_threshold=self._default_signal_threshold,
            clock=self._clock,
            event_id_factory=self._event_id_factory,
        )
