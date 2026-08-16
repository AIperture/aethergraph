"""Canonical memory-facade composition over one coherent storage bundle."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import StorageBundle, StorageScope

from .canonical_facade import CanonicalMemoryFacade


class CanonicalMemoryFacadeFactory:
    """Bind canonical memory facades without exposing provider stores to consumers."""

    def __init__(
        self,
        *,
        bundle: StorageBundle,
        owner_scope: StorageScope,
        hot_max_events: int = 500,
        hot_ttl_seconds: float = 900.0,
        monotonic_clock: Callable[[], float] = monotonic,
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
                    monotonic_clock=clock,
                )
                ```

        Args:
            bundle: One coherent already-open canonical storage bundle.
            owner_scope: Exact trusted provider ownership scope.
            hot_max_events: Positive maximum events retained by each facade cache.
            hot_ttl_seconds: Positive insertion-age lifetime for facade cache entries.
            monotonic_clock: Monotonic cache-expiry clock shared by bound facades.

        Returns:
            None: The inactive-until-S9 factory is ready without provider I/O.

        Notes:
            The owning `StorageComposition` retains lifecycle responsibility. App and
            client compatibility metadata are not accepted as owner dimensions.
        """
        validate_storage_owner_scope(owner_scope)
        if isinstance(hot_max_events, bool) or not isinstance(hot_max_events, int):
            raise TypeError("hot_max_events must be an integer")
        if hot_max_events < 1:
            raise ValueError("hot_max_events must be positive")
        if isinstance(hot_ttl_seconds, bool) or not isinstance(hot_ttl_seconds, int | float):
            raise TypeError("hot_ttl_seconds must be numeric")
        if hot_ttl_seconds <= 0:
            raise ValueError("hot_ttl_seconds must be positive")
        self._bundle = bundle
        self.owner_scope = owner_scope
        self._hot_max_events = hot_max_events
        self._hot_ttl_seconds = float(hot_ttl_seconds)
        self._monotonic_clock = monotonic_clock

    def for_execution(self, execution_scope: StorageScope) -> CanonicalMemoryFacade:
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
                    StorageScope(session_id="session-1", agent_id="writer")
                )
                ```

        Args:
            execution_scope: Partial canonical dimensions selected by runtime scope policy.

        Returns:
            CanonicalMemoryFacade: Bound facade over the same coherent provider bundle.

        Notes:
            Memory-level policy is resolved before this call. The factory does not
            derive scope from deprecated `app_id`, `client_id`, or a legacy bucket string.
        """
        scope = merge_storage_scope(self.owner_scope, **execution_scope.as_filter())
        return CanonicalMemoryFacade(
            event_store=self._bundle.memory_events,
            state_store=self._bundle.state,
            search_backend=self._bundle.search,
            scope=scope,
            hot_max_events=self._hot_max_events,
            hot_ttl_seconds=self._hot_ttl_seconds,
            monotonic_clock=self._monotonic_clock,
        )
