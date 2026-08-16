"""Canonical artifact facade composition over one coherent storage bundle."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from aethergraph.storage.contracts import StorageBundle, StorageScope

from .canonical_facade import CanonicalArtifactFacade
from .canonical_public import CanonicalPublicArtifactFacade


class CanonicalArtifactFacadeFactory:
    """Bind canonical artifact facades without exposing provider stores to consumers."""

    def __init__(
        self,
        *,
        bundle: StorageBundle,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
        artifact_id_factory: Callable[[], str] = lambda: f"artifact-{uuid4().hex}",
        occurrence_id_factory: Callable[[], str] = lambda: f"occurrence-{uuid4().hex}",
    ) -> None:
        """Compose one artifact-facade factory from an already-open bundle.

        The factory captures the provider-authoritative immutable-content owner and
        supplies the bundle's focused stores to every bound facade. It performs no I/O
        and does not select, open, or fall back to another provider.

        Examples:
            Compose from runtime storage:
                ```python
                factory = CanonicalArtifactFacadeFactory(
                    bundle=bundle,
                    owner_scope=open_request.owner_scope,
                    clock=open_request.clock.now,
                )
                ```

            Supply deterministic identities in a test:
                ```python
                factory = CanonicalArtifactFacadeFactory(
                    bundle=fake_bundle,
                    owner_scope=owner,
                    clock=clock,
                    artifact_id_factory=lambda: "artifact-1",
                    occurrence_id_factory=lambda: "occurrence-1",
                )
                ```

        Args:
            bundle: One coherent already-open canonical storage bundle.
            owner_scope: Exact immutable-content owner from provider composition.
            clock: UTC timestamp source shared by bound artifact facades.
            artifact_id_factory: Identity source for writes without an explicit ID.
            occurrence_id_factory: Occurrence identity source for writes without one.

        Returns:
            None: The inactive-until-S9 facade factory is ready without provider I/O.

        Notes:
            Deprecated App/client metadata is not accepted as owner input and cannot
            influence provider selection, authorization, or partitioning.
        """
        if not owner_scope.as_filter():
            raise ValueError("owner_scope must contain at least one canonical dimension")
        self._bundle = bundle
        self.owner_scope = owner_scope
        self._clock = clock
        self._artifact_id_factory = artifact_id_factory
        self._occurrence_id_factory = occurrence_id_factory

    def for_execution(
        self,
        execution_scope: StorageScope,
        *,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> CanonicalArtifactFacade:
        """Bind one facade to a partial canonical execution scope.

        Provider-authoritative owner dimensions are merged into the requested
        execution scope. Any conflicting populated dimension fails before a facade is
        returned, preventing consumer code from widening or replacing ownership.

        Examples:
            Bind one run:
                ```python
                facade = factory.for_execution(StorageScope(run_id="run-1"))
                ```

            Bind tool provenance:
                ```python
                facade = factory.for_execution(
                    StorageScope(run_id="run-1", node_id="node-1"),
                    tool_name="reporter",
                    tool_version="1.0",
                )
                ```

        Args:
            execution_scope: Partial canonical execution dimensions to merge with the
                exact provider owner.
            tool_name: Optional producing Tool name.
            tool_version: Optional producing Tool version.

        Returns:
            CanonicalArtifactFacade: Bound facade over the same coherent bundle.

        Notes:
            `app_id`, `client_id`, physical paths, and logical `scope_id` are absent
            from `StorageScope` and cannot be inferred by this factory.
        """
        return self._facade(
            _merge_execution_scope(self.owner_scope, execution_scope),
            tool_name=tool_name,
            tool_version=tool_version,
        )

    def for_owner(self) -> CanonicalArtifactFacade:
        """Bind one owner-wide facade for authorized API queries and maintenance.

        The execution scope equals the provider-authoritative owner scope, allowing
        callers to supply narrower canonical filters to individual query methods.

        Examples:
            Bind an owner-wide API facade:
                ```python
                facade = factory.for_owner()
                ```

            Query one run through the owner facade:
                ```python
                facade = factory.for_owner()
                page = await facade.query_public_artifacts(
                    scope=StorageScope(run_id="run-1"),
                )
                ```

        Args:
            None.

        Returns:
            CanonicalArtifactFacade: Owner-bound facade over the same bundle.

        Notes:
            The method performs no identity-label lookup and no provider fallback.
        """
        return self._facade(self.owner_scope)

    def for_public_execution(
        self,
        execution_scope: StorageScope,
        *,
        tool_name: str | None = None,
        tool_version: str | None = None,
        deprecated_app_id: str | None = None,
    ) -> CanonicalPublicArtifactFacade:
        """Bind stable public Artifact behavior to one canonical execution scope.

        The public projection and low-level canonical facade share the same coherent
        bundle stores and exact owner/execution scope.

        Examples:
            Bind NodeContext Artifacts:
                ```python
                artifacts = factory.for_public_execution(
                    StorageScope(run_id="run-1", node_id="node-1")
                )
                ```

            Bind Tool provenance and deprecated App response metadata:
                ```python
                artifacts = factory.for_public_execution(
                    StorageScope(run_id="run-1"),
                    tool_name="reporter",
                    tool_version="1.0",
                    deprecated_app_id="app-1",
                )
                ```

        Args:
            execution_scope: Partial canonical execution dimensions merged with owner.
            tool_name: Optional producing Tool name.
            tool_version: Optional producing Tool version.
            deprecated_app_id: Optional deprecated response-only App metadata.

        Returns:
            CanonicalPublicArtifactFacade: Stable public projection over one canonical facade.

        Notes:
            This method performs no provider lifecycle operation. Deprecated App
            metadata never affects provider scope, search, authorization, or identity.
        """
        return CanonicalPublicArtifactFacade(
            canonical=self.for_execution(
                execution_scope,
                tool_name=tool_name,
                tool_version=tool_version,
            ),
            deprecated_app_id=deprecated_app_id,
        )

    def _facade(
        self,
        execution_scope: StorageScope,
        *,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> CanonicalArtifactFacade:
        return CanonicalArtifactFacade(
            blobs=self._bundle.blobs,
            artifacts=self._bundle.artifacts,
            search=self._bundle.search,
            runs=self._bundle.runs,
            sessions=self._bundle.sessions,
            owner_scope=self.owner_scope,
            execution_scope=execution_scope,
            tool_name=tool_name,
            tool_version=tool_version,
            clock=self._clock,
            artifact_id_factory=self._artifact_id_factory,
            occurrence_id_factory=self._occurrence_id_factory,
        )


def _merge_execution_scope(owner: StorageScope, execution: StorageScope) -> StorageScope:
    dimensions = owner.as_filter()
    for name, value in execution.as_filter().items():
        current = dimensions.get(name)
        if current is not None and current != value:
            raise ValueError(f"execution_scope conflicts with owner_scope {name}")
        dimensions[name] = value
    return StorageScope(**dimensions)
