"""Canonical runtime-service composition over one storage bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from aethergraph.observability.canonical_inspection import (
    CanonicalInspectionReader,
    RunStatusResolver,
)
from aethergraph.observability.canonical_runtime_output import (
    CanonicalRuntimeOutputSink,
    bind_canonical_runtime_output,
)
from aethergraph.observability.canonical_service import (
    CanonicalObservationService,
    bind_canonical_observation_service,
)
from aethergraph.observability.inspection import ObservabilityIdentity
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.services.agent_state.canonical_facade import CanonicalAgentStateFacade
from aethergraph.services.artifacts.canonical_factory import CanonicalArtifactFacadeFactory
from aethergraph.services.auth.canonical_store import (
    CanonicalAuthStore,
    bind_canonical_auth_store,
)
from aethergraph.services.canonical_kv import (
    CanonicalKeyValueFacade,
    bind_canonical_key_value_facade,
)
from aethergraph.services.canonical_metering import (
    CanonicalMeteringStore,
    bind_canonical_metering_store,
)
from aethergraph.services.canonical_storage_scope import merge_storage_scope
from aethergraph.services.continuations.canonical_store import (
    CanonicalContinuationLeaseStore,
    CanonicalContinuationStore,
    bind_canonical_continuation_lease_store,
    bind_canonical_continuation_store,
)
from aethergraph.services.control.canonical_stores import (
    CanonicalControlStores,
    bind_canonical_control_stores,
)
from aethergraph.services.integration.canonical_factory import (
    CanonicalIntegrationPersistence,
    bind_canonical_integration_persistence,
)
from aethergraph.services.memory.canonical_factory import CanonicalMemoryFacadeFactory
from aethergraph.services.registry.canonical_manifest_store import (
    CanonicalRegistrationManifestStore,
    bind_canonical_registration_manifest_store,
)
from aethergraph.services.state_stores.canonical_store import CanonicalGraphStateStore
from aethergraph.services.triggers.canonical_store import (
    CanonicalTriggerStore,
    bind_canonical_trigger_store,
)
from aethergraph.services.viz.canonical_service import (
    CanonicalVizService,
    build_canonical_viz_service,
)
from aethergraph.storage.contracts import StateStore, StorageBundle, StorageScope

if TYPE_CHECKING:
    from aethergraph.contracts.services.llm import LLMClientProtocol


@dataclass(frozen=True, slots=True)
class CanonicalStorageServices:
    """Canonical service projections sharing one provider-owned bundle."""

    key_value: CanonicalKeyValueFacade
    metering: CanonicalMeteringStore
    control: CanonicalControlStores
    continuations: CanonicalContinuationStore
    continuation_leases: CanonicalContinuationLeaseStore
    triggers: CanonicalTriggerStore
    auth: CanonicalAuthStore
    registration_manifests: CanonicalRegistrationManifestStore
    integration: CanonicalIntegrationPersistence
    observations: CanonicalObservationService
    runtime_output: CanonicalRuntimeOutputSink
    viz: CanonicalVizService
    memory_factory: CanonicalMemoryFacadeFactory
    artifact_factory: CanonicalArtifactFacadeFactory
    graph_state: CanonicalGraphStateStore
    _state_store: StateStore = field(repr=False)
    _owner_scope: StorageScope = field(repr=False)

    def agent_state(self, scope: StorageScope) -> CanonicalAgentStateFacade:
        """Create one execution-scoped Agent-state facade.

        Intro:
            Merges a requested runtime scope with the trusted provider owner and binds
            the canonical state repository without opening or selecting storage.

        Examples:
            Bind run-scoped Agent state:
                ```python
                state = services.agent_state(StorageScope(run_id="run-1"))
                ```

            Bind node-scoped Agent state:
                ```python
                state = services.agent_state(
                    StorageScope(run_id="run-1", node_id="node-1")
                )
                ```

        Args:
            scope: Runtime dimensions to merge with the trusted ownership scope.

        Returns:
            CanonicalAgentStateFacade: Scoped facade over the bundle state repository.

        Notes:
            Conflicting tenant or project identity fails closed. The returned facade
            does not own provider lifecycle and has no fallback state store.
        """
        merged = merge_storage_scope(self._owner_scope, **scope.as_filter())
        return CanonicalAgentStateFacade(state_store=self._state_store, scope=merged)

    def inspection(
        self,
        *,
        identity: ObservabilityIdentity | None = None,
        run_status_resolver: RunStatusResolver | None = None,
    ) -> CanonicalInspectionReader:
        """Create one request-identity-aware canonical inspection reader.

        Intro:
            Binds bounded observation reads at the consumer boundary so local, demo,
            and cloud identities are never captured as one global container identity.

        Examples:
            Bind local inspection:
                ```python
                reader = services.inspection()
                ```

            Bind an authenticated cloud request:
                ```python
                reader = services.inspection(
                    identity=ObservabilityIdentity(
                        mode="cloud", user_id="user-1", org_id="org-1"
                    )
                )
                ```

        Args:
            identity: Optional exact read identity; omission selects local semantics.
            run_status_resolver: Optional bounded run-status batch resolver.

        Returns:
            CanonicalInspectionReader: Request-scoped reader over canonical observations.

        Notes:
            Construction performs no I/O and does not expose the provider bundle.
        """
        if identity is None:
            return CanonicalInspectionReader(
                self.observations,
                run_status_resolver=run_status_resolver,
            )
        return CanonicalInspectionReader(
            self.observations,
            identity=identity,
            run_status_resolver=run_status_resolver,
        )


def bind_canonical_storage_services(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
    observation_policy: ObservationPolicy,
    llm: LLMClientProtocol | None = None,
    runtime_output_tags: tuple[str, ...] = (),
    memory_hot_max_events: int = 500,
    memory_hot_ttl_seconds: float = 900.0,
    memory_signal_threshold: float = 0.0,
) -> CanonicalStorageServices:
    """Bind the complete runtime storage-service graph to one coherent bundle.

    Intro:
        Centralizes every provider-backed runtime projection. Request and execution
        identities remain factories on the returned aggregate rather than global
        mutable state.

    Examples:
        Bind a ready local bundle:
            ```python
            services = bind_canonical_storage_services(
                bundle=bundle,
                owner_scope=open_request.owner_scope,
                clock=open_request.clock.now,
                observation_policy=policy,
            )
            ```

        Bind an external provider with an LLM distiller:
            ```python
            services = bind_canonical_storage_services(
                bundle=external_bundle,
                owner_scope=owner_scope,
                clock=clock.now,
                observation_policy=ObservationPolicy(capture_mode="metadata"),
                llm=llm,
                runtime_output_tags=("external",),
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope without execution dimensions.
        clock: Timezone-aware UTC source shared by canonical service projections.
        observation_policy: Exact capture, redaction, and retention policy.
        llm: Optional LLM client used only for explicit Memory distillation.
        runtime_output_tags: Immutable provider-neutral output classification tags.
        memory_hot_max_events: Positive per-facade in-memory event bound.
        memory_hot_ttl_seconds: Positive in-memory event insertion lifetime.
        memory_signal_threshold: Finite default explicit distillation threshold.

    Returns:
        CanonicalStorageServices: Provider-backed service graph over exact bundle fields.

    Notes:
        The binder performs no selection, open, health check, I/O, fallback, global
        activation, or close. The owning `StorageComposition` retains lifecycle.
    """
    observations = bind_canonical_observation_service(
        bundle=bundle,
        owner_scope=owner_scope,
        policy=observation_policy,
    )
    return CanonicalStorageServices(
        key_value=bind_canonical_key_value_facade(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        metering=bind_canonical_metering_store(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        control=bind_canonical_control_stores(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        continuations=bind_canonical_continuation_store(
            bundle=bundle,
            owner_scope=owner_scope,
        ),
        continuation_leases=bind_canonical_continuation_lease_store(
            bundle=bundle,
            owner_scope=owner_scope,
        ),
        triggers=bind_canonical_trigger_store(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        auth=bind_canonical_auth_store(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        registration_manifests=bind_canonical_registration_manifest_store(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        integration=bind_canonical_integration_persistence(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        observations=observations,
        runtime_output=bind_canonical_runtime_output(
            bundle=bundle,
            owner_scope=owner_scope,
            tags=runtime_output_tags,
        ),
        viz=build_canonical_viz_service(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        memory_factory=CanonicalMemoryFacadeFactory(
            bundle=bundle,
            owner_scope=owner_scope,
            hot_max_events=memory_hot_max_events,
            hot_ttl_seconds=memory_hot_ttl_seconds,
            default_signal_threshold=memory_signal_threshold,
            llm=llm,
            clock=clock,
        ),
        artifact_factory=CanonicalArtifactFacadeFactory(
            bundle=bundle,
            owner_scope=owner_scope,
            clock=clock,
        ),
        graph_state=CanonicalGraphStateStore(
            state_store=bundle.state,
            event_store=bundle.events,
            run_repository=bundle.runs,
        ),
        _state_store=bundle.state,
        _owner_scope=owner_scope,
    )
