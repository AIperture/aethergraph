from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

# ---- core services ----
from aethergraph.config.config import AppSettings
from aethergraph.contracts.integration import HostManifest
from aethergraph.contracts.services.llm import EmbeddingClientProtocol

# ---- trigger services ----
from aethergraph.contracts.services.trigger import TriggerService
from aethergraph.core.runtime.continuation_timer import ContinuationTimerService
from aethergraph.core.runtime.run_cancellation import RunCancellationRegistry
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.runtime_registry import current_registry, set_current_registry
from aethergraph.observability import (
    AgentEventTypeRegistry,
    CanonicalMeteringService,
    LoggingConfig,
    ObservationPolicy,
    RetentionPolicy,
    StdLoggerService,
    register_default_agent_event_types,
)
from aethergraph.observability.canonical_inspection import CanonicalInspectionReader
from aethergraph.observability.canonical_retention import ProviderRetentionJanitor
from aethergraph.observability.canonical_service import CanonicalObservationService
from aethergraph.server.admission import RunBurstLimiter
from aethergraph.server.security.credentials import EnvironmentSecretStore, resolve_auth_secret
from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
from aethergraph.services.auth.authz import AllowAllAuthz
from aethergraph.services.auth.canonical_authn import CanonicalAuthnService
from aethergraph.services.channel.channel_bus import ChannelBus

# ---- channel services ----
from aethergraph.services.channel.factory import build_bus, make_channel_adapters_from_env
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.llm.embed_factory import build_embedding_clients
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.factory import build_llm_clients
from aethergraph.services.llm.image_factory import build_image_generation_clients
from aethergraph.services.llm.image_service import ImageGenerationService
from aethergraph.services.llm.provider_transport import ProviderRateGate
from aethergraph.services.llm.service import LLMService

# ---- memory services ----
from aethergraph.services.memory.canonical_factory import CanonicalMemoryFacadeFactory

# ---- Other components ----
from aethergraph.services.registry.registration_service import RegistrationService
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.resume.multi_scheduler_resume_bus import MultiSchedulerResumeBus
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.services.scope.scope_factory import ScopeFactory
from aethergraph.services.triggers.engine import TriggerEngine
from aethergraph.services.triggers.trigger_service import TriggerServiceImpl
from aethergraph.services.viz.canonical_service import CanonicalVizService
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.storage.builtin_local import build_builtin_local_storage_registry
from aethergraph.storage.composition import StorageComposition
from aethergraph.storage.contracts import (
    StorageConfigurationError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
    StorageSecretResolver,
)
from aethergraph.storage.provider_registry import StorageProviderFactory, StorageProviderRegistry
from aethergraph.storage.providers.local_sqlite.manifest import LOCAL_PROVIDER_NAME
from aethergraph.storage.runtime_requirements import create_runtime_storage_composition

from .canonical_storage import CanonicalStorageServices, bind_canonical_storage_services

SERVICE_KEYS = [
    # core
    "registry",
    "logger",
    "clock",
    "channels",
    # continuations and resume
    "cont_store",
    "sched_registry",
    "wait_registry",
    "resume_bus",
    "resume_router",
    "continuation_timer",
    # storage and artifacts
    "kv",
    "artifact_factory",
    # memory
    "memory_factory",
    # optional
    "llm",
    "prompts",
    "authn",
    "authz",
    "metering",
    "observability",
]


class _UnavailableStorageSecrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise StorageConfigurationError(
            f"No storage secret resolver was supplied for {reference!r}"
        )


def _default_workspace_id(root: Path) -> str:
    normalized = str(root.resolve()).replace("\\", "/").casefold()
    return "workspace-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _runtime_storage_registry(
    *,
    selection: StorageProviderSelection,
    workspace_id: str,
    auth_secret: str,
    embedder: EmbeddingClientProtocol | None,
    providers: Mapping[str, StorageProviderFactory],
) -> StorageProviderRegistry:
    if selection.provider == LOCAL_PROVIDER_NAME:
        registry = build_builtin_local_storage_registry(
            selection=selection,
            workspace_id=workspace_id,
            auth_signing_secret=auth_secret,
            embedder=embedder,
        )
        for name, factory in providers.items():
            registry.register(name, factory)
        return registry
    return StorageProviderRegistry(providers)


@dataclass
class DefaultContainer:
    # root
    root: str

    # scope
    scope_factory: ScopeFactory

    # core
    registry: UnifiedRegistry
    logger: StdLoggerService
    clock: SystemClock
    storage_composition: StorageComposition
    storage_services: CanonicalStorageServices

    # channels and interactions
    channels: ChannelBus

    # continuations and resume
    cont_store: Any
    sched_registry: SchedulerRegistry
    wait_registry: WaitRegistry
    resume_bus: MultiSchedulerResumeBus
    resume_router: ResumeRouter
    continuation_timer: ContinuationTimerService
    state_store: Any
    trigger_engine: TriggerEngine
    trigger_service: TriggerService
    trigger_store: Any

    # provider-backed runtime services
    kv: Any
    artifact_factory: Any
    artifact_service: CanonicalPublicArtifactFacade
    registration_manifest_store: Any
    registration_service: RegistrationService
    memory_factory: CanonicalMemoryFacadeFactory

    # viz - only useful with frontend; otherwise this is a pure storage service for metrics and images
    viz_service: CanonicalVizService | None = None

    # optional llm service
    llm: LLMService | None = None
    observability: CanonicalInspectionReader | None = None
    observation_sink: CanonicalObservationService | None = None
    retention_janitor: ProviderRetentionJanitor | None = None
    embed_service: EmbeddingService | None = None
    image_service: ImageGenerationService | None = None

    # run controls -- for http endpoints and run manager
    run_store: Any | None = None
    run_result_store: Any | None = None
    run_manager: RunManager | None = None  # RunManager
    run_cancellation_registry: RunCancellationRegistry | None = None
    session_store: Any | None = None

    # optional services (not used by default)
    authn: CanonicalAuthnService | None = None
    authz: AllowAllAuthz | None = None

    metering: CanonicalMeteringService | None = None
    run_burst_limiter: RunBurstLimiter | None = None
    agent_event_registry: AgentEventTypeRegistry | None = None

    # extensible services
    ext_services: dict[str, Any] = field(default_factory=dict)

    # settings -- not a service, but useful to have around
    settings: AppSettings | None = None

    # installed only by an explicit immutable AG Host deployment
    host_manifest: HostManifest | None = None
    integration_ingress: Any | None = None
    semantic_events: Any | None = None
    semantic_turn_monitor: Any | None = None

    # opt-in host runtime output capture; disabled for normal CLI/server hosts
    runtime_output_sink: Any | None = None

    _storage_ready: bool = field(default=False, init=False, repr=False)

    async def start_storage(self) -> None:
        """Validate and publish the one prepared storage bundle.

        Intro:
            Completes capability and health admission before any background service
            or persistent runtime operation is allowed to start.

        Examples:
            Start an application container:
                ```python
                await container.start_storage()
                ```

            Reuse the idempotent readiness barrier:
                ```python
                await container.start_storage()
                await container.start_storage()
                ```

        Args:
            None.

        Returns:
            None: The exact prepared bundle is operationally ready.

        Notes:
            Failure is terminal for selection and never opens a fallback provider.
        """
        await self.storage_composition.start()
        self._storage_ready = True

    async def close_storage(self) -> None:
        """Close the one provider-owned storage composition.

        Intro:
            Delegates shutdown once background writers and schedulers have stopped,
            retaining retryability when provider close fails.

        Examples:
            Close application storage:
                ```python
                await container.close_storage()
                ```

            Close an unstarted prepared container:
                ```python
                await prepared.close_storage()
                ```

        Args:
            None.

        Returns:
            None: Provider resources are closed or were already closed.

        Notes:
            Individual stores are never closed by the container.
        """
        self.logger.close()
        try:
            await self.storage_composition.close()
        finally:
            self._storage_ready = False

    def require_storage_ready(self) -> None:
        """Fail a persistent operation before provider readiness.

        Intro:
            Provides the synchronous assertion used by public runtime boundaries
            after their asynchronous readiness barrier.

        Examples:
            Assert readiness before a query:
                ```python
                container.require_storage_ready()
                ```

            Reject a merely prepared container:
                ```python
                with pytest.raises(RuntimeError):
                    prepared.require_storage_ready()
                ```

        Args:
            None.

        Returns:
            None: Storage was admitted and published by the lifecycle owner.

        Notes:
            This method performs no health probe and cannot transition lifecycle state.
        """
        if not self._storage_ready:
            raise RuntimeError("Canonical storage is not ready.")


def build_default_container(
    *,
    root: str | None = None,
    cfg: AppSettings | None = None,
    channel_adapters: Mapping[str, Any] | None = None,
    storage_selection: StorageProviderSelection | None = None,
    storage_providers: Mapping[str, StorageProviderFactory] | None = None,
    storage_secrets: StorageSecretResolver | None = None,
    workspace_id: str | None = None,
    owner_scope: StorageScope | None = None,
) -> DefaultContainer:
    """Build one prepared runtime container over exactly one storage provider.

    Intro:
        Resolves and synchronously prepares one canonical bundle, binds every
        provider-backed runtime service to it, and leaves health publication to the
        explicit asynchronous container readiness barrier.

    Examples:
        Build a development container from application settings:
            ```python
            container = build_default_container(root="./workspace", cfg=settings)
            ```

        Build an explicitly injected external-provider container:
            ```python
            container = build_default_container(
                root="./deployment",
                cfg=settings,
                channel_adapters={},
                storage_selection=external_selection,
                storage_providers={"company.external": external_factory},
            )
            ```

    Args:
        root: Operational workspace root overriding `cfg.workspace`.
        cfg: Exact application settings or None to load development settings.
        channel_adapters: Exact adapter mapping. None selects development adapters
            from `cfg`; an empty mapping installs no transport adapters.
        storage_selection: Optional exact selection overriding `cfg.storage_provider`.
        storage_providers: Explicit trusted external provider factories.
        storage_secrets: Optional resolver required by selected external providers.
        workspace_id: Optional stable exact workspace identity.
        owner_scope: Optional trusted canonical storage owner.

    Returns:
        DefaultContainer: Prepared container awaiting `start_storage()`.

    Notes:
        The builder performs no readiness publication or fallback. A selected
        external provider is never replaced with the built-in local provider.
    """
    if cfg is None:
        from aethergraph.config.context import set_current_settings
        from aethergraph.config.loader import load_settings

        cfg = load_settings()
        set_current_settings(cfg)

    root = root or cfg.workspace
    cfg.workspace = root
    root_p = Path(root).resolve() if root else Path(cfg.workspace).resolve()
    resolved_workspace_id = workspace_id or _default_workspace_id(root_p)
    resolved_owner_scope = owner_scope or StorageScope(project_id=resolved_workspace_id)
    auth_secret = resolve_auth_secret(deploy_mode=cfg.deploy_mode, configured=cfg.auth.secret)
    selection = storage_selection or cfg.storage_provider.to_selection()
    credential_store = EnvironmentSecretStore()
    provider_rate_gate = ProviderRateGate()

    observation_policy = ObservationPolicy(
        capture_mode=cfg.llm.observability.capture_mode,
        full_prompt_ttl_days=cfg.observability.retention.max_full_prompt_age_days,
    )

    # Model clients are inert during construction. Persistence services are bound
    # immediately after provider preparation and before any client can be published.
    llm_clients = build_llm_clients(
        cfg.llm,
        credential_store,
        observation_sink=None,
        observation_capture_mode=cfg.llm.observability.capture_mode,
        rate_gate=provider_rate_gate,
    )
    embed_clients = build_embedding_clients(
        cfg.embed,
        credential_store,
        metering=None,
        rate_gate=provider_rate_gate,
        operation_quota_cfg=cfg.model_operation_usage_quota.embedding,
    )
    embed_client = embed_clients["default"] if embed_clients else None
    image_clients = build_image_generation_clients(
        cfg.image_generation,
        credential_store,
        metering=None,
        rate_gate=provider_rate_gate,
        operation_quota_cfg=cfg.model_operation_usage_quota.image_generation,
    )

    storage_registry = _runtime_storage_registry(
        selection=selection,
        workspace_id=resolved_workspace_id,
        auth_secret=auth_secret,
        embedder=embed_client,
        providers=dict(storage_providers or {}),
    )
    storage_composition = create_runtime_storage_composition(storage_registry)
    clock = SystemClock()
    bundle = storage_composition.prepare(
        StorageOpenRequest(
            workspace_id=resolved_workspace_id,
            workspace_root=root_p,
            owner_scope=resolved_owner_scope,
            selection=selection,
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=clock,
            secrets=storage_secrets or _UnavailableStorageSecrets(),
        )
    )
    services = bind_canonical_storage_services(
        bundle=bundle,
        owner_scope=resolved_owner_scope,
        clock=clock.now,
        observation_policy=observation_policy,
        llm=llm_clients.get("default"),
        memory_hot_max_events=cfg.memory.hot_limit,
        memory_hot_ttl_seconds=cfg.memory.hot_ttl_s,
        memory_signal_threshold=cfg.memory.signal_threshold,
    )
    metering_service = CanonicalMeteringService(services.metering)
    for client in llm_clients.values():
        client.observation_sink = services.observations
        client.metering = metering_service
    for client in embed_clients.values():
        client.metering = metering_service
    for client in image_clients.values():
        client.metering = metering_service

    llm_profiles = {"default": cfg.llm.default, **dict(cfg.llm.profiles or {})}
    llm_service = (
        LLMService(clients=llm_clients, secrets=credential_store, profiles=llm_profiles)
        if llm_clients
        else None
    )
    embed_service = EmbeddingService(clients=embed_clients) if embed_clients else None
    image_service = ImageGenerationService(image_clients) if image_clients else None
    if llm_service is not None and image_service is not None:
        llm_service.bind_image_service(image_service)

    logger_factory = StdLoggerService.build(
        LoggingConfig.from_cfg(cfg, log_dir=str(root_p / "logs")),
        observation_store=services.observations if cfg.observability.persist_logs else None,
    )
    registry: UnifiedRegistry = current_registry()
    set_current_registry(registry)
    scope_factory = ScopeFactory()
    sched_registry = SchedulerRegistry()
    wait_registry = WaitRegistry()
    resume_bus = MultiSchedulerResumeBus(
        registry=sched_registry,
        store=services.continuations,
        logger=logger_factory.for_service(ns="resume_bus"),
    )
    resume_router = ResumeRouter(
        store=services.continuations,
        runner=resume_bus,
        logger=logger_factory.for_service(ns="resume_router"),
        wait_registry=wait_registry,
    )
    continuation_timer = ContinuationTimerService(
        continuation_store=services.continuations,
        lease_store=services.continuation_leases,
        resume_router=resume_router,
        clock=clock,
        logger=logger_factory.for_service(ns="continuation_timer"),
    )
    selected_channel_adapters = (
        make_channel_adapters_from_env(cfg) if channel_adapters is None else dict(channel_adapters)
    )
    channels = build_bus(
        selected_channel_adapters,
        logger=logger_factory.for_channel(),
        resume_router=resume_router,
        cont_store=services.continuations,
    )

    run_cancellation_registry = RunCancellationRegistry()
    run_manager = RunManager(
        run_store=services.control.runs,
        result_store=services.control.run_results,
        state_store=services.graph_state,
        registry=registry,
        sched_registry=sched_registry,
        cancellation_registry=run_cancellation_registry,
        max_concurrent_runs=cfg.rate_limit.max_concurrent_runs,
    )
    rl_settings = cfg.rate_limit
    run_burst_limiter = RunBurstLimiter(
        max_events=rl_settings.burst_max_runs,
        window_seconds=rl_settings.burst_window_seconds,
    )
    authn = CanonicalAuthnService(
        store=services.auth,
        secret=auth_secret,
        cookie_name=cfg.auth.cookie_name,
        cookie_secure=cfg.auth.cookie_secure,
        cookie_samesite=cfg.auth.cookie_samesite,
        session_ttl_seconds=cfg.auth.session_ttl_seconds,
        grant_ttl_seconds=cfg.auth.grant_ttl_seconds,
        public_demo_fallback_enabled=cfg.auth.public_demo_fallback_enabled,
    )
    authz = AllowAllAuthz()
    agent_event_registry = register_default_agent_event_types(AgentEventTypeRegistry())
    artifact_service = services.artifact_factory.for_public_execution(StorageScope())
    registration_service = RegistrationService(
        registry=registry,
        manifest_store=services.registration_manifests,
        artifacts=artifact_service,
    )
    trigger_service = TriggerServiceImpl(
        store=services.triggers,
        observation_sink=services.observations,
        logger=logger_factory.for_service(ns="trigger_service"),
    )
    trigger_engine = TriggerEngine(
        store=services.triggers,
        run_manager=run_manager,
        run_store=services.control.runs,
        observation_sink=services.observations,
        logger=logger_factory.for_service(ns="trigger_engine"),
    )
    retention_cfg = cfg.observability.retention
    retention_janitor = ProviderRetentionJanitor(
        services.observations,
        RetentionPolicy(
            max_age_days=retention_cfg.max_age_days,
            error_max_age_days=retention_cfg.error_max_age_days,
            max_full_prompt_age_days=retention_cfg.max_full_prompt_age_days,
            max_bytes_per_trace=retention_cfg.max_bytes_per_trace,
            max_total_bytes=retention_cfg.max_total_bytes,
            max_retained_traces=retention_cfg.max_retained_traces,
            max_retained_runs=retention_cfg.max_retained_runs,
            max_observations_per_purge=retention_cfg.max_observations_per_purge,
        ),
        interval_seconds=retention_cfg.janitor_interval_seconds,
    )

    container = DefaultContainer(
        root=str(root_p),
        scope_factory=scope_factory,
        registry=registry,
        logger=logger_factory,
        clock=clock,
        storage_composition=storage_composition,
        storage_services=services,
        channels=channels,
        cont_store=services.continuations,
        sched_registry=sched_registry,
        wait_registry=wait_registry,
        resume_bus=resume_bus,
        resume_router=resume_router,
        continuation_timer=continuation_timer,
        trigger_store=services.triggers,
        trigger_engine=trigger_engine,
        trigger_service=trigger_service,
        kv=services.key_value,
        state_store=services.graph_state,
        artifact_factory=services.artifact_factory,
        artifact_service=artifact_service,
        registration_manifest_store=services.registration_manifests,
        registration_service=registration_service,
        viz_service=services.viz,
        memory_factory=services.memory_factory,
        llm=llm_service,
        observability=services.inspection(),
        observation_sink=services.observations,
        retention_janitor=retention_janitor,
        embed_service=embed_service,
        image_service=image_service,
        run_store=services.control.runs,
        run_result_store=services.control.run_results,
        run_manager=run_manager,
        run_cancellation_registry=run_cancellation_registry,
        session_store=services.control.sessions,
        authn=authn,
        authz=authz,
        metering=metering_service,
        run_burst_limiter=run_burst_limiter,
        agent_event_registry=agent_event_registry,
        settings=cfg,
    )

    return container


# Singleton (used unless the host sets their own)
DEFAULT_CONTAINER: DefaultContainer | None = None


def get_container() -> DefaultContainer:
    global DEFAULT_CONTAINER
    if DEFAULT_CONTAINER is None:
        DEFAULT_CONTAINER = build_default_container()
    return DEFAULT_CONTAINER


def set_container(c: DefaultContainer) -> None:
    global DEFAULT_CONTAINER
    DEFAULT_CONTAINER = c
