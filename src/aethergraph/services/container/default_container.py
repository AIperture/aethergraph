from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

# ---- core services ----
from aethergraph.config.config import AppSettings
from aethergraph.contracts.services.execution import ExecutionService

# ---- optional services (not used by default) ----
# ---- scheduler ---- TODO: move to a separate server to handle scheduling across threads/processes
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.services.sessions import SessionStore
from aethergraph.contracts.services.state_stores import GraphStateStore

# ---- trigger services ----
from aethergraph.contracts.services.trigger import TriggerService
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler

# ---- artifact services ----
from aethergraph.core.runtime.run_cancellation import RunCancellationRegistry
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.runtime_registry import current_registry, set_current_registry
from aethergraph.services.auth.authn import AuthnService
from aethergraph.services.auth.authz import AllowAllAuthz
from aethergraph.services.channel.channel_bus import ChannelBus

# from aethergraph.services.eventhub.event_hub import EventHub
from aethergraph.services.channel.event_hub import EventHub

# ---- channel services ----
from aethergraph.services.channel.factory import build_bus, make_channel_adapters_from_env
from aethergraph.services.channel.ingress import ChannelIngress
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import (
    FSContinuationStore,  # AsyncContinuationStore
)
from aethergraph.services.eventbus.inmem import InMemoryEventBus
from aethergraph.services.execution.local_python import LocalPythonExecutionService

# ---- Global Indices ----
from aethergraph.services.indices.global_indices import GlobalIndices
from aethergraph.services.inspect import (
    AgentEventTypeRegistry,
    JsonlLLMObservationStore,
    register_default_agent_event_types,
)
from aethergraph.services.knowledge.chunker import TextSplitter

# ---- kv services ----
from aethergraph.services.knowledge.local_fs_backend import LocalFSKnowledgeBackend
from aethergraph.services.llm.embed_factory import build_embedding_clients
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.factory import build_llm_clients
from aethergraph.services.llm.observability import (
    ConsoleLLMObservationSink,
    JsonlLLMObservationSink,
)
from aethergraph.services.llm.service import LLMService
from aethergraph.services.logger.std import LoggingConfig, StdLoggerService
from aethergraph.services.mcp.service import MCPService

# ---- memory services ----
from aethergraph.services.memory.factory import MemoryFactory
from aethergraph.services.metering.eventlog_metering import EventLogMeteringService

# ---- Planning components ----
from aethergraph.services.planning.action_catalog import ActionCatalog
from aethergraph.services.planning.flow_validator import FlowValidator
from aethergraph.services.planning.planner_service import PlannerService

# ---- Other components ----
from aethergraph.services.rate_limit.inmem_rate_limit import SimpleRateLimiter
from aethergraph.services.redactor.simple import RegexRedactor  # Simple PII redactor
from aethergraph.services.registry.registration_service import RegistrationService
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.resume.multi_scheduler_resume_bus import MultiSchedulerResumeBus
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.services.scope.scope_factory import ScopeFactory
from aethergraph.services.secrets.env import EnvSecrets
from aethergraph.services.skills.skill_registry import SkillRegistry
from aethergraph.services.tracing import EventLogTracer, NoopTracer
from aethergraph.services.triggers.engine import TriggerEngine
from aethergraph.services.triggers.trigger_service import TriggerServiceImpl
from aethergraph.services.viz.viz_service import VizService
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.wakeup.memory_queue import ThreadSafeWakeupQueue
from aethergraph.services.websearch.httpx_fetcher import HttpxWebPageFetcher
from aethergraph.services.websearch.providers.default import DefaultWebSearchProvider
from aethergraph.services.websearch.service import WebSearchService
from aethergraph.services.websearch.types import WebSearchEngine

# ---- storage builders ----
from aethergraph.storage.factory import (
    build_artifact_index,
    build_artifact_store,
    build_continuation_store,
    build_doc_store,
    build_event_log,
    build_graph_state_store,
    build_memory_hotlog,
    build_memory_persistence,
    build_run_store,
    build_session_store,
)
from aethergraph.storage.kv.inmem_kv import InMemoryKV as EphemeralKV
from aethergraph.storage.kv.sqlite_kv_sync import SQLiteKVSync
from aethergraph.storage.metering.meter_event import EventLogMeteringStore
from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore
from aethergraph.storage.search_factory import build_kb_search_backend, build_search_backend
from aethergraph.storage.triggers.trigger_docstore import DocTriggerStore

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
    "wakeup_queue",
    # storage and artifacts
    "kv_hot",
    "artifacts",
    "artifact_index",
    # memory
    "memory_factory",
    # optional
    "llm",
    "event_bus",
    "prompts",
    "authn",
    "authz",
    "redactor",
    "metering",
    "tracer",
    "secrets",
]


@dataclass
class DefaultContainer:
    # root
    root: str

    # scope
    scope_factory: ScopeFactory

    # schedulers
    schedulers: dict[str, Any]

    # core
    registry: UnifiedRegistry
    logger: StdLoggerService
    clock: SystemClock

    # channels and interactions
    channels: ChannelBus
    eventhub: EventHub

    # continuations and resume
    cont_store: FSContinuationStore
    sched_registry: SchedulerRegistry
    wait_registry: WaitRegistry
    resume_bus: MultiSchedulerResumeBus
    resume_router: ResumeRouter
    wakeup_queue: ThreadSafeWakeupQueue
    state_store: GraphStateStore
    trigger_engine: TriggerEngine
    trigger_service: TriggerService
    trigger_store: TriggerStore

    # storage and artifacts
    doc_store: DocStore
    kv_hot: EphemeralKV
    artifacts: AsyncArtifactStore
    artifact_index: AsyncArtifactIndex
    registration_manifest_store: RegistrationManifestStore
    registration_service: RegistrationService
    eventlog: EventLog
    global_indices: GlobalIndices
    kb_backend: LocalFSKnowledgeBackend  # for now just use the local FS backend as the default; in the future, we can make this swappable like the global indices backend, and add auto-indexing to the NodeKB facade

    # memory
    memory_factory: MemoryFactory

    # viz - only useful with frontend; otherwise this is a pure storage service for metrics and images
    viz_service: VizService | None = None

    # optional llm service
    llm: LLMService | None = None
    llm_observation_sink: Any | None = None
    llm_observation_path: str | None = None
    llm_observation_store: Any | None = None
    mcp: MCPService | None = None
    embed_service: EmbeddingService | None = None
    web_search: WebSearchService | None = None

    # run controls -- for http endpoints and run manager
    run_store: RunStore | None = None
    run_manager: RunManager | None = None  # RunManager
    run_cancellation_registry: RunCancellationRegistry | None = None
    session_store: SessionStore | None = None  # SessionStore

    # planner
    planner_service: PlannerService | None = None

    # skills
    skills_registry: SkillRegistry | None = None

    # optional services (not used by default)
    execution: ExecutionService | None = None
    event_bus: InMemoryEventBus | None = None
    authn: AuthnService | None = None
    authz: AllowAllAuthz | None = None
    redactor: RegexRedactor | None = None

    metering: MeteringService | None = None
    rate_limiter: SimpleRateLimiter | None = None
    tracer: NoopTracer | EventLogTracer | None = None
    agent_event_registry: AgentEventTypeRegistry | None = None
    secrets: EnvSecrets | None = None

    # extensible services
    ext_services: dict[str, Any] = field(default_factory=dict)

    # settings -- not a service, but useful to have around
    settings: AppSettings | None = None

    # channel ingress (set after init to avoid circular dependency)
    channel_ingress: ChannelIngress | None = None  # set after init to avoid circular dependency


def build_default_container(
    *,
    root: str | None = None,
    cfg: AppSettings | None = None,
) -> DefaultContainer:
    """Build the default service container with standard services.
    if "root" is provided, use it as the base directory for storage; else use from cfg.workspace.
    if cfg is not provided, load from default AppSettings.
    """
    if cfg is None:
        from aethergraph.config.context import set_current_settings
        from aethergraph.config.loader import load_settings

        cfg = load_settings()
        set_current_settings(cfg)

    root = root or cfg.workspace
    # override workspace in cfg to match
    cfg.workspace = root

    # we use user specified root if provided, else from config/env
    root_p = Path(root).resolve() if root else Path(cfg.workspace).resolve()
    (root_p / "kv").mkdir(parents=True, exist_ok=True)
    (root_p / "index").mkdir(parents=True, exist_ok=True)
    (root_p / "memory").mkdir(parents=True, exist_ok=True)

    # Scope factory
    scope_factory = ScopeFactory()

    # event log for metering and channel events --
    # TODO: make configurable from cfg
    eventlog = build_event_log(cfg)

    # core services
    logger_factory = StdLoggerService.build(
        LoggingConfig.from_cfg(cfg, log_dir=str(root_p / "logs")),
        event_log=eventlog,
    )

    clock = SystemClock()
    # registry = UnifiedRegistry()
    registry: UnifiedRegistry = current_registry()
    set_current_registry(registry)  # set global registry, ensure singleton (optional)

    # continuations and resume
    cont_store = build_continuation_store(cfg)

    sched_registry = SchedulerRegistry()
    wait_registry = WaitRegistry()
    resume_bus = MultiSchedulerResumeBus(
        registry=sched_registry,
        store=cont_store,
        logger=logger_factory.for_service(ns="resume_bus"),
    )
    resume_router = ResumeRouter(
        store=cont_store,
        runner=resume_bus,
        logger=logger_factory.for_service(ns="resume_router"),
        wait_registry=wait_registry,
    )
    wakeup_queue = ThreadSafeWakeupQueue()  # TODO: this is a placeholder, not fully implemented
    # state_store = JsonGraphStateStore(root=str(root_p / "graph_states"))
    state_store = build_graph_state_store(cfg)

    # global scheduler
    global_sched = GlobalForwardScheduler(
        registry=sched_registry,
        global_max_concurrency=None,  # TODO: make configurable
        logger=logger_factory.for_scheduler(),
    )
    schedulers = {
        "global": global_sched,
        "registry": sched_registry,
    }

    # channels
    event_hub = (
        EventHub()
    )  # in-memory event hub for WebUI and other real-time events; not configurable yet
    channel_adapters = make_channel_adapters_from_env(cfg, event_log=eventlog, event_hub=event_hub)
    channels = build_bus(
        channel_adapters,
        default="console:stdin",
        logger=logger_factory.for_channel(),
        resume_router=resume_router,
        cont_store=cont_store,
    )

    # storage and artifacts -- kv_hot has special methods for hot data, do not use other persistent kv here
    kv_hot = EphemeralKV()

    artifacts = build_artifact_store(cfg)
    artifact_index = build_artifact_index(cfg)

    viz_service = VizService(event_log=eventlog)

    # Metering service
    # TODO: make metering service configurable
    metering_store = EventLogMeteringStore(event_log=eventlog)
    metering = EventLogMeteringService(store=metering_store)

    # optional services
    secrets = (
        EnvSecrets()
    )  # get secrets from env vars -- for local development; in prod, use a proper secrets manager
    obs_cfg = cfg.llm.observability
    llm_observation_sink = None
    llm_observation_path: str | None = None
    llm_observation_store = None
    if obs_cfg.enabled:
        if obs_cfg.sink == "console":
            llm_observation_sink = ConsoleLLMObservationSink(prompt_view=obs_cfg.prompt_view)
        elif obs_cfg.sink == "file":
            obs_path = Path(obs_cfg.path)
            if not obs_path.is_absolute():
                obs_path = root_p / obs_path
            llm_observation_sink = JsonlLLMObservationSink(obs_path)
            llm_observation_path = str(obs_path)
            llm_observation_store = JsonlLLMObservationStore(obs_path)
        else:
            raise ValueError(f"Unsupported LLM observability sink: {obs_cfg.sink!r}")
    llm_clients = build_llm_clients(
        cfg.llm,
        secrets,
        observation_sink=llm_observation_sink,
        observation_capture_mode=obs_cfg.capture_mode,
    )  # return {profile: GenericLLMClient}
    llm_service = LLMService(clients=llm_clients) if llm_clients else None

    embed_clients = build_embedding_clients(
        cfg.embed, secrets, metering=metering
    )  # return {profile: GenericEmbeddingClient}
    embed_service = EmbeddingService(clients=embed_clients) if embed_clients else None
    embed_client = embed_clients["default"] if embed_clients else None

    mcp = MCPService()  # empty MCP service; users can register clients as needed

    # web search service uses a provider-agnostic default no-op provider.
    # This keeps `fetch` always available even when search providers are not configured.
    default_web_provider = DefaultWebSearchProvider()
    web_search: WebSearchService | None = WebSearchService(
        providers={WebSearchEngine.custom: default_web_provider},
        default_engine=WebSearchEngine.custom,
        page_fetcher=HttpxWebPageFetcher(),
    )

    # memory factory
    persistence = build_memory_persistence(cfg)
    hotlog = build_memory_hotlog(cfg)
    memory_factory = MemoryFactory(
        hotlog=hotlog,
        persistence=persistence,
        artifacts=artifacts,
        hot_limit=int(cfg.memory.hot_limit),
        hot_ttl_s=int(cfg.memory.hot_ttl_s),
        default_signal_threshold=float(cfg.memory.signal_threshold),
        logger=logger_factory.for_service(ns="memory"),
        llm_service=llm_service.get("default") if llm_service else None,
    )

    # run store and manager
    run_store = build_run_store(cfg)
    run_cancellation_registry = RunCancellationRegistry()
    run_manager = RunManager(
        run_store=run_store,
        registry=registry,
        sched_registry=sched_registry,
        cancellation_registry=run_cancellation_registry,
        max_concurrent_runs=cfg.rate_limit.max_concurrent_runs,
    )
    session_store = build_session_store(cfg)

    # rate limiter
    rl_settings = cfg.rate_limit
    rate_limiter = SimpleRateLimiter(
        max_events=rl_settings.burst_max_runs,
        window_seconds=rl_settings.burst_window_seconds,
    )

    # auth services
    auth_secret = (
        cfg.auth.secret.get_secret_value()
        if cfg.auth.secret is not None
        else "aethergraph-dev-secret"
    )
    auth_db_path = str(root_p / "auth" / "auth_kv.db")
    auth_grant_store = SQLiteKVSync(auth_db_path, prefix="grant:")
    auth_invite_store = SQLiteKVSync(auth_db_path, prefix="invite:")
    authn = AuthnService(
        secret=auth_secret,
        cookie_name=cfg.auth.cookie_name,
        cookie_secure=cfg.auth.cookie_secure,
        cookie_samesite=cfg.auth.cookie_samesite,
        session_ttl_seconds=cfg.auth.session_ttl_seconds,
        grant_ttl_seconds=cfg.auth.grant_ttl_seconds,
        public_demo_fallback_enabled=cfg.auth.public_demo_fallback_enabled,
        grant_store=auth_grant_store,
        invite_store=auth_invite_store,
    )
    authn.load_persisted()
    authz = AllowAllAuthz()

    # global scoped indices
    # from aethergraph.storage.search_backend.generic_vector_backend import SQLiteVectorSearchBackend

    # search_backend = SQLiteVectorSearchBackend(
    #     index=vec_index,
    #     embedder=embed_client,
    # )

    global_indices_backend = build_search_backend(cfg=cfg, embedder=embed_client)
    global_indices = GlobalIndices(backend=global_indices_backend)  # to be set up later as needed

    kb_search_backend = build_kb_search_backend(cfg, embedder=embed_client)
    kb_backend = LocalFSKnowledgeBackend(
        corpus_root=os.path.join(os.path.abspath(cfg.workspace), cfg.knowledge.corpus_root),
        artifacts=artifacts,  # this is store, not Facade with auto-indexing, long doc has its own indexing method
        search_backend=kb_search_backend,
        embed_client=embed_client,
        llm_client=llm_clients.get("default") if llm_clients else None,
        chunker=TextSplitter(),
        logger=logger_factory.for_service(ns="kb_backend"),
    )

    # Execution service
    execution = (
        LocalPythonExecutionService()
    )  # simple local python executor -- NOT SANDBOXED; just for local functionality testing

    # Planner service
    catalog = ActionCatalog(registry=registry)
    flow_validator = FlowValidator(catalog=catalog)
    planner_service = PlannerService(
        catalog=catalog,
        llm=llm_service.get("default") if llm_service else None,
        validator=flow_validator,
        run_manager=run_manager,
    )

    # skills registry
    skills_registry = SkillRegistry()
    agent_event_registry = register_default_agent_event_types(AgentEventTypeRegistry())

    # trigger services
    doc_store = build_doc_store(cfg)
    registration_manifest_store = RegistrationManifestStore(doc_store=doc_store)
    registration_service = RegistrationService(
        registry=registry,
        manifest_store=registration_manifest_store,
        artifact_store=artifacts,
        artifact_index=artifact_index,
    )
    trigger_store = DocTriggerStore(
        doc_store=doc_store
    )  # for simplicity, we use the event log as the backing store for triggers; in the future, we can make this swappable like other storage services
    trigger_service = TriggerServiceImpl(
        store=trigger_store,
        event_log=eventlog,
        logger=logger_factory.for_service(ns="trigger_service"),
    )
    trigger_engine = TriggerEngine(
        store=trigger_store,
        run_manager=run_manager,
        event_log=eventlog,
        run_store=run_store,
        logger=logger_factory.for_service(ns="trigger_engine"),
    )

    container = DefaultContainer(
        root=str(root_p),
        scope_factory=scope_factory,
        schedulers=schedulers,
        registry=registry,
        logger=logger_factory,
        clock=clock,
        channels=channels,
        eventhub=event_hub,
        skills_registry=skills_registry,
        cont_store=cont_store,
        sched_registry=sched_registry,
        wait_registry=wait_registry,
        resume_bus=resume_bus,
        resume_router=resume_router,
        wakeup_queue=wakeup_queue,
        trigger_store=trigger_store,
        trigger_engine=trigger_engine,
        trigger_service=trigger_service,
        execution=execution,
        planner_service=planner_service,
        doc_store=doc_store,
        kv_hot=kv_hot,
        state_store=state_store,
        artifacts=artifacts,
        artifact_index=artifact_index,
        registration_manifest_store=registration_manifest_store,
        registration_service=registration_service,
        global_indices=global_indices,
        kb_backend=kb_backend,
        viz_service=viz_service,
        eventlog=eventlog,
        memory_factory=memory_factory,
        llm=llm_service,
        llm_observation_sink=llm_observation_sink,
        llm_observation_path=llm_observation_path,
        llm_observation_store=llm_observation_store,
        embed_service=embed_service,
        web_search=web_search,
        mcp=mcp,
        run_store=run_store,
        run_manager=run_manager,
        run_cancellation_registry=run_cancellation_registry,
        session_store=session_store,
        secrets=secrets,
        event_bus=None,
        authn=authn,
        authz=authz,
        redactor=None,
        metering=metering,
        rate_limiter=rate_limiter,
        tracer=EventLogTracer(event_log=eventlog, event_hub=event_hub)
        if eventlog is not None
        else NoopTracer(),
        agent_event_registry=agent_event_registry,
        settings=cfg,
    )

    # channel ingress (after container is built to avoid circular dependency)
    container.channel_ingress = ChannelIngress(
        container=container, logger=logger_factory.for_channel()
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
