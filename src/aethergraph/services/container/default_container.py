from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- core services ----
from aethergraph.config.config import AppSettings

# ---- optional services (not used by default) ----
from aethergraph.contracts.services.llm import LLMClientProtocol

# ---- scheduler ---- TODO: move to a separate server to handle scheduling across threads/processes
from aethergraph.contracts.services.state_stores import GraphStateStore
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler

# ---- artifact services ----
from aethergraph.services.auth.dev import AllowAllAuthz, DevTokenAuthn
from aethergraph.services.channel.channel_bus import ChannelBus

# ---- channel services ----
from aethergraph.services.channel.factory import build_bus, make_channel_adapters_from_env
from aethergraph.services.channel.ingress import ChannelIngress
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import (
    FSContinuationStore,  # AsyncContinuationStore
)
from aethergraph.services.eventbus.inmem import InMemoryEventBus

# ---- kv services ----
from aethergraph.services.kv.ephemeral import EphemeralKV
from aethergraph.services.kv.sqlite_kv import SQLiteKV
from aethergraph.services.llm.factory import build_llm_clients
from aethergraph.services.llm.service import LLMService
from aethergraph.services.logger.std import LoggingConfig, StdLoggerService
from aethergraph.services.mcp.service import MCPService

# ---- memory services ----
from aethergraph.services.memory.factory import MemoryFactory
from aethergraph.services.memory.hotlog_kv import KVHotLog
from aethergraph.services.memory.indices import KVIndices
from aethergraph.services.memory.persist_fs import FSPersistence
from aethergraph.services.metering.noop import NoopMetering
from aethergraph.services.prompts.file_store import FilePromptStore
from aethergraph.services.rag.chunker import TextSplitter
from aethergraph.services.rag.facade import RAGFacade

# ---- RAG components ----
from aethergraph.services.rag.index_factory import create_vector_index
from aethergraph.services.redactor.simple import RegexRedactor  # Simple PII redactor
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.resume.multi_scheduler_resume_bus import MultiSchedulerResumeBus
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.services.secrets.env import EnvSecrets
from aethergraph.services.tracing.noop import NoopTracer
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.wakeup.memory_queue import ThreadSafeWakeupQueue
from aethergraph.storage.factory import (
    build_artifact_index,
    build_artifact_store,
    build_continuation_store,
    build_graph_state_store,
)

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
    "kv_durable",
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

    # schedulers
    schedulers: dict[str, Any]

    # core
    registry: UnifiedRegistry
    logger: StdLoggerService
    clock: SystemClock

    # channels and interactions
    channels: ChannelBus

    # continuations and resume
    cont_store: FSContinuationStore
    sched_registry: SchedulerRegistry
    wait_registry: WaitRegistry
    resume_bus: MultiSchedulerResumeBus
    resume_router: ResumeRouter
    wakeup_queue: ThreadSafeWakeupQueue
    state_store: GraphStateStore

    # storage and artifacts
    kv_hot: EphemeralKV
    kv_durable: SQLiteKV
    artifacts: AsyncArtifactStore
    artifact_index: AsyncArtifactIndex

    # memory
    memory_factory: MemoryFactory

    # optional llm service
    llm: LLMClientProtocol | None = None
    rag: RAGFacade | None = None
    mcp: MCPService | None = None

    # optional services (not used by default)
    event_bus: InMemoryEventBus | None = None
    prompts: FilePromptStore | None = None
    authn: DevTokenAuthn | None = None
    authz: AllowAllAuthz | None = None
    redactor: RegexRedactor | None = None
    metering: NoopMetering | None = None
    tracer: NoopTracer | None = None
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
    if "root" is provided, use it as the base directory for storage; else use from cfg/root.
    if cfg is not provided, load from default AppSettings.
    """
    if cfg is None:
        from aethergraph.config.context import set_current_settings
        from aethergraph.config.loader import load_settings

        cfg = load_settings()
        set_current_settings(cfg)

    root = root or cfg.root
    # override root in cfg to match
    cfg.root = root

    # we use user specified root if provided, else from config/env
    root_p = Path(root).resolve() if root else Path(cfg.root).resolve()
    (root_p / "kv").mkdir(parents=True, exist_ok=True)
    (root_p / "index").mkdir(parents=True, exist_ok=True)
    (root_p / "memory").mkdir(parents=True, exist_ok=True)

    # core services
    logger_factory = StdLoggerService.build(
        LoggingConfig.from_cfg(cfg, log_dir=str(root_p / "logs"))
    )
    clock = SystemClock()
    registry = UnifiedRegistry()

    # continuations and resume
    cont_store = build_continuation_store(cfg)

    sched_registry = SchedulerRegistry()
    wait_registry = WaitRegistry()
    resume_bus = MultiSchedulerResumeBus(
        registry=sched_registry, store=cont_store, logger=logger_factory.for_run()
    )
    resume_router = ResumeRouter(
        store=cont_store,
        runner=resume_bus,
        logger=logger_factory.for_run(),
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
    channel_adapters = make_channel_adapters_from_env(cfg)
    channels = build_bus(
        channel_adapters,
        default="console:stdin",
        logger=logger_factory.for_run(),
        resume_router=resume_router,
        cont_store=cont_store,
    )

    # storage and artifacts
    kv_hot = EphemeralKV()
    kv_durable = SQLiteKV(str(root_p / "kv" / "kv.sqlite"))

    artifacts = build_artifact_store(cfg)
    artifact_index = build_artifact_index(cfg)

    # memory
    hotlog = KVHotLog(kv=kv_hot)
    persistence = FSPersistence(base_dir=str(root_p / "memory"))
    indices = KVIndices(kv=kv_durable, hot_ttl_s=7 * 24 * 3600)

    # optional services
    secrets = (
        EnvSecrets()
    )  # get secrets from env vars -- for local development; in prod, use a proper secrets manager
    llm_clients = build_llm_clients(cfg.llm, secrets)  # return {profile: GenericLLMClient}
    llm_service = LLMService(clients=llm_clients) if llm_clients else None

    rag_cfg = cfg.rag
    vec_index = create_vector_index(
        backend=rag_cfg.backend, index_path=str(root_p / "rag" / "rag_index"), dim=rag_cfg.dim
    )

    rag_facade = RAGFacade(
        corpus_root=str(root_p / "rag" / "rag_corpora"),
        artifacts=artifacts,
        embed_client=llm_service.get("default"),
        llm_client=llm_service.get("default"),
        index_backend=vec_index,
        chunker=TextSplitter(),
        logger=logger_factory.for_run(),
    )
    mcp = MCPService()  # empty MCP service; users can register clients as needed

    memory_factory = MemoryFactory(
        hotlog=hotlog,
        persistence=persistence,
        indices=indices,
        artifacts=artifacts,
        hot_limit=int(cfg.memory.hot_limit),
        hot_ttl_s=int(cfg.memory.hot_ttl_s),
        default_signal_threshold=float(cfg.memory.signal_threshold),
        logger=logger_factory.for_run(),
        llm_service=llm_service.get("default") if llm_service else None,
        rag_facade=rag_facade,
    )

    container = DefaultContainer(
        root=str(root_p),
        schedulers=schedulers,
        registry=registry,
        logger=logger_factory,
        clock=clock,
        channels=channels,
        cont_store=cont_store,
        sched_registry=sched_registry,
        wait_registry=wait_registry,
        resume_bus=resume_bus,
        resume_router=resume_router,
        wakeup_queue=wakeup_queue,
        kv_hot=kv_hot,
        kv_durable=kv_durable,
        state_store=state_store,
        artifacts=artifacts,
        artifact_index=artifact_index,
        memory_factory=memory_factory,
        llm=llm_service,
        rag=rag_facade,
        mcp=mcp,
        secrets=secrets,
        event_bus=None,
        prompts=None,
        authn=None,
        authz=None,
        redactor=None,
        metering=None,
        tracer=None,
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
