from __future__ import annotations
from dataclasses import dataclass, field 
from typing import Any, Dict, Mapping, Optional
import os 
from pathlib import Path 

# ---- scheduler ---- TODO: move to a separate server to handle scheduling across threads/processes
from aethergraph.contracts.services.state_stores import GraphStateStore
from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler

# ---- core services ----
from aethergraph.config.config import AppSettings
from aethergraph.services.rag.index_factory import create_vector_index
from aethergraph.services.state_stores.json_store import JsonGraphStateStore
from aethergraph.services.mcp.service import MCPService
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.logger.std import LoggingConfig, StdLoggerService
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore    # AsyncContinuationStore
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.resume.multi_scheduler_resume_bus import MultiSchedulerResumeBus 
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.wakeup.memory_queue import ThreadSafeWakeupQueue


# ---- kv services ----
from aethergraph.services.kv.ephemeral import EphemeralKV
from aethergraph.services.kv.sqlite_kv import SQLiteKV 

# ---- artifact services ----
from aethergraph.services.artifacts.fs_store import FSArtifactStore # AsyncArtifactStore 
from aethergraph.services.artifacts.jsonl_index import JsonlArtifactIndex # AsyncArtifactIndex

# ---- memory services ----
from aethergraph.services.memory.factory import MemoryFactory
from aethergraph.services.memory.hotlog_kv import KVHotLog 
from aethergraph.services.memory.persist_fs import  FSPersistence
from aethergraph.services.memory.indices import KVIndices

# ---- channel services ----
from aethergraph.services.channel.factory import make_channel_adapters_from_env, build_bus
from aethergraph.services.channel.channel_bus import ChannelBus 

# ---- optional services (not used by default) ----
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.llm.factory import build_llm_clients
from aethergraph.services.llm.service import LLMService

# ---- RAG components ----
from aethergraph.services.rag.index.faiss_index import FAISSVectorIndex # FAISS-based vector index
from aethergraph.services.rag.index.sqlite_index import SQLiteVectorIndex # SQLite-based vector index
from aethergraph.services.rag.chunker import TextSplitter

from aethergraph.services.eventbus.inmem import InMemoryEventBus
from aethergraph.services.prompts.file_store import FilePromptStore
from aethergraph.services.auth.dev import DevTokenAuthn, AllowAllAuthz 
from aethergraph.services.redactor.simple import RegexRedactor # Simple PII redactor
from aethergraph.services.metering.noop import NoopMetering 
from aethergraph.services.tracing.noop import NoopTracer
from aethergraph.services.secrets.env import EnvSecrets 


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
    schedulers: Dict[str, Any]
    
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
    artifacts: FSArtifactStore
    artifact_index: JsonlArtifactIndex 

    # memory 
    memory_factory: MemoryFactory

    # optional llm service
    llm: Optional[LLMClientProtocol] = None
    rag: Optional[RAGFacade] = None
    mcp: Optional[MCPService] = None

    # optional services (not used by default)
    event_bus: Optional[InMemoryEventBus] = None
    prompts: Optional[FilePromptStore] = None
    authn: Optional[DevTokenAuthn] = None
    authz: Optional[AllowAllAuthz] = None
    redactor: Optional[RegexRedactor] = None
    metering: Optional[NoopMetering] = None
    tracer: Optional[NoopTracer] = None
    secrets: Optional[EnvSecrets] = None

    # extensible services
    ext_services: Dict[str, Any] = field(default_factory=dict)

    # settings -- not a service, but useful to have around
    settings: Optional[AppSettings] = None

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
        from aethergraph.config.loader import load_settings
        from aethergraph.config.context import set_current_settings
        cfg = load_settings()
        set_current_settings(cfg)


    root = root or cfg.root
    # override root in cfg to match
    cfg.root = root
    
    # we use user specified root if provided, else from config/env
    root_p = Path(root).resolve() if root else Path(cfg.root).resolve()
    (root_p / "kv").mkdir(parents=True, exist_ok=True)
    (root_p / "continuations").mkdir(parents=True, exist_ok=True)
    (root_p / "index").mkdir(parents=True, exist_ok=True)
    (root_p / "memory").mkdir(parents=True, exist_ok=True)
    (root_p / "graph_states").mkdir(parents=True, exist_ok=True)

    # core services
    logger_factory = StdLoggerService.build(LoggingConfig.from_cfg(cfg, log_dir=str(root_p / "logs")))
    clock = SystemClock()
    registry = UnifiedRegistry()

    # continuations and resume
    cont_store = FSContinuationStore(root=str(root_p / "continuations"), secret=os.urandom(32))
    sched_registry = SchedulerRegistry()
    wait_registry = WaitRegistry()
    resume_bus = MultiSchedulerResumeBus(registry=sched_registry, store=cont_store, logger=logger_factory.for_run())
    resume_router = ResumeRouter(store=cont_store, runner=resume_bus, logger=logger_factory.for_run(), wait_registry=wait_registry)
    wakeup_queue = ThreadSafeWakeupQueue() # TODO: this is a placeholder, not fully implemented 
    state_store = JsonGraphStateStore(root=str(root_p / "graph_states"))

    # global scheduler
    global_sched = GlobalForwardScheduler(
        registry=sched_registry,
        global_max_concurrency=None,  # TODO: make configurable
        logger=logger_factory.for_scheduler()
    )
    schedulers = {"global": global_sched,
                  "registry": sched_registry,}


    # channels
    channel_adapters = make_channel_adapters_from_env(cfg) 
    channels = build_bus(channel_adapters, default="console:stdin", logger=logger_factory.for_run(), resume_router=resume_router, cont_store=cont_store)

    # storage and artifacts
    kv_hot = EphemeralKV()
    kv_durable = SQLiteKV(str(root_p / "kv" / "kv.sqlite"))
    artifacts = FSArtifactStore(str(root_p / "artifacts"))                        # async wrapper over FileArtifactStoreSync
    artifact_index = JsonlArtifactIndex(str(root_p / "index" / "artifacts.jsonl"))

    # memory
    hotlog = KVHotLog(kv=kv_hot)
    persistence = FSPersistence(base_dir=str(root_p / "memory"))
    indices = KVIndices(kv=kv_durable, hot_ttl_s=7*24*3600)

    # optional services
    secrets = EnvSecrets() # get secrets from env vars -- for local development; in prod, use a proper secrets manager
    llm_clients = build_llm_clients(cfg.llm, secrets) # return {profile: GenericLLMClient}
    llm_service = LLMService(clients=llm_clients) if llm_clients else None
    
    rag_cfg = cfg.rag
    vec_index = create_vector_index(backend=rag_cfg.backend, index_path=str(root_p / "rag" / "rag_index"), dim=rag_cfg.dim)

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


    return DefaultContainer(
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


# Singleton (used unless the host sets their own)
DEFAULT_CONTAINER: Optional[DefaultContainer] = None

def get_container() -> DefaultContainer:
    global DEFAULT_CONTAINER
    if DEFAULT_CONTAINER is None:
        DEFAULT_CONTAINER = build_default_container()
    return DEFAULT_CONTAINER

def set_container(c: DefaultContainer) -> None:
    global DEFAULT_CONTAINER
    DEFAULT_CONTAINER = c