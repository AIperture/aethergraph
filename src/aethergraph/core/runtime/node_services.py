from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.logger.std import StdLoggerService
from aethergraph.services.mcp.service import MCPService
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.waits.wait_registry import WaitRegistry


@dataclass
class NodeServices:
    channels: ChannelBus
    continuation_store: FSContinuationStore
    artifact_store: Any  # e.g., ArtifactFacadeAsync
    wait_registry: WaitRegistry | None = None
    clock: SystemClock | None = None
    logger: StdLoggerService | None = (
        None  # StdLoggerService.for_node_ctx() will be used in NodeContext
    )
    kv: Any | None = None
    memory: Any | None = None  # MemoryFactory (for cross-session needs)
    memory_facade: MemoryFacade | None = None  # bound memory for this node
    llm: LLMClientProtocol | None = None  # LLMService
    rag: RAGFacade | None = None  # RAGService
    mcp: MCPService | None = None  # MCPService
