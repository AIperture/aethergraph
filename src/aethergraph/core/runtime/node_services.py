
from dataclasses import dataclass
from typing import Any, Optional

from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.mcp.service import MCPService
from aethergraph.services.rag.facade import RAGFacade
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.logger.std import StdLoggerService 
    

@dataclass
class NodeServices:
    channels: ChannelBus
    continuation_store: FSContinuationStore
    artifact_store: Any                   # e.g., ArtifactFacadeAsync
    wait_registry: Optional[WaitRegistry] = None
    clock: Optional[SystemClock] = None
    logger: Optional[StdLoggerService] = None  # StdLoggerService.for_node_ctx() will be used in NodeContext
    kv: Optional[Any] = None
    memory: Optional[Any] = None          # MemoryFactory (for cross-session needs)
    memory_facade: Optional[MemoryFacade] = None  # bound memory for this node
    llm: Optional[LLMClientProtocol] = None             # LLMService
    rag: Optional[RAGFacade] = None             # RAGService
    mcp: Optional[MCPService] = None                    # MCPService