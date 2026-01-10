from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aethergraph.services.indices.scoped_indices import ScopedIndices

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.llm.service import LLMService
from aethergraph.services.logger.std import StdLoggerService
from aethergraph.services.mcp.service import MCPService
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.rag.node_rag import NodeRAG
from aethergraph.services.viz.facade import VizFacade
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
    viz: VizFacade | None = None  # VizFacade
    llm: LLMService | None = None  # LLMService
    rag: NodeRAG | None = None  # RAGService
    mcp: MCPService | None = None  # MCPService
    run_manager: RunManager | None = None  # RunManager
    indices: ScopedIndices | None = None  # ScopedIndices for this node
