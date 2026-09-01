from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.observability.logger import StdLoggerService
from aethergraph.services.agent_state import CanonicalAgentStateFacade
from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
from aethergraph.services.canonical_kv import CanonicalKeyValueFacade
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.image_service import ImageGenerationService
from aethergraph.services.llm.service import LLMService
from aethergraph.services.memory.canonical_public import CanonicalPublicMemoryFacade
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry

if TYPE_CHECKING:
    from .node_context import NodeContext


class AgentContextFactory(Protocol):
    """Construct one recipient-bound context without changing runtime ownership."""

    def __call__(self, context: NodeContext, *, agent_id: str) -> NodeContext: ...


@dataclass
class NodeServices:
    channels: ChannelBus
    continuation_store: AsyncContinuationStore
    artifact_store: CanonicalPublicArtifactFacade
    wait_registry: WaitRegistry | None = None
    clock: SystemClock | None = None
    logger: StdLoggerService | None = (
        None  # StdLoggerService.for_node_ctx() will be used in NodeContext
    )
    kv: CanonicalKeyValueFacade | None = None
    memory: Any | None = None  # MemoryFactory (for cross-session needs)
    memory_facade: CanonicalPublicMemoryFacade | None = None
    agent_state: CanonicalAgentStateFacade | None = None
    viz: VizFacade | None = None  # VizFacade
    llm: LLMService | None = None  # LLMService
    embedding: EmbeddingService | None = None  # EmbeddingService
    image_model: ImageGenerationService | None = None  # ImageGenerationService
    runner: RunFacade | None = None  # RunFacade for child run orchestration
    triggers: TriggerFacade | None = None  # TriggerFacade for firing triggers from nodes
    registry: RegistryFacade | None = None  # Scope-bound runtime registry facade
    agent_context_factory: AgentContextFactory | None = None
