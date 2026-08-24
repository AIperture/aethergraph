from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aethergraph.observability.logger import StdLoggerService
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.image_service import ImageGenerationService
from aethergraph.services.llm.service import LLMService
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry


@dataclass
class NodeServices:
    channels: ChannelBus
    continuation_store: Any
    artifact_store: Any  # e.g., ArtifactFacadeAsync
    wait_registry: WaitRegistry | None = None
    clock: SystemClock | None = None
    logger: StdLoggerService | None = (
        None  # StdLoggerService.for_node_ctx() will be used in NodeContext
    )
    kv: Any | None = None
    memory: Any | None = None  # MemoryFactory (for cross-session needs)
    memory_facade: Any | None = None  # bound public memory for this node
    agent_state: Any | None = None  # bound canonical agent-state facade
    viz: VizFacade | None = None  # VizFacade
    llm: LLMService | None = None  # LLMService
    embedding: EmbeddingService | None = None  # EmbeddingService
    image_model: ImageGenerationService | None = None  # ImageGenerationService
    runner: RunFacade | None = None  # RunFacade for child run orchestration
    triggers: TriggerFacade | None = None  # TriggerFacade for firing triggers from nodes
    registry: RegistryFacade | None = None  # Scope-bound runtime registry facade
    agent_context_factory: Callable[..., Any] | None = None  # canonical context rebinding
