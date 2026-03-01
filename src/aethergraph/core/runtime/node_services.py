from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.knowledge.node_kb import NodeKB
from aethergraph.services.llm.service import LLMService
from aethergraph.services.logger.std import StdLoggerService
from aethergraph.services.mcp.service import MCPService
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.planning.planner_service import PlannerService
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.skills.skill_registry import SkillRegistry
from aethergraph.services.triggers.trigger_facade import TriggerFacade
from aethergraph.services.viz.facade import VizFacade
from aethergraph.services.waits.wait_registry import WaitRegistry
from aethergraph.services.websearch.facade import WebSearchFacade


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
    mcp: MCPService | None = None  # MCPService
    runner: RunFacade | None = None  # RunFacade for child run orchestration
    indices: ScopedIndices | None = None  # ScopedIndices for this node
    execution: Any | None = None  # ExecutionService
    planner_service: PlannerService | None = None  # PlannerService
    skills: SkillRegistry | None = None  # SkillRegistry
    kb: NodeKB | None = None  # NodeKB
    triggers: TriggerFacade | None = None  # TriggerFacade for firing triggers from nodes
    web_search: WebSearchFacade | None = None  # Web search facade
