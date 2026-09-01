"""Closed public context contract for statically validated authored Tools."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from aethergraph.contracts.json_values import JsonValue

if TYPE_CHECKING:
    from aethergraph.contracts.services.llm import (
        EmbeddingClientProtocol,
        ImageGenerationClientProtocol,
        LLMClientProtocol,
    )
    from aethergraph.services.agent_state import AgentStateBackend, CanonicalAgentStateHandle
    from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
    from aethergraph.services.canonical_kv import CanonicalKeyValueFacade
    from aethergraph.services.channel.session import ChannelSession
    from aethergraph.services.llm.providers import Provider
    from aethergraph.services.memory.canonical_public import CanonicalPublicMemoryFacade
    from aethergraph.services.runner.facade import RunFacade
    from aethergraph.services.scope.scope import Scope, ScopeLevel
    from aethergraph.services.triggers.trigger_facade import TriggerFacade
    from aethergraph.services.viz.facade import VizFacade

StateT = TypeVar("StateT")


class NodeContextProtocol(Protocol):
    """Expose canonical Tool services without implicit dynamic attributes."""

    run_id: str
    session_id: str
    graph_id: str
    node_id: str

    def logger(self) -> logging.Logger: ...
    def channel(self, channel_key: str | None = None) -> ChannelSession: ...
    def artifacts(self) -> CanonicalPublicArtifactFacade: ...
    def kv(self) -> CanonicalKeyValueFacade: ...
    def memory(self) -> CanonicalPublicMemoryFacade: ...
    def runner(self) -> RunFacade: ...
    def triggers(self) -> TriggerFacade: ...
    def viz(self) -> VizFacade: ...
    def llm(
        self,
        profile: str = "default",
        *,
        provider: Provider | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
    ) -> LLMClientProtocol: ...
    def embedding(self, profile: str = "default") -> EmbeddingClientProtocol: ...
    def image_model(self, profile: str = "default") -> ImageGenerationClientProtocol: ...
    def state(
        self,
        key: str,
        *,
        model: type[StateT] | None = None,
        default_factory: Callable[[], StateT] | None = None,
        level: ScopeLevel | None = None,
        scope: Scope | None = None,
        backend: AgentStateBackend = "hybrid",
        tags: list[str] | None = None,
        meta: dict[str, JsonValue] | None = None,
        kind: str = "state.snapshot",
    ) -> CanonicalAgentStateHandle[StateT]: ...
    def svc(self, name: str) -> Any: ...


__all__ = ["NodeContextProtocol"]
