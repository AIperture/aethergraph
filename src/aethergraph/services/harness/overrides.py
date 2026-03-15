from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from aethergraph.contracts.services.execution import CodeExecutionResult
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.services.llm.service import LLMService
from aethergraph.services.websearch.facade import WebSearchFacade

_active_registry: ContextVar[OperatorOverrideRegistry | None] = ContextVar(
    "aethergraph_harness_operator_registry",
    default=None,
)


@dataclass
class OperatorOverride:
    operator_type: str
    operation: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    mode: str = "fixed_result"
    result: Any = None
    sequence: list[Any] = field(default_factory=list)
    error_type: str = "RuntimeError"
    error_message: str = "operator override error"
    latency_ms: int = 0
    _cursor: int = 0

    def matches(
        self, *, operator_type: str, operation: str, graph_id: str | None, node_id: str | None
    ) -> bool:
        if self.operator_type != operator_type:
            return False
        if self.operation is not None and self.operation != operation:
            return False
        if self.graph_id is not None and self.graph_id != graph_id:
            return False
        if self.node_id is not None and self.node_id != node_id:
            return False
        return True

    def next_payload(self) -> Any:
        if self.mode == "sequence":
            if self._cursor >= len(self.sequence):
                raise RuntimeError("operator override sequence exhausted")
            item = self.sequence[self._cursor]
            self._cursor += 1
            return item
        return self.result


@dataclass
class OperatorOverrideRegistry:
    overrides: list[OperatorOverride] = field(default_factory=list)

    def add(self, override: OperatorOverride) -> None:
        self.overrides.append(override)

    def resolve(
        self,
        *,
        operator_type: str,
        operation: str,
        graph_id: str | None,
        node_id: str | None,
    ) -> OperatorOverride | None:
        for override in self.overrides:
            if override.matches(
                operator_type=operator_type,
                operation=operation,
                graph_id=graph_id,
                node_id=node_id,
            ):
                return override
        return None


def active_operator_overrides() -> OperatorOverrideRegistry | None:
    return _active_registry.get()


@contextmanager
def use_operator_overrides(registry: OperatorOverrideRegistry | None):
    token = _active_registry.set(registry)
    try:
        yield
    finally:
        _active_registry.reset(token)


async def _apply_override(override: OperatorOverride, *, fallback) -> Any:
    if override.latency_ms:
        await asyncio.sleep(max(0, override.latency_ms) / 1000.0)
    if override.mode == "delegate":
        return await fallback()
    if override.mode == "error":
        raise RuntimeError(override.error_message)
    return override.next_payload()


class HarnessLLMService:
    def __init__(
        self,
        inner: LLMService,
        *,
        registry: OperatorOverrideRegistry,
        graph_id: str | None,
        node_id: str,
    ):
        self._inner = inner
        self._registry = registry
        self._graph_id = graph_id
        self._node_id = node_id

    def get(self, name: str = "default"):
        client = self._inner.get(name)
        return HarnessLLMClient(
            client,
            registry=self._registry,
            graph_id=self._graph_id,
            node_id=self._node_id,
        )

    def configure_profile(self, *args, **kwargs):
        client = self._inner.configure_profile(*args, **kwargs)
        return HarnessLLMClient(
            client,
            registry=self._registry,
            graph_id=self._graph_id,
            node_id=self._node_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class HarnessLLMClient:
    def __init__(
        self, inner: Any, *, registry: OperatorOverrideRegistry, graph_id: str | None, node_id: str
    ):
        self._inner = inner
        self._registry = registry
        self._graph_id = graph_id
        self._node_id = node_id

    async def chat(self, *args, **kwargs):
        override = self._registry.resolve(
            operator_type="llm",
            operation="chat",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.chat(*args, **kwargs)
        payload = await _apply_override(
            override, fallback=lambda: self._inner.chat(*args, **kwargs)
        )
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload
        if isinstance(payload, dict):
            return payload.get("text", ""), payload.get("usage", {})
        return str(payload), {}

    async def chat_stream(self, *args, **kwargs):
        override = self._registry.resolve(
            operator_type="llm",
            operation="chat_stream",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.chat_stream(*args, **kwargs)
        payload = await _apply_override(
            override, fallback=lambda: self._inner.chat_stream(*args, **kwargs)
        )
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload
        if isinstance(payload, dict):
            return payload.get("text", ""), payload.get("usage", {})
        return str(payload), {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class HarnessExecutionService:
    def __init__(
        self, inner: Any, *, registry: OperatorOverrideRegistry, graph_id: str | None, node_id: str
    ):
        self._inner = inner
        self._registry = registry
        self._graph_id = graph_id
        self._node_id = node_id

    async def execute(self, request):
        override = self._registry.resolve(
            operator_type="execution",
            operation="execute",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.execute(request)
        payload = await _apply_override(override, fallback=lambda: self._inner.execute(request))
        if isinstance(payload, CodeExecutionResult):
            return payload
        if isinstance(payload, dict):
            return CodeExecutionResult(**payload)
        raise RuntimeError("Execution override must return CodeExecutionResult or a dict payload")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class HarnessWebSearchFacade:
    def __init__(
        self,
        inner: WebSearchFacade,
        *,
        registry: OperatorOverrideRegistry,
        graph_id: str | None,
        node_id: str,
    ):
        self._inner = inner
        self._registry = registry
        self._graph_id = graph_id
        self._node_id = node_id

    async def search(self, *args, **kwargs):
        override = self._registry.resolve(
            operator_type="web_search",
            operation="search",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.search(*args, **kwargs)
        return await _apply_override(override, fallback=lambda: self._inner.search(*args, **kwargs))

    async def fetch(self, *args, **kwargs):
        override = self._registry.resolve(
            operator_type="web_search",
            operation="fetch",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.fetch(*args, **kwargs)
        return await _apply_override(override, fallback=lambda: self._inner.fetch(*args, **kwargs))

    async def search_and_fetch(self, *args, **kwargs):
        override = self._registry.resolve(
            operator_type="web_search",
            operation="search_and_fetch",
            graph_id=self._graph_id,
            node_id=self._node_id,
        )
        if override is None:
            return await self._inner.search_and_fetch(*args, **kwargs)
        return await _apply_override(
            override, fallback=lambda: self._inner.search_and_fetch(*args, **kwargs)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_node_services(
    services: NodeServices, *, graph_id: str | None, node_id: str
) -> NodeServices:
    registry = active_operator_overrides()
    if registry is None:
        return services
    if services.llm is not None:
        services.llm = HarnessLLMService(
            services.llm,
            registry=registry,
            graph_id=graph_id,
            node_id=node_id,
        )
    if services.execution is not None:
        services.execution = HarnessExecutionService(
            services.execution,
            registry=registry,
            graph_id=graph_id,
            node_id=node_id,
        )
    if services.web_search is not None:
        services.web_search = HarnessWebSearchFacade(
            services.web_search,
            registry=registry,
            graph_id=graph_id,
            node_id=node_id,
        )
    return services
