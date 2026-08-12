from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from aethergraph.services.llm.contracts import ModelRequest
    from aethergraph.services.llm.streaming import ModelEvent
    from aethergraph.services.llm.tool_calling import ModelResponse, ToolCallResponse


class LLMClientProtocol(Protocol):
    def estimate(self, request: ModelRequest) -> Any: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...

    def generate_stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    def estimate_chat_request(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> Any: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> tuple[str | ToolCallResponse, dict[str, int]]: ...
    async def raw(
        self,
        *,
        method: str = "POST",
        path: str | None = None,
        url: str | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        return_response: bool = False,
    ) -> Any: ...


class EmbeddingClientProtocol(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs,
    ) -> list[list[float]]: ...

    async def embed_one(
        self,
        text: str,
        *,
        model: str | None = None,
        **kwargs,
    ) -> list[float]: ...
