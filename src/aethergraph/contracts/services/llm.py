from collections.abc import Sequence
from typing import Any, Protocol


class LLMClientProtocol(Protocol):
    async def chat(self, messages: list[dict[str, Any]], **kw) -> tuple[str, dict[str, int]]: ...
    async def embed(self, texts: list[str], **kw) -> list[list[float]]: ...
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
    ) -> list[list[float]]:
        """
        Batch-embed texts. Returns one vector per text.
        """

    async def embed_one(
        self,
        text: str,
        *,
        model: str | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Convenience method: embed a single string.
        Default implementation can call embed([text])[0].
        """
