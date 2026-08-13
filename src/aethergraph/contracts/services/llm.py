from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from aethergraph.services.llm.contracts import ModelRequest
    from aethergraph.services.llm.streaming import ModelEvent
    from aethergraph.services.llm.tool_calling import ModelResponse, ToolCallResponse
    from aethergraph.services.llm.types import (
        EmbeddingResult,
        ImageFormat,
        ImageGenerationResult,
        ImageResponseFormat,
    )


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
    async def embed_result(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Embed a batch and retain typed provider usage.

        Intro:
            Defines the canonical embedding result boundary consumed by runtime
            accounting before vector-only compatibility projection.

        Examples:
            Embed one text with usage:
                ```python
                result = await client.embed_result(["hello"])
                ```

            Override the configured model:
                ```python
                result = await client.embed_result(
                    ["hello", "world"], model="embedding-v2"
                )
                ```

        Args:
            self: Configured embedding client.
            texts: Ordered text inputs.
            model: Optional per-call model override.
            **kwargs: Bounded provider options and metering dimensions.

        Returns:
            EmbeddingResult: Ordered vectors and operation-specific usage.

        Notes:
            The `embed()` compatibility method may project only the vectors, but
            it must delegate through this lifecycle so usage is not discarded.
        """

        ...

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


class ImageGenerationClientProtocol(Protocol):
    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int | None = None,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        output_format: ImageFormat | None = None,
        response_format: ImageResponseFormat | None = None,
        background: str | None = None,
        input_images: list[str] | None = None,
        azure_api_version: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate images through one configured image-operation client.

        Intro:
            Defines the provider-neutral request surface implemented by exact-
            bound image clients returned from `NodeContext.image_model()`.

        Examples:
            Generate with profile defaults:
                ```python
                result = await client.generate_image("A quiet observatory")
                ```

            Supply image-editing inputs:
                ```python
                result = await client.generate_image(
                    "Make the sky violet",
                    input_images=["data:image/png;base64,aW1hZ2U="],
                )
                ```

        Args:
            self: Configured image-generation client.
            prompt: Text description of the requested output.
            model: Optional per-call model override.
            n: Optional output count overriding the profile default.
            size: Optional image dimensions.
            quality: Optional provider quality mode.
            style: Optional provider style mode.
            output_format: Optional encoded image format.
            response_format: Optional response transport format.
            background: Optional provider background mode.
            input_images: Optional source-image data URLs.
            azure_api_version: Optional Azure Images API version.
            **kwargs: Bounded adapter-private options.

        Returns:
            ImageGenerationResult: Normalized images, provider usage, and raw data.

        Notes:
            Implementations must not select a different endpoint after request
            inspection or transport failure.
        """

        ...
