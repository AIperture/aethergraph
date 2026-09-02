from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Protocol

from aethergraph.contracts.json_values import JsonValue

if TYPE_CHECKING:
    from aethergraph.services.llm.contracts import ModelRequest
    from aethergraph.services.llm.streaming import ModelEvent
    from aethergraph.services.llm.tool_calling import (
        ModelResponse,
        ToolCallRequest,
        ToolCallResponse,
    )
    from aethergraph.services.llm.types import (
        ChatOutputFormat,
        EmbeddingResult,
        ImageFormat,
        ImageGenerationResult,
        ImageResponseFormat,
        LLMRequestEstimate,
        PromptCacheRequest,
        StructuredOutputRequest,
    )


class LLMClientProtocol(Protocol):
    def estimate(self, request: ModelRequest) -> LLMRequestEstimate: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...

    def generate_stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    def estimate_chat_request(
        self,
        messages: list[dict[str, JsonValue]],
        *,
        max_output_tokens: int | None,
        structured_output: StructuredOutputRequest | None = None,
        tool_request: ToolCallRequest | None = None,
        json_schema: dict[str, JsonValue] | None = None,
        tools: list[dict[str, JsonValue]] | None = None,
        model: str | None = None,
    ) -> LLMRequestEstimate: ...

    async def chat(
        self,
        messages: list[dict[str, JsonValue]],
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat = "text",
        structured_output: StructuredOutputRequest | None = None,
        tool_request: ToolCallRequest | None = None,
        prompt_cache: PromptCacheRequest | None = None,
        model: str | None = None,
    ) -> tuple[str | ToolCallResponse, dict[str, int]]: ...


class EmbeddingClientProtocol(Protocol):
    async def embed_result(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        azure_api_version: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
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
            dimensions: Optional requested output-vector dimensionality.
            extra_body: Explicit provider extension body.
            azure_api_version: Optional Azure API version.
            user_id: Optional metering user identity.
            org_id: Optional metering organization identity.
            run_id: Optional metering run identity.
            graph_id: Optional metering graph identity.

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
        dimensions: int | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        azure_api_version: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
    ) -> list[list[float]]: ...

    async def embed_one(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        azure_api_version: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
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
        Returns:
            ImageGenerationResult: Normalized images, provider usage, and raw data.

        Notes:
            Implementations must not select a different endpoint after request
            inspection or transport failure.
        """

        ...
