"""Exact image-generation endpoint-adapter dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, field
from typing import Any

from aethergraph.services.llm.adapters.azure_images import AzureImagesAdapter
from aethergraph.services.llm.adapters.gemini_images import GeminiImagesAdapter
from aethergraph.services.llm.adapters.openai_images import OpenAIImagesAdapter
from aethergraph.services.llm.provider_transport import ProviderCallResult
from aethergraph.services.llm.types import (
    ImageFormat,
    ImageGenerationResult,
    ImageResponseFormat,
    LLMUnsupportedFeatureError,
)

ImageAdapterResult = ProviderCallResult[ImageGenerationResult]
ImageHandler = Callable[[Any, "ImageAdapterInvocation"], Awaitable[ImageAdapterResult]]


@dataclass(frozen=True)
class ImageAdapterInvocation:
    """Carry one prepared single-attempt image-generation invocation.

    Intro:
        Freezes provider-neutral image facade state before exact endpoint dispatch
        and detaches mutable input-image and option collections.

    Examples:
        Build a text-only image request:
            ```python
            invocation = ImageAdapterInvocation(
                prompt="A quiet observatory",
                model="image-model",
                n=1,
                size=None,
                quality=None,
                style=None,
                output_format=None,
                response_format=None,
                background=None,
                input_images=(),
                azure_api_version=None,
            )
            ```

        Build an image-conditioned request:
            ```python
            invocation = ImageAdapterInvocation(
                prompt="Make the sky violet",
                model="image-model",
                n=1,
                size="1024x1024",
                quality="high",
                style=None,
                output_format="png",
                response_format="b64_json",
                background="transparent",
                input_images=("data:image/png;base64,aW1hZ2U=",),
                azure_api_version=None,
            )
            ```

    Args:
        prompt: Text description of the requested image output.
        model: Exact configured image model or deployment identity.
        n: Number of images requested.
        size: Optional provider image dimensions.
        quality: Optional provider quality mode.
        style: Optional provider style mode.
        output_format: Optional encoded image format.
        response_format: Optional response transport format.
        background: Optional provider background mode.
        input_images: Detached optional source-image data URLs.
        azure_api_version: Optional Azure Images API version.
        options: Detached bounded adapter-private options.

    Returns:
        ImageAdapterInvocation: Immutable prepared adapter-call state.

    Notes:
        Retry, rate gating, accounting, metering, observations, and tracing remain
        owned by the shared image-generation lifecycle.
    """

    prompt: str
    model: str
    n: int
    size: str | None
    quality: str | None
    style: str | None
    output_format: ImageFormat | None
    response_format: ImageResponseFormat | None
    background: str | None
    input_images: tuple[str, ...]
    azure_api_version: str | None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach image invocation state.

        Intro:
            Rejects empty model/prompt identities and invalid image counts before
            any physical adapter can issue transport I/O.

        Examples:
            Validate a normal request:
                ```python
                assert ImageAdapterInvocation(
                    prompt="A compass",
                    model="image-model",
                    n=1,
                    size=None,
                    quality=None,
                    style=None,
                    output_format=None,
                    response_format=None,
                    background=None,
                    input_images=(),
                    azure_api_version=None,
                ).n == 1
                ```

            Reject a zero image count:
                ```python
                try:
                    ImageAdapterInvocation(
                        prompt="A compass",
                        model="image-model",
                        n=0,
                        size=None,
                        quality=None,
                        style=None,
                        output_format=None,
                        response_format=None,
                        background=None,
                        input_images=(),
                        azure_api_version=None,
                    )
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized image invocation.

        Returns:
            None: Completes after normalized values are stored.

        Notes:
            Provider-specific feature support remains adapter-owned and fails
            without fallback when the selected endpoint cannot represent a field.
        """

        prompt = str(self.prompt or "").strip()
        model = str(self.model or "").strip()
        if not prompt:
            raise ValueError("image adapter invocation requires a prompt")
        if not model:
            raise ValueError("image adapter invocation requires a model")
        if isinstance(self.n, bool) or int(self.n) < 1:
            raise ValueError("image adapter invocation requires n >= 1")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "input_images", tuple(str(item) for item in self.input_images))
        object.__setattr__(self, "options", copy.deepcopy(self.options))

    def option_dict(self) -> dict[str, Any]:
        """Return detached adapter-private image options.

        Intro:
            Supplies one mutable option mapping per physical attempt.

        Examples:
            Read an empty mapping:
                ```python
                options = invocation.option_dict()
                ```

            Mutate without changing the invocation:
                ```python
                invocation.option_dict()["seed"] = 7
                ```

        Args:
            self: Prepared image invocation.

        Returns:
            dict[str, Any]: Deep-copied adapter-private options.

        Notes:
            Retry attempts cannot share mutations through this projection.
        """

        return copy.deepcopy(self.options)


async def _invoke_openai_images(
    host: Any, invocation: ImageAdapterInvocation
) -> ImageAdapterResult:
    return await OpenAIImagesAdapter.invoke(
        host,
        invocation.prompt,
        model=invocation.model,
        n=invocation.n,
        size=invocation.size,
        quality=invocation.quality,
        style=invocation.style,
        output_format=invocation.output_format,
        response_format=invocation.response_format,
        background=invocation.background,
        **invocation.option_dict(),
    )


async def _invoke_azure_images(host: Any, invocation: ImageAdapterInvocation) -> ImageAdapterResult:
    return await AzureImagesAdapter.invoke(
        host,
        invocation.prompt,
        model=invocation.model,
        n=invocation.n,
        size=invocation.size,
        quality=invocation.quality,
        style=invocation.style,
        output_format=invocation.output_format,
        response_format=invocation.response_format,
        background=invocation.background,
        azure_api_version=invocation.azure_api_version,
        **invocation.option_dict(),
    )


async def _invoke_gemini_images(
    host: Any, invocation: ImageAdapterInvocation
) -> ImageAdapterResult:
    return await GeminiImagesAdapter.invoke(
        host,
        invocation.prompt,
        model=invocation.model,
        input_images=list(invocation.input_images) or None,
        **invocation.option_dict(),
    )


_IMAGE_HANDLERS: dict[str, ImageHandler] = {
    "openai_images": _invoke_openai_images,
    "azure_images": _invoke_azure_images,
    "gemini_image_generation": _invoke_gemini_images,
}


def registered_image_adapter_ids() -> frozenset[str]:
    """Return exact endpoint IDs with physical image handlers.

    Intro:
        Exposes immutable runtime coverage for canonical registry conformance
        without exposing mutable handler state.

    Examples:
        Check OpenAI Images coverage:
            ```python
            assert "openai_images" in registered_image_adapter_ids()
            ```

        Check a Chat endpoint:
            ```python
            assert "openai_responses" not in registered_image_adapter_ids()
            ```

    Args:
        This function accepts no arguments.

    Returns:
        frozenset[str]: Exact registered physical image adapter identities.

    Notes:
        Capability descriptors remain owned by the canonical endpoint registry.
    """

    return frozenset(_IMAGE_HANDLERS)


async def invoke_image_adapter(
    host: Any,
    *,
    adapter_id: str,
    invocation: ImageAdapterInvocation,
) -> ImageAdapterResult:
    """Invoke one exact registered image-generation adapter.

    Intro:
        Resolves physical image behavior solely by canonical endpoint adapter ID
        and fails closed when no implementation is registered.

    Examples:
        Invoke OpenAI Images:
            ```python
            result = await invoke_image_adapter(
                client,
                adapter_id="openai_images",
                invocation=invocation,
            )
            ```

        Invoke Gemini image generation:
            ```python
            result = await invoke_image_adapter(
                client,
                adapter_id="gemini_image_generation",
                invocation=invocation,
            )
            ```

    Args:
        host: Bound generic client owning shared transport primitives.
        adapter_id: Exact selected image endpoint-adapter identity.
        invocation: Frozen prepared image invocation.

    Returns:
        ImageAdapterResult: Normalized images, usage, and transport metadata.

    Notes:
        Selection contains no provider-name branching or fallback. Shared retry
        and terminal accounting remain outside this single-attempt boundary.
    """

    handler = _IMAGE_HANDLERS.get(str(adapter_id or "").strip())
    if handler is None:
        raise LLMUnsupportedFeatureError(
            host.provider,
            invocation.model,
            "image_generation",
            f"endpoint adapter {adapter_id!r} has no image-generation implementation",
        )
    return await handler(host, invocation)
