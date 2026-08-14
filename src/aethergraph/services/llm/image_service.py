"""Named image-generation client service."""

from __future__ import annotations

from typing import Any

from aethergraph.contracts.services.llm import ImageGenerationClientProtocol
from aethergraph.services.llm.types import (
    ImageFormat,
    ImageGenerationResult,
    ImageResponseFormat,
)


class ImageGenerationService:
    """Hold independently configured image-generation clients by profile name."""

    def __init__(self, clients: dict[str, ImageGenerationClientProtocol]) -> None:
        """Create an exact named image-client registry.

        Intro:
            Copies the supplied mapping so later caller mutation cannot alter the
            service's profile registry.

        Examples:
            Register a default client:
                ```python
                service = ImageGenerationService({"default": client})
                ```

            Register named profiles:
                ```python
                service = ImageGenerationService(
                    {"default": default_client, "design": design_client}
                )
                ```

        Args:
            self: Newly allocated image-generation service.
            clients: Exact clients keyed by configured profile name.

        Returns:
            None: Initializes the copied client registry.

        Notes:
            The service performs no fallback profile selection.
        """

        self._clients = dict(clients)

    def get(self, name: str = "default") -> ImageGenerationClientProtocol:
        """Return one exact configured image-generation client.

        Intro:
            Resolves a stable named client without constructing a Chat-backed
            compatibility client.

        Examples:
            Read the default client:
                ```python
                client = service.get()
                ```

            Read a named client:
                ```python
                client = service.get("design")
                ```

        Args:
            self: Container-owned image-generation service.
            name: Exact configured profile name.

        Returns:
            ImageGenerationClientProtocol: Stable configured client identity.

        Notes:
            Missing profiles raise `KeyError`; no default fallback is attempted.
        """

        return self._clients[name]

    async def generate_image(
        self,
        prompt: str,
        *,
        profile: str = "default",
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
        **kw: Any,
    ) -> ImageGenerationResult:
        """Generate images through one exact named profile.

        Intro:
            Delegates directly to the selected operation client so endpoint,
            transport, retry, rate, observation, and metering ownership remain there.

        Examples:
            Generate with the default profile:
                ```python
                result = await service.generate_image("A quiet observatory")
                ```

            Generate with a named profile:
                ```python
                result = await service.generate_image(
                    "A glass compass",
                    profile="design",
                    size="1024x1024",
                )
                ```

        Args:
            self: Container-owned image-generation service.
            prompt: Text description of the requested output.
            profile: Exact configured image profile name.
            model: Optional per-call model override.
            n: Optional per-call output count.
            size: Optional per-call image dimensions.
            quality: Optional per-call provider quality mode.
            style: Optional per-call provider style mode.
            output_format: Optional encoded image format.
            response_format: Optional response transport format.
            background: Optional provider background mode.
            input_images: Optional source-image data URLs.
            azure_api_version: Optional Azure Images API version.
            **kw: Bounded adapter-private options.

        Returns:
            ImageGenerationResult: Normalized images, provider usage, and raw data.

        Notes:
            This service adds no provider dispatch, retry, or fallback layer.
        """

        return await self.get(profile).generate_image(
            prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            style=style,
            output_format=output_format,
            response_format=response_format,
            background=background,
            input_images=input_images,
            azure_api_version=azure_api_version,
            **kw,
        )

    async def aclose(self) -> None:
        """Close transport resources for all configured image clients.

        Intro:
            Visits each distinct client once and invokes its optional asynchronous
            close boundary.

        Examples:
            Close at application shutdown:
                ```python
                await service.aclose()
                ```

            Close an empty service safely:
                ```python
                await ImageGenerationService({}).aclose()
                ```

        Args:
            self: Container-owned image-generation service.

        Returns:
            None: Completes after every close-capable client is processed.

        Notes:
            Client-specific cross-loop close behavior remains client-owned.
        """

        seen: set[int] = set()
        for client in self._clients.values():
            if id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
