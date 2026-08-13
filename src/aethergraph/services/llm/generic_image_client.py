"""Provider-neutral image-generation client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aethergraph.contracts.services.llm import ImageGenerationClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.llm.adapters import ImageAdapterInvocation
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.http_lifecycle import (
    _close_http_clients,
    _ensure_loop_http_client,
)
from aethergraph.services.llm.image_runtime import _execute_image_generation
from aethergraph.services.llm.provider_transport import (
    ProviderRateGate,
    ProviderRetryExecutor,
    ProviderRetrySettings,
)
from aethergraph.services.llm.registry import provider_default_base_url, resolve_endpoint_adapter
from aethergraph.services.llm.types import (
    ImageFormat,
    ImageGenerationResult,
    ImageGenerationUsage,
    ImageResponseFormat,
)
from aethergraph.services.llm.usage_metering import _record_image_generation_metering


class GenericImageGenerationClient(ImageGenerationClientProtocol):
    """Own one independent image-generation profile and transport lifecycle."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        endpoint_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float = 60.0,
        retry_settings: ProviderRetrySettings | None = None,
        rate_limit_group: str | None = None,
        rate_gate: ProviderRateGate | None = None,
        metering: MeteringService | None = None,
        default_count: int = 1,
        default_size: str | None = None,
        default_quality: str | None = None,
        default_output_format: ImageFormat | None = None,
        default_response_format: ImageResponseFormat | None = None,
        default_background: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        """Create an independently configured image-generation client.

        Intro:
            Validates and pins one exact image endpoint before transport, then
            stores only image-operation defaults and shared infrastructure
            controls.

        Examples:
            Create an OpenAI Images client:
                ```python
                client = GenericImageGenerationClient(
                    provider="openai",
                    model="gpt-image-1",
                    endpoint_id="openai_images",
                )
                ```

            Create an Azure image deployment:
                ```python
                client = GenericImageGenerationClient(
                    provider="azure",
                    model="image-model",
                    endpoint_id="azure_images",
                    base_url="https://example.openai.azure.com",
                    azure_deployment="image-prod",
                )
                ```

        Args:
            self: Newly allocated image-generation client.
            provider: Registered provider identity.
            model: Default image model or deployment identity.
            endpoint_id: Exact registered image endpoint adapter.
            base_url: Optional provider API base URL override.
            api_key: Optional already-resolved provider credential.
            azure_deployment: Optional Azure deployment identity.
            timeout: HTTP request timeout in seconds.
            retry_settings: Optional bounded provider retry policy.
            rate_limit_group: Optional shared provider quota bucket.
            rate_gate: Optional container-shared rate gate.
            metering: Optional shared model metering service.
            default_count: Default number of generated images.
            default_size: Optional default image dimensions.
            default_quality: Optional default provider quality mode.
            default_output_format: Optional default encoded image format.
            default_response_format: Optional default response transport format.
            default_background: Optional default provider background mode.
            profile_name: Optional configured profile identity.

        Returns:
            None: Initializes a lazy, exact-bound client.

        Notes:
            No Chat client or AG Engine service is imported or required. The
            physical HTTP client is created lazily on the active event loop.
        """

        adapter = resolve_endpoint_adapter(
            provider,
            "image_generation",
            endpoint_id=endpoint_id,
        )
        self.provider = str(provider).strip().lower()
        self.model = str(model).strip()
        self.endpoint_id = adapter.adapter_id
        self.base_url = base_url or provider_default_base_url(self.provider) or ""
        self.api_key = resolve_provider_credential(
            provider_id=self.provider,
            direct=api_key,
            secret_ref=None,
            secrets=None,
        ).value
        self.azure_deployment = azure_deployment
        self.timeout = float(timeout)
        self.rate_limit_group = rate_limit_group
        self.metering = metering
        self.default_count = int(default_count)
        self.default_size = default_size
        self.default_quality = default_quality
        self.default_output_format = default_output_format
        self.default_response_format = default_response_format
        self.default_background = default_background
        self.profile_name = profile_name
        self._provider_retry = ProviderRetryExecutor(
            retry_settings,
            rate_gate=rate_gate,
            base_url=self.base_url,
            credential=self.api_key,
        )
        self._client: httpx.AsyncClient | None = None
        self._retired_http_clients: list[httpx.AsyncClient] = []
        self._bound_loop = None
        self._logger = logging.getLogger(__name__)

    async def _ensure_client(self) -> None:
        self._client, self._bound_loop, retired = _ensure_loop_http_client(
            self._client,
            self._bound_loop,
            timeout=self.timeout,
        )
        if retired is not None:
            self._retired_http_clients.append(retired)

    def _current_dimensions(self) -> dict[str, Any]:
        context = current_meter_context.get()
        return {
            "user_id": context.get("user_id"),
            "org_id": context.get("org_id"),
            "run_id": context.get("run_id"),
            "graph_id": context.get("graph_id"),
            "session_id": context.get("session_id"),
            "app_id": context.get("app_id"),
            "agent_id": context.get("agent_id"),
            "node_id": context.get("node_id"),
            "trace_id": context.get("trace_id"),
            "span_id": context.get("span_id"),
            "profile_name": self.profile_name,
        }

    async def _account_usage(
        self,
        model: str,
        usage: ImageGenerationUsage,
        image_count: int,
        size: str | None,
        quality: str | None,
        latency_ms: int,
    ) -> None:
        await _record_image_generation_metering(
            self.metering or current_metering(),
            provider=self.provider,
            model=model,
            usage=usage,
            image_count=image_count,
            size=size,
            quality=quality,
            latency_ms=latency_ms,
            dimensions=self._current_dimensions(),
            logger=self._logger,
        )

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
        **kw: Any,
    ) -> ImageGenerationResult:
        """Generate images through this profile's exact endpoint.

        Intro:
            Applies image-profile defaults, freezes one adapter invocation, and
            executes the centralized retry, rate, tracing, and metering lifecycle.

        Examples:
            Generate with profile defaults:
                ```python
                result = await client.generate_image("A quiet observatory")
                ```

            Generate an image-conditioned transparent PNG:
                ```python
                result = await client.generate_image(
                    "Make the sky violet",
                    size="1024x1024",
                    output_format="png",
                    response_format="b64_json",
                    background="transparent",
                    input_images=["data:image/png;base64,aW1hZ2U="],
                )
                ```

        Args:
            self: Exact-bound image-generation client.
            prompt: Text description of the requested output.
            model: Optional per-call model override.
            n: Optional output count overriding the profile default.
            size: Optional image dimensions overriding the profile default.
            quality: Optional quality mode overriding the profile default.
            style: Optional provider style mode.
            output_format: Optional encoded format overriding the profile default.
            response_format: Optional transport format overriding the profile default.
            background: Optional background mode overriding the profile default.
            input_images: Optional source-image data URLs.
            azure_api_version: Optional Azure Images API version.
            **kw: Bounded adapter-private options.

        Returns:
            ImageGenerationResult: Normalized images, provider usage, and raw data.

        Notes:
            The endpoint never switches according to request fields or provider
            failure. Unsupported combinations fail in the selected adapter.
        """

        invocation = ImageAdapterInvocation(
            prompt=prompt,
            model=model or self.model,
            n=self.default_count if n is None else n,
            size=self.default_size if size is None else size,
            quality=self.default_quality if quality is None else quality,
            style=style,
            output_format=self.default_output_format if output_format is None else output_format,
            response_format=(
                self.default_response_format if response_format is None else response_format
            ),
            background=self.default_background if background is None else background,
            input_images=tuple(input_images or ()),
            azure_api_version=azure_api_version,
            options=kw,
        )
        return await _execute_image_generation(
            self,
            adapter_id=self.endpoint_id,
            invocation=invocation,
            account_usage=self._account_usage,
            dimensions=self._current_dimensions(),
        )

    async def aclose(self) -> None:
        """Close active and safely retired image HTTP clients.

        Intro:
            Owns cleanup for transports created across event-loop changes during
            the image client's lifetime.

        Examples:
            Close one client:
                ```python
                await client.aclose()
                ```

            Close all clients through the service:
                ```python
                await service.aclose()
                ```

        Args:
            self: Image-generation client owning transport resources.

        Returns:
            None: Closes every distinct reachable HTTP transport.

        Notes:
            Cross-loop close failures are logged and do not prevent remaining
            transports from being processed.
        """

        clients = [self._client, *self._retired_http_clients]
        self._retired_http_clients = []
        await _close_http_clients(
            clients,
            logger=self._logger,
            warning_key="image_http_client_close_failed",
        )
