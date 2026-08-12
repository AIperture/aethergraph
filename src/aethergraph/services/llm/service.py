import asyncio
import logging

import httpx

from aethergraph.config.llm import LLMProfile

from ..secrets.base import Secrets
from .generic_client import GenericLLMClient
from .profiles import PromptCachePolicy
from .providers import Provider
from .structured_output import StructuredOutputPolicy

logger = logging.getLogger("aethergraph.services.llm")


class LLMService:
    """Holds multiple LLM clients (default + named profiles)."""

    def __init__(
        self,
        clients: dict[str, GenericLLMClient],
        secrets: Secrets | None = None,
        profiles: dict[str, LLMProfile] | None = None,
    ):
        self._clients = clients
        self._secrets = secrets
        self._profiles = dict(profiles or {})

    def get(self, name: str = "default") -> GenericLLMClient:
        return self._clients[name]

    def has(self, name: str) -> bool:
        return name in self._clients

    def profile(self, name: str = "default") -> LLMProfile | None:
        return self._profiles.get(name)

    async def aclose(self):
        for c in self._clients.values():
            await c.aclose()

    # --- Runtime profile helpers ---------------------------------
    def configure_profile(
        self,
        profile: str = "default",
        *,
        provider: Provider | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
        compatibility_policy: str | None = None,
        structured_output_policy: StructuredOutputPolicy | None = None,
        prompt_cache_policy: PromptCachePolicy | None = None,
        context_window_tokens: int | None = None,
        vision_enabled: bool | None = None,
        vision_max_images: int | None = None,
        vision_max_image_bytes: int | None = None,
        vision_resize_enabled: bool | None = None,
        vision_resize_max_dimension: int | None = None,
        vision_resize_max_pixels: int | None = None,
        vision_resize_jpeg_quality: int | None = None,
        vision_resize_min_jpeg_quality: int | None = None,
        vision_accepted_mime_prefixes: list[str] | tuple[str, ...] | None = None,
        vision_accepted_mime_types: list[str] | tuple[str, ...] | None = None,
    ) -> GenericLLMClient:
        """Create or update one in-memory LLM profile and client.

        The service applies supplied values to the named client and its
        canonical profile metadata. It does not persist configuration.

        Examples:
            Update the default model:
            ```python
            client = service.configure_profile(
                profile="default",
                provider="openai",
                model="gpt-5-mini",
            )
            ```

            Require native structured output for a named profile:
            ```python
            client = service.configure_profile(
                profile="extractor",
                structured_output_policy="native_required",
            )
            ```

        Args:
            profile: Profile name to create or update.
            provider: Optional provider override.
            model: Optional model override.
            base_url: Optional provider base URL override.
            api_key: Optional in-memory API key override.
            azure_deployment: Optional Azure deployment override.
            timeout: Optional HTTP timeout in seconds.
            reasoning_effort: Optional provider-neutral reasoning effort.
            thinking_mode: Optional provider-neutral thinking mode.
            compatibility_policy: Optional unsupported-feature policy.
            structured_output_policy: Optional structured-output capability
                policy.
            prompt_cache_policy: Optional stable-prefix cache requirement policy.
            context_window_tokens: Optional model context-window capacity.
            vision_enabled: Optional image-input capability flag.
            vision_max_images: Optional maximum images per vision call.
            vision_max_image_bytes: Optional maximum bytes per image.
            vision_resize_enabled: Optional image resize flag.
            vision_resize_max_dimension: Optional resized width/height ceiling.
            vision_resize_max_pixels: Optional resized pixel-count ceiling.
            vision_resize_jpeg_quality: Optional initial JPEG quality.
            vision_resize_min_jpeg_quality: Optional minimum JPEG quality.
            vision_accepted_mime_prefixes: Optional accepted MIME prefixes.
            vision_accepted_mime_types: Optional accepted exact MIME types.

        Returns:
            GenericLLMClient: Updated or newly created client.

        Notes:
            Persistence remains the host application's responsibility. Studio
            persists profiles before invoking this hot-reload boundary.
        """
        if profile not in self._clients:
            template = self._clients.get("default")
            client = GenericLLMClient(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                azure_deployment=azure_deployment,
                timeout=timeout or 60.0,
                reasoning_effort=reasoning_effort or getattr(template, "reasoning_effort", None),
                thinking_mode=thinking_mode or getattr(template, "thinking_mode", None),
                compatibility_policy=compatibility_policy
                or getattr(template, "compatibility_policy", "compat"),
                structured_output_policy=structured_output_policy
                or getattr(
                    template,
                    "structured_output_policy",
                    "best_available",
                ),
                prompt_cache_policy=prompt_cache_policy
                or getattr(template, "prompt_cache_policy", "auto"),
                context_window_tokens=(
                    context_window_tokens
                    if context_window_tokens is not None
                    else getattr(template, "context_window_tokens", None)
                ),
                observation_sink=getattr(template, "observation_sink", None),
                observation_capture_mode=getattr(template, "observation_capture_mode", "manifest"),
            )
            self._clients[profile] = client
            self._profiles[profile] = self._updated_profile(
                profile,
                provider=provider,
                model=model,
                base_url=base_url,
                azure_deployment=azure_deployment,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                thinking_mode=thinking_mode,
                compatibility_policy=compatibility_policy,
                structured_output_policy=structured_output_policy,
                prompt_cache_policy=prompt_cache_policy,
                context_window_tokens=context_window_tokens,
                vision_enabled=vision_enabled,
                vision_max_images=vision_max_images,
                vision_max_image_bytes=vision_max_image_bytes,
                vision_resize_enabled=vision_resize_enabled,
                vision_resize_max_dimension=vision_resize_max_dimension,
                vision_resize_max_pixels=vision_resize_max_pixels,
                vision_resize_jpeg_quality=vision_resize_jpeg_quality,
                vision_resize_min_jpeg_quality=vision_resize_min_jpeg_quality,
                vision_accepted_mime_prefixes=vision_accepted_mime_prefixes,
                vision_accepted_mime_types=vision_accepted_mime_types,
            )
            return client

        c = self._clients[profile]
        if provider is not None:
            c.provider = provider  # type: ignore[assignment]
        if model is not None:
            c.model = model
        if base_url is not None:
            c.base_url = base_url
        if api_key is not None:
            c.api_key = api_key
        if azure_deployment is not None:
            c.azure_deployment = azure_deployment
        if timeout is not None:
            # Recreate client with new timeout
            old_client = c._client
            c._client = httpx.AsyncClient(timeout=timeout)
            try:
                # best-effort async close
                asyncio.create_task(old_client.aclose())
            except RuntimeError:
                logger.warning("Failed to close old httpx client")
        if compatibility_policy is not None:
            c.compatibility_policy = compatibility_policy
        if structured_output_policy is not None:
            c.structured_output_policy = structured_output_policy
        if prompt_cache_policy is not None:
            c.prompt_cache_policy = prompt_cache_policy
        if context_window_tokens is not None:
            c.context_window_tokens = int(context_window_tokens)
        if reasoning_effort is not None:
            c.reasoning_effort = reasoning_effort
        if thinking_mode is not None:
            c.thinking_mode = thinking_mode
        self._profiles[profile] = self._updated_profile(
            profile,
            provider=provider,
            model=model,
            base_url=base_url,
            azure_deployment=azure_deployment,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
            compatibility_policy=compatibility_policy,
            structured_output_policy=structured_output_policy,
            prompt_cache_policy=prompt_cache_policy,
            context_window_tokens=context_window_tokens,
            vision_enabled=vision_enabled,
            vision_max_images=vision_max_images,
            vision_max_image_bytes=vision_max_image_bytes,
            vision_resize_enabled=vision_resize_enabled,
            vision_resize_max_dimension=vision_resize_max_dimension,
            vision_resize_max_pixels=vision_resize_max_pixels,
            vision_resize_jpeg_quality=vision_resize_jpeg_quality,
            vision_resize_min_jpeg_quality=vision_resize_min_jpeg_quality,
            vision_accepted_mime_prefixes=vision_accepted_mime_prefixes,
            vision_accepted_mime_types=vision_accepted_mime_types,
        )
        return c

    def _updated_profile(
        self,
        name: str,
        *,
        provider: Provider | None,
        model: str | None,
        base_url: str | None,
        azure_deployment: str | None,
        timeout: float | None,
        reasoning_effort: str | None,
        thinking_mode: str | None,
        compatibility_policy: str | None,
        structured_output_policy: StructuredOutputPolicy | None,
        prompt_cache_policy: PromptCachePolicy | None,
        context_window_tokens: int | None,
        vision_enabled: bool | None,
        vision_max_images: int | None,
        vision_max_image_bytes: int | None,
        vision_resize_enabled: bool | None,
        vision_resize_max_dimension: int | None,
        vision_resize_max_pixels: int | None,
        vision_resize_jpeg_quality: int | None,
        vision_resize_min_jpeg_quality: int | None,
        vision_accepted_mime_prefixes: list[str] | tuple[str, ...] | None,
        vision_accepted_mime_types: list[str] | tuple[str, ...] | None,
    ) -> LLMProfile:
        current = self._profiles.get(name) or self._profiles.get("default") or LLMProfile()
        updated = current.model_copy(deep=True)
        if provider is not None:
            updated.provider = provider
        if model is not None:
            updated.model = model
        if base_url is not None:
            updated.base_url = base_url
        if azure_deployment is not None:
            updated.azure_deployment = azure_deployment
        if timeout is not None:
            updated.timeout = timeout
        if reasoning_effort is not None:
            updated.reasoning_effort = reasoning_effort  # type: ignore[assignment]
        if thinking_mode is not None:
            updated.thinking_mode = thinking_mode  # type: ignore[assignment]
        if compatibility_policy is not None:
            updated.compatibility_policy = compatibility_policy  # type: ignore[assignment]
        if structured_output_policy is not None:
            updated.structured_output_policy = structured_output_policy
        if prompt_cache_policy is not None:
            updated.prompt_cache_policy = prompt_cache_policy
        if context_window_tokens is not None:
            updated.context_window_tokens = int(context_window_tokens)
        if vision_enabled is not None:
            updated.vision_enabled = vision_enabled
        if vision_max_images is not None:
            updated.vision_max_images = vision_max_images
        if vision_max_image_bytes is not None:
            updated.vision_max_image_bytes = vision_max_image_bytes
        if vision_resize_enabled is not None:
            updated.vision_resize_enabled = vision_resize_enabled
        if vision_resize_max_dimension is not None:
            updated.vision_resize_max_dimension = vision_resize_max_dimension
        if vision_resize_max_pixels is not None:
            updated.vision_resize_max_pixels = vision_resize_max_pixels
        if vision_resize_jpeg_quality is not None:
            updated.vision_resize_jpeg_quality = vision_resize_jpeg_quality
        if vision_resize_min_jpeg_quality is not None:
            updated.vision_resize_min_jpeg_quality = vision_resize_min_jpeg_quality
        if vision_accepted_mime_prefixes is not None:
            updated.vision_accepted_mime_prefixes = [
                str(item) for item in vision_accepted_mime_prefixes
            ]
        if vision_accepted_mime_types is not None:
            updated.vision_accepted_mime_types = [str(item) for item in vision_accepted_mime_types]
        return updated

    # --- Quick start helpers ---
    def set_key(
        self, provider: str, model: str, api_key: str, profile: str = "default"
    ) -> GenericLLMClient:
        """
        Quickly set/override an API key for a profile at runtime (in-memory).
        Creates the profile if it doesn't exist yet.
        """
        return self.configure_profile(
            profile=profile,
            provider=provider,  # type: ignore[arg-type]
            model=model,
            api_key=api_key,
        )

    def persist_key(self, secret_name: str, api_key: str):
        """
        Optional: store the key via the installed Secrets provider for later runs.
        Implement only after Secrets supports write (e.g., dev file store). Env-based usually won't.
        """
        raise NotImplementedError("persist_key not implemented in this Secrets provider")
        if not self._secrets or not hasattr(self._secrets, "set"):
            raise RuntimeError("Secrets provider is not writable")
        self._secrets.set(secret_name, api_key)  # type: ignore[attr-defined]
