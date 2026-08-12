"""Canonical provider and endpoint-adapter registry for model operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Literal

ModelOperation = Literal["chat", "embeddings", "image_generation"]


@dataclass(frozen=True)
class EndpointAdapterDescriptor:
    """Declare one selectable provider protocol implementation."""

    adapter_id: str
    protocol_family: str
    implemented_operations: tuple[ModelOperation, ...]
    implementation_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderDescriptor:
    """Declare one provider identity independently of its endpoint protocols."""

    provider_id: str
    display_name: str
    endpoint_ids: tuple[str, ...]
    default_endpoints: Mapping[ModelOperation, str]
    default_base_url: str | None
    base_url_env: str | None
    credential_envs: tuple[str, ...]
    model_discovery_adapter_id: str | None = None
    studio_visible: bool = True

    def __post_init__(self) -> None:
        """Freeze nested endpoint selection data after initialization.

        Intro:
            Frozen dataclasses do not freeze mutable mapping values by default,
            so provider operation defaults are copied into a read-only mapping.

        Examples:
            Inspect a default endpoint:
                ```python
                assert descriptor.default_endpoints["chat"]
                ```

            Observe mutation rejection:
                ```python
                try:
                    descriptor.default_endpoints["chat"] = "other"
                except TypeError:
                    pass
                ```

        Args:
            self: Newly initialized provider descriptor.

        Returns:
            None: Freezes the nested default-endpoint mapping in place.

        Notes:
            Descriptor identities and endpoint tuples are already immutable.
        """

        object.__setattr__(
            self,
            "default_endpoints",
            MappingProxyType(dict(self.default_endpoints)),
        )


_ENDPOINTS = (
    EndpointAdapterDescriptor(
        "openai_responses",
        "responses",
        ("chat",),
        ("image_input", "streaming", "native_tools", "native_tool_search", "structured_output"),
    ),
    EndpointAdapterDescriptor(
        "openai_chat_completions",
        "chat.completions",
        ("chat",),
        ("image_input", "streaming", "native_tools", "structured_output"),
    ),
    EndpointAdapterDescriptor(
        "azure_responses",
        "responses",
        ("chat",),
        ("image_input", "streaming", "native_tools", "native_tool_search", "structured_output"),
    ),
    EndpointAdapterDescriptor(
        "azure_chat_completions",
        "chat.completions",
        ("chat",),
        ("image_input", "streaming", "native_tools", "structured_output"),
    ),
    EndpointAdapterDescriptor(
        "anthropic_messages",
        "messages",
        ("chat",),
        ("image_input", "streaming", "native_tools", "native_tool_search", "structured_output"),
    ),
    EndpointAdapterDescriptor(
        "gemini_generate_content",
        "generateContent",
        ("chat",),
        ("image_input", "streaming", "native_tools", "structured_output"),
    ),
    EndpointAdapterDescriptor("openai_embeddings", "openai.embeddings", ("embeddings",)),
    EndpointAdapterDescriptor("azure_embeddings", "azure.embeddings", ("embeddings",)),
    EndpointAdapterDescriptor("gemini_embeddings", "google.embeddings", ("embeddings",)),
    EndpointAdapterDescriptor("openai_images", "openai.images", ("image_generation",)),
    EndpointAdapterDescriptor("azure_images", "azure.images", ("image_generation",)),
    EndpointAdapterDescriptor(
        "gemini_image_generation",
        "google.image_generation",
        ("image_generation",),
    ),
    EndpointAdapterDescriptor("dummy_chat", "dummy.chat", ("chat",)),
)

_PROVIDERS = (
    ProviderDescriptor(
        "openai",
        "OpenAI",
        ("openai_responses", "openai_chat_completions", "openai_embeddings", "openai_images"),
        {
            "chat": "openai_responses",
            "embeddings": "openai_embeddings",
            "image_generation": "openai_images",
        },
        "https://api.openai.com/v1",
        "OPENAI_BASE_URL",
        ("OPENAI_API_KEY",),
        "openai_models",
    ),
    ProviderDescriptor(
        "azure",
        "Azure OpenAI",
        ("azure_responses", "azure_chat_completions", "azure_embeddings", "azure_images"),
        {
            "chat": "azure_chat_completions",
            "embeddings": "azure_embeddings",
            "image_generation": "azure_images",
        },
        None,
        "AZURE_OPENAI_ENDPOINT",
        ("AZURE_OPENAI_KEY",),
        "azure_openai_deployments",
    ),
    ProviderDescriptor(
        "anthropic",
        "Anthropic",
        ("anthropic_messages",),
        {"chat": "anthropic_messages"},
        "https://api.anthropic.com",
        None,
        ("ANTHROPIC_API_KEY",),
        "anthropic_models",
    ),
    ProviderDescriptor(
        "google",
        "Google Gemini",
        ("gemini_generate_content", "gemini_embeddings", "gemini_image_generation"),
        {
            "chat": "gemini_generate_content",
            "embeddings": "gemini_embeddings",
            "image_generation": "gemini_image_generation",
        },
        "https://generativelanguage.googleapis.com",
        None,
        ("GOOGLE_API_KEY",),
        "gemini_models",
    ),
    ProviderDescriptor(
        "openrouter",
        "OpenRouter",
        ("openai_chat_completions",),
        {"chat": "openai_chat_completions"},
        "https://openrouter.ai/api/v1",
        None,
        ("OPENROUTER_API_KEY",),
        "openai_compatible_models",
    ),
    ProviderDescriptor(
        "deepseek",
        "DeepSeek",
        ("openai_chat_completions",),
        {"chat": "openai_chat_completions"},
        "https://api.deepseek.com",
        "DEEPSEEK_BASE_URL",
        ("DEEPSEEK_API_KEY",),
        "openai_compatible_models",
    ),
    ProviderDescriptor(
        "lmstudio",
        "LM Studio",
        ("openai_chat_completions", "openai_embeddings"),
        {"chat": "openai_chat_completions", "embeddings": "openai_embeddings"},
        "http://localhost:1234/v1",
        "LMSTUDIO_BASE_URL",
        (),
        "openai_compatible_models",
    ),
    ProviderDescriptor(
        "ollama",
        "Ollama",
        ("openai_chat_completions", "openai_embeddings"),
        {"chat": "openai_chat_completions", "embeddings": "openai_embeddings"},
        "http://localhost:11434/v1",
        "OLLAMA_BASE_URL",
        (),
        "openai_compatible_models",
    ),
    ProviderDescriptor(
        "openai_compatible",
        "OpenAI-compatible endpoint",
        ("openai_chat_completions", "openai_embeddings"),
        {"chat": "openai_chat_completions", "embeddings": "openai_embeddings"},
        None,
        "OPENAI_COMPATIBLE_BASE_URL",
        ("OPENAI_COMPATIBLE_API_KEY",),
        "openai_compatible_models",
    ),
    ProviderDescriptor(
        "dummy",
        "Dummy (tests)",
        ("dummy_chat",),
        {"chat": "dummy_chat"},
        None,
        None,
        (),
        None,
        False,
    ),
)

ENDPOINT_ADAPTERS = {item.adapter_id: item for item in _ENDPOINTS}
PROVIDERS = {item.provider_id: item for item in _PROVIDERS}


def get_provider_descriptor(provider_id: str) -> ProviderDescriptor:
    """Return one exact provider descriptor or fail closed.

    Intro:
        Provider lookup normalizes surrounding whitespace and case, then
        requires a registered identity without substituting a default.

    Examples:
        Resolve OpenAI:
            ```python
            descriptor = get_provider_descriptor("openai")
            assert descriptor.display_name == "OpenAI"
            ```

        Reject an unknown provider:
            ```python
            try:
                get_provider_descriptor("missing")
            except KeyError:
                pass
            ```

    Args:
        provider_id: Configured provider identity.

    Returns:
        ProviderDescriptor: Immutable registered provider descriptor.

    Notes:
        `dummy` is registered for tests but is not Studio-visible.
    """

    key = str(provider_id or "").strip().lower()
    try:
        return PROVIDERS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM provider: {provider_id!r}") from exc


def get_endpoint_adapter(adapter_id: str) -> EndpointAdapterDescriptor:
    """Return one exact endpoint-adapter descriptor or fail closed.

    Intro:
        Adapter lookup selects protocol identity independently from provider
        identity and never switches according to request features.

    Examples:
        Resolve the Responses adapter:
            ```python
            adapter = get_endpoint_adapter("openai_responses")
            assert "chat" in adapter.implemented_operations
            ```

        Reject an unknown adapter:
            ```python
            try:
                get_endpoint_adapter("missing")
            except KeyError:
                pass
            ```

    Args:
        adapter_id: Registered endpoint-adapter identity.

    Returns:
        EndpointAdapterDescriptor: Immutable adapter descriptor.

    Notes:
        This lookup performs no provider request or capability inference.
    """

    key = str(adapter_id or "").strip()
    try:
        return ENDPOINT_ADAPTERS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM endpoint adapter: {adapter_id!r}") from exc


def resolve_endpoint_adapter(
    provider_id: str,
    operation: ModelOperation,
    *,
    endpoint_id: str | None = None,
) -> EndpointAdapterDescriptor:
    """Resolve one provider endpoint for an operation before invocation.

    Intro:
        The resolver validates provider membership, endpoint membership, and
        operation support. An explicit endpoint is never replaced after an
        incompatibility.

    Examples:
        Resolve the default OpenAI Chat endpoint:
            ```python
            adapter = resolve_endpoint_adapter("openai", "chat")
            assert adapter.adapter_id == "openai_responses"
            ```

        Select Azure Chat Completions explicitly:
            ```python
            adapter = resolve_endpoint_adapter(
                "azure", "chat", endpoint_id="azure_chat_completions"
            )
            ```

    Args:
        provider_id: Registered provider identity.
        operation: Required model operation.
        endpoint_id: Optional explicit endpoint selection.

    Returns:
        EndpointAdapterDescriptor: Exactly one compatible endpoint adapter.

    Notes:
        Resolution occurs before request features are inspected.
    """

    provider = get_provider_descriptor(provider_id)
    selected = endpoint_id or provider.default_endpoints.get(operation)
    if selected is None:
        raise ValueError(f"Provider {provider.provider_id!r} does not implement {operation!r}.")
    if selected not in provider.endpoint_ids:
        raise ValueError(
            f"Endpoint {selected!r} is not registered for provider {provider.provider_id!r}."
        )
    adapter = get_endpoint_adapter(selected)
    if operation not in adapter.implemented_operations:
        raise ValueError(f"Endpoint {selected!r} does not implement {operation!r}.")
    return adapter


def resolve_endpoint_family(
    provider_id: str,
    operation: ModelOperation,
    endpoint_family: str,
) -> EndpointAdapterDescriptor:
    """Resolve one provider endpoint from its public protocol-family name.

    Intro:
        Existing public provider APIs identify families such as `responses` or
        `messages`. This boundary resolver maps that name to exactly one
        registered adapter for a provider and operation.

    Examples:
        Resolve OpenAI Responses:
            ```python
            adapter = resolve_endpoint_family("openai", "chat", "responses")
            assert adapter.adapter_id == "openai_responses"
            ```

        Reject an unavailable family:
            ```python
            try:
                resolve_endpoint_family("anthropic", "chat", "responses")
            except ValueError:
                pass
            ```

    Args:
        provider_id: Registered provider identity.
        operation: Required model operation.
        endpoint_family: Existing public protocol-family name.

    Returns:
        EndpointAdapterDescriptor: Unique matching endpoint adapter.

    Notes:
        The mapping is a compatibility boundary and performs no request-based
        adapter switching.
    """

    provider = get_provider_descriptor(provider_id)
    family = str(endpoint_family or "").strip()
    matches = tuple(
        ENDPOINT_ADAPTERS[adapter_id]
        for adapter_id in provider.endpoint_ids
        if ENDPOINT_ADAPTERS[adapter_id].protocol_family == family
        and operation in ENDPOINT_ADAPTERS[adapter_id].implemented_operations
    )
    if len(matches) != 1:
        raise ValueError(
            f"Provider {provider.provider_id!r} has no unique {operation!r} "
            f"endpoint family {family!r}."
        )
    return matches[0]


def provider_default_base_url(
    provider_id: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the registry-owned default base URL for one provider.

    Intro:
        Environment-backed provider endpoints override static registry defaults.
        Empty environment values are ignored and trailing slashes are removed.

    Examples:
        Resolve the OpenAI default:
            ```python
            assert provider_default_base_url("openai") == "https://api.openai.com/v1"
            ```

        Override a local LM Studio endpoint:
            ```python
            url = provider_default_base_url(
                "lmstudio", environ={"LMSTUDIO_BASE_URL": "http://localhost:9000/v1/"}
            )
            assert url == "http://localhost:9000/v1"
            ```

    Args:
        provider_id: Registered provider identity.
        environ: Optional environment mapping used instead of `os.environ`.

    Returns:
        str | None: Normalized configured default URL, when one exists.

    Notes:
        Custom compatible endpoints intentionally have no invented default URL.
    """

    descriptor = get_provider_descriptor(provider_id)
    values = os.environ if environ is None else environ
    configured = values.get(descriptor.base_url_env, "") if descriptor.base_url_env else ""
    value = configured.strip() or descriptor.default_base_url
    return value.rstrip("/") if value else None


__all__ = [
    "ENDPOINT_ADAPTERS",
    "PROVIDERS",
    "EndpointAdapterDescriptor",
    "ModelOperation",
    "ProviderDescriptor",
    "get_endpoint_adapter",
    "get_provider_descriptor",
    "provider_default_base_url",
    "resolve_endpoint_adapter",
    "resolve_endpoint_family",
]
