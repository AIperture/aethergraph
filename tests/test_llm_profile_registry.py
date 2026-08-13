from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

from pydantic import SecretStr, ValidationError
import pytest

from aethergraph.config.llm import (
    EmbeddingProfile,
    ImageGenerationProfileSettings,
    LLMProfile,
    LLMSettings,
)
from aethergraph.services.llm import (
    ENDPOINT_ADAPTERS,
    PROVIDERS,
    ChatProfile,
    EmbeddingDefaults,
    get_provider_descriptor,
    provider_default_base_url,
    resolve_endpoint_adapter,
)
from aethergraph.services.llm.adapters.embedding import registered_embedding_adapter_ids
from aethergraph.services.llm.compat import (
    chat_profile_from_legacy,
    embedding_profile_from_legacy,
    image_generation_profile_from_settings,
    resolve_legacy_chat_adapter,
)
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.embed_factory import embed_client_from_profile
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.factory import build_llm_clients, client_from_profile
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient
from aethergraph.services.llm.image_factory import image_client_from_profile
from aethergraph.services.llm.providers import Provider


class _Secrets:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)


def test_provider_literal_and_registry_have_one_identity_set() -> None:
    assert set(get_args(Provider)) == set(PROVIDERS)


def test_registry_endpoints_exist_and_support_declared_defaults() -> None:
    for provider in PROVIDERS.values():
        assert set(provider.endpoint_ids) <= set(ENDPOINT_ADAPTERS)
        for operation, endpoint_id in provider.default_endpoints.items():
            assert endpoint_id in provider.endpoint_ids
            assert operation in ENDPOINT_ADAPTERS[endpoint_id].implemented_operations


def test_registry_rejects_unknown_provider_and_cross_provider_endpoint() -> None:
    with pytest.raises(KeyError, match="Unknown LLM provider"):
        get_provider_descriptor("missing")

    with pytest.raises(ValueError, match="not registered"):
        resolve_endpoint_adapter("anthropic", "chat", endpoint_id="openai_responses")


def test_registry_resolves_static_and_environment_base_urls() -> None:
    assert provider_default_base_url("openai", environ={}) == "https://api.openai.com/v1"
    assert (
        provider_default_base_url(
            "lmstudio",
            environ={"LMSTUDIO_BASE_URL": "http://localhost:9000/v1/"},
        )
        == "http://localhost:9000/v1"
    )
    assert provider_default_base_url("openai_compatible", environ={}) is None
    assert (
        provider_default_base_url(
            "openai",
            environ={"OPENAI_BASE_URL": "https://gateway.example/v1/"},
        )
        == "https://gateway.example/v1"
    )
    assert (
        provider_default_base_url(
            "deepseek",
            environ={"DEEPSEEK_BASE_URL": "https://deepseek.example/v1/"},
        )
        == "https://deepseek.example/v1"
    )
    assert (
        provider_default_base_url(
            "openai_compatible",
            environ={"OPENAI_COMPATIBLE_BASE_URL": "http://localhost:9000/v1/"},
        )
        == "http://localhost:9000/v1"
    )


def test_credential_resolution_uses_inline_store_environment_precedence() -> None:
    secrets = _Secrets({"profile-key": "stored"})

    inline = resolve_provider_credential(
        provider_id="openai",
        direct=SecretStr("inline"),
        secret_ref="profile-key",
        secrets=secrets,
        environ={"OPENAI_API_KEY": "environment"},
    )
    stored = resolve_provider_credential(
        provider_id="openai",
        direct=None,
        secret_ref="profile-key",
        secrets=secrets,
        environ={"OPENAI_API_KEY": "environment"},
    )
    environment = resolve_provider_credential(
        provider_id="openai",
        direct=None,
        secret_ref="missing",
        secrets=secrets,
        environ={"OPENAI_API_KEY": "environment"},
    )

    assert (inline.value, inline.source_ref) == ("inline", None)
    assert (stored.value, stored.source_ref) == ("stored", "profile-key")
    assert (environment.value, environment.source_ref) == (
        "environment",
        "OPENAI_API_KEY",
    )


@pytest.mark.parametrize(
    ("provider", "expected_key"),
    [
        ("openai", "openai-key"),
        ("azure", "azure-key"),
        ("google", "google-key"),
        ("openrouter", "openrouter-key"),
        ("openai_compatible", "compatible-key"),
    ],
)
def test_embedding_client_uses_exact_registry_connection_defaults(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected_key: str,
) -> None:
    values = {
        "OPENAI_API_KEY": "openai-key",
        "AZURE_OPENAI_KEY": "azure-key",
        "GOOGLE_API_KEY": "google-key",
        "OPENROUTER_API_KEY": "openrouter-key",
        "OPENAI_COMPATIBLE_API_KEY": "compatible-key",
        "AZURE_OPENAI_ENDPOINT": "https://azure.example",
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:9000/v1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    client = GenericEmbeddingClient(provider=provider, model="embed-test")

    assert client.api_key == expected_key


def test_legacy_chat_profile_projection_separates_operation_concerns() -> None:
    legacy = LLMProfile(
        provider="openai",
        model="gpt-5.6",
        embed_model="text-embedding-3-large",
        api_key_ref="OPENAI_API_KEY",
        reasoning_effort="high",
        compatibility_policy="strict",
        prompt_cache_policy="required",
        context_window_tokens=128_000,
        vision_enabled=True,
        vision_max_images=3,
    )

    canonical = chat_profile_from_legacy(legacy)

    assert canonical.operation == "chat"
    assert canonical.connection.endpoint_id == "openai_responses"
    assert canonical.model.model_id == "gpt-5.6"
    assert canonical.credentials.secret_ref == "OPENAI_API_KEY"
    assert canonical.defaults.reasoning_effort == "high"
    assert canonical.defaults.compatibility_policy == "strict"
    assert canonical.defaults.prompt_cache_policy == "required"
    assert canonical.defaults.context_window_tokens == 128_000
    assert canonical.input_policy.max_images == 3
    assert canonical.input_policy.allow_remote_urls is True
    assert canonical.capability_overrides.image_input == "supported"
    assert "embed_model" not in canonical.model_dump()
    assert legacy.embed_model == "text-embedding-3-large"


def test_chat_factory_consumes_canonical_profile_without_losing_policy() -> None:
    canonical = chat_profile_from_legacy(
        LLMProfile(
            provider="openai",
            model="gpt-5.6",
            compatibility_policy="strict",
            structured_output_policy="native_required",
            prompt_cache_policy="disabled",
            context_window_tokens=128_000,
            vision_enabled=True,
            vision_max_images=2,
        )
    )

    client = client_from_profile(canonical, _Secrets())

    assert client.provider == "openai"
    assert client.endpoint_id == "openai_responses"
    assert client.model == "gpt-5.6"
    assert client.compatibility_policy == "strict"
    assert client.structured_output_policy == "native_required"
    assert client.prompt_cache_policy == "disabled"
    assert client.context_window_tokens == 128_000
    assert client.image_preparation_policy is not None
    assert client.image_preparation_policy.image_input_enabled is True
    assert client.image_preparation_policy.allow_remote_urls is True
    assert client.image_preparation_policy.max_images == 2


def test_legacy_profile_preserves_explicit_endpoint_selection() -> None:
    legacy = LLMProfile(
        provider="azure",
        model="deployment-a",
        endpoint_id="azure_responses",
        azure_deployment="deployment-a",
    )

    canonical = chat_profile_from_legacy(legacy)
    client = client_from_profile(canonical, _Secrets())

    assert canonical.connection.endpoint_id == "azure_responses"
    assert client.endpoint_id == "azure_responses"


def test_explicit_endpoint_rejects_cross_provider_binding() -> None:
    with pytest.raises(ValueError, match="not registered"):
        GenericLLMClient(
            provider="anthropic",
            model="claude-test",
            endpoint_id="openai_responses",
        )


def test_endpointless_legacy_settings_keep_temporary_compatibility_dispatch() -> None:
    clients = build_llm_clients(
        LLMSettings(
            default=LLMProfile(
                provider="azure",
                model="deployment-a",
                azure_deployment="deployment-a",
                base_url="https://example.openai.azure.com",
            )
        ),
        _Secrets(),
    )

    assert clients["default"].endpoint_id is None


def test_endpointless_azure_compatibility_selection_is_explicit_and_bounded() -> None:
    direct = resolve_legacy_chat_adapter("azure", None, has_tool_request=False)
    tools = resolve_legacy_chat_adapter("azure", None, has_tool_request=True)
    pinned = resolve_legacy_chat_adapter(
        "azure",
        "azure_chat_completions",
        has_tool_request=True,
    )

    assert direct.adapter_id == "azure_chat_completions"
    assert tools.adapter_id == "azure_responses"
    assert pinned.adapter_id == "azure_chat_completions"


def test_legacy_embedding_profile_projection_is_independent_from_chat() -> None:
    canonical = embedding_profile_from_legacy(
        EmbeddingProfile(provider="google", model="text-embedding-004")
    )

    assert canonical.operation == "embeddings"
    assert canonical.connection.endpoint_id == "gemini_embeddings"
    assert canonical.model.model_id == "text-embedding-004"
    assert embed_client_from_profile(canonical, _Secrets()).endpoint_id == "gemini_embeddings"


def test_embedding_factory_preserves_canonical_dimension_default() -> None:
    canonical = embedding_profile_from_legacy(
        EmbeddingProfile(provider="google", model="gemini-embedding-001")
    ).model_copy(update={"defaults": EmbeddingDefaults(dimensions=256)})

    client = embed_client_from_profile(canonical, _Secrets())

    assert client.default_dimensions == 256


def test_named_embedding_profile_inherits_dimension_default() -> None:
    template = GenericEmbeddingClient(
        provider="dummy",
        model="default",
        default_dimensions=64,
    )
    service = EmbeddingService({"default": template})

    named = service.configure_profile(name="search", provider="dummy", model="search")

    assert isinstance(named, GenericEmbeddingClient)
    assert named.default_dimensions == 64


def test_image_profile_projection_and_client_are_independent_from_chat() -> None:
    canonical = image_generation_profile_from_settings(
        ImageGenerationProfileSettings(
            provider="google",
            model="gemini-image-test",
            count=2,
            output_format="png",
        )
    )
    client = image_client_from_profile(canonical, _Secrets(), profile_name="design")

    assert canonical.operation == "image_generation"
    assert canonical.connection.endpoint_id == "gemini_image_generation"
    assert canonical.defaults.count == 2
    assert client.endpoint_id == "gemini_image_generation"
    assert client.profile_name == "design"


def test_embedding_runtime_registry_matches_advertised_operations() -> None:
    advertised = {
        adapter_id
        for adapter_id, descriptor in ENDPOINT_ADAPTERS.items()
        if "embeddings" in descriptor.implemented_operations
    }

    assert registered_embedding_adapter_ids() == advertised


def test_existing_openrouter_and_dummy_embedding_support_is_canonical() -> None:
    assert resolve_endpoint_adapter("openrouter", "embeddings").adapter_id == "openai_embeddings"
    assert resolve_endpoint_adapter("dummy", "embeddings").adapter_id == "dummy_embeddings"


@pytest.mark.asyncio
async def test_dummy_embedding_uses_the_exact_registered_adapter() -> None:
    client = GenericEmbeddingClient(provider="dummy", model="dummy-embedding")

    assert await client.embed(["hello", ""]) == [[5.0], [0.0]]


def test_canonical_profiles_are_closed_and_immutable() -> None:
    canonical = chat_profile_from_legacy(LLMProfile())

    with pytest.raises(ValidationError, match="Extra inputs"):
        ChatProfile.model_validate({**canonical.model_dump(), "legacy": True})

    with pytest.raises(ValidationError, match="frozen"):
        canonical.connection.provider_id = "anthropic"  # type: ignore[misc]


def test_model_factories_do_not_reintroduce_duplicate_resolvers() -> None:
    llm_root = Path(__file__).parents[1] / "src" / "aethergraph" / "services" / "llm"
    definitions: list[tuple[str, str]] = []
    for path in llm_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.extend(
            (node.name, path.relative_to(llm_root).as_posix())
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name
            in {
                "_resolve_key",
                "_provider_default_base_url",
                "resolve_provider_credential",
                "provider_default_base_url",
            }
        )

    assert sorted(definitions) == [
        ("provider_default_base_url", "registry.py"),
        ("resolve_provider_credential", "credentials.py"),
    ]
