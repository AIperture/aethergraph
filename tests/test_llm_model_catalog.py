from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from aethergraph.config.llm import LLMProfile
from aethergraph.services.llm import (
    EmbeddingCapabilityOverrides,
    EmbeddingProfileSpec,
    ImageGenerationCapabilityOverrides,
    ImageGenerationProfile,
    ModelRequest,
    ModelSelection,
    ProviderConnection,
    ToolDefinition,
    ToolDiscoveryRequest,
    message_from_text,
    resolve_embedding_profile,
    resolve_image_generation_profile,
    resolve_model_request,
)
from aethergraph.services.llm.capabilities import resolve_chat_profile
from aethergraph.services.llm.catalog import (
    ModelCatalog,
    ModelCatalogEntry,
    catalog_digest,
    load_model_catalog,
    resolve_model_catalog_capability_entry,
    resolve_model_catalog_entry,
    validate_catalog,
)
from aethergraph.services.llm.catalog.maintenance import catalog_report, main
from aethergraph.services.llm.compat import chat_profile_from_legacy
from aethergraph.services.llm.profiles import ChatCapabilityOverrides
from aethergraph.services.llm.tool_discovery import (
    resolve_tool_discovery_capabilities,
)


def test_production_catalog_is_valid_and_digest_is_deterministic() -> None:
    catalog = load_model_catalog()

    assert catalog.entries
    assert catalog_digest(catalog) == catalog_digest(catalog)
    assert len(catalog_digest(catalog)) == 64
    assert all(entry.sources for entry in catalog.entries if entry.native_tool_search)
    assert validate_catalog(catalog) == ()


def test_catalog_maintenance_command_uses_production_loader(capsys) -> None:
    assert main(["validate"]) == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith("catalog ok ")

    report = catalog_report(load_model_catalog())
    assert report["digest"] == catalog_digest()
    assert report["entry_count"] == len(load_model_catalog().entries)
    assert report["operations"] == ["chat", "embeddings", "image_generation"]


def test_catalog_resolves_operation_capabilities_without_manufacturing_unknowns() -> None:
    embedding = resolve_model_catalog_capability_entry(
        "openai",
        "text-embedding-3-large",
        "embeddings",
        "openai_embeddings",
        capability="embeddings",
    )
    image = resolve_model_catalog_capability_entry(
        "google",
        "gemini-3.1-flash-image",
        "image_generation",
        "gemini_image_generation",
        capability="image_generation",
    )

    assert embedding is not None and embedding.embeddings is not None
    assert embedding.embeddings.dimensions == "supported"
    assert image is not None and image.image_generation is not None
    assert image.image_generation.multiple_outputs == "unknown"
    assert (
        resolve_model_catalog_capability_entry(
            "openai",
            "future-embedding-model",
            "embeddings",
            "openai_embeddings",
            capability="embeddings",
        )
        is None
    )


def test_catalog_contains_only_provider_native_tool_search_modes() -> None:
    catalog = load_model_catalog()

    assert {mode.mode for entry in catalog.entries for mode in entry.native_tool_search} <= {
        "native_hosted",
        "native_client",
    }
    assert "engine_projected" not in json.dumps(catalog.model_dump(mode="json"))


def test_catalog_resolves_narrow_openai_family_and_rejects_unsupported_tiers() -> None:
    for model in (
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-pro-2026-06-01",
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-luna-2026-08-01",
    ):
        assert resolve_model_catalog_entry("openai", model, "chat", "openai_responses") is not None
    for model in ("gpt-5.4-nano", "gpt-5.5-mini", "gpt-5.6-pro", "gpt-6"):
        assert resolve_model_catalog_entry("openai", model, "chat", "openai_responses") is None


def test_catalog_unknown_model_does_not_manufacture_capabilities() -> None:
    assert resolve_model_catalog_entry("openai", "future-model", "chat", "openai_responses") is None


def test_catalog_resolves_overlapping_facts_within_capability_domain() -> None:
    tool_search = resolve_model_catalog_entry("openai", "gpt-5.6", "chat", "openai_responses")
    structured = resolve_model_catalog_capability_entry(
        "openai",
        "gpt-5.6",
        "chat",
        "openai_responses",
        capability="structured_output",
    )
    prompt_cache = resolve_model_catalog_capability_entry(
        "openai",
        "gpt-5.6",
        "chat",
        "openai_responses",
        capability="prompt_cache",
    )

    assert tool_search is not None and tool_search.native_tool_search
    assert structured is not None and structured.structured_output is not None
    assert structured.structured_output.native_strict_schema
    assert prompt_cache is not None and prompt_cache.prompt_cache is not None
    assert prompt_cache.prompt_cache.mode == "explicit"


def test_tool_discovery_resolution_reads_production_catalog() -> None:
    openai = resolve_tool_discovery_capabilities("openai", "gpt-5.6", "responses")
    anthropic = resolve_tool_discovery_capabilities(
        "anthropic", "claude-sonnet-4-5-20250929", "messages"
    )

    assert openai is not None
    assert tuple(mode.mode for mode in openai.supported_modes) == (
        "native_hosted",
        "native_client",
    )
    assert anthropic is not None
    assert tuple(mode.mode for mode in anthropic.supported_modes) == (
        "native_hosted",
        "native_client",
    )
    assert (
        resolve_tool_discovery_capabilities("google", "gemini-2.5-pro", "generateContent") is None
    )


def test_positive_native_search_entry_requires_verified_evidence() -> None:
    with pytest.raises(ValidationError, match="verified URL evidence"):
        ModelCatalogEntry.model_validate(
            {
                "catalog_key": "test/model/v1",
                "provider_id": "openai",
                "operation": "chat",
                "endpoint_ids": ["openai_responses"],
                "model_id": "test-model",
                "native_tool_search": [
                    {
                        "mode": "native_client",
                        "replay_requirement": "previous_response",
                        "result_limit_behavior": "request_bound",
                        "max_results": 5,
                        "protocol_version": "test.v1",
                        "selection_owner": "application",
                        "tool_representation": "search_schema_manifest",
                        "inventory_timing": "search",
                        "path_transport": "manifest",
                    }
                ],
                "sources": [],
                "verified_at": "2026-08-12",
                "catalog_revision": 1,
                "evidence_status": "unknown",
            }
        )


def test_catalog_entry_rejects_multiple_capability_domains() -> None:
    entry = resolve_model_catalog_capability_entry(
        "openai",
        "text-embedding-3-small",
        "embeddings",
        "openai_embeddings",
        capability="embeddings",
    )
    assert entry is not None
    payload = entry.model_dump(mode="json")
    payload["image_generation"] = {
        "text_to_image": "supported",
        "image_editing": "unknown",
        "multiple_outputs": "unknown",
    }

    with pytest.raises(ValidationError, match="exactly one capability domain"):
        ModelCatalogEntry.model_validate(payload)

    payload.pop("image_generation")
    payload["operation"] = "chat"
    with pytest.raises(ValidationError, match="domain does not match operation"):
        ModelCatalogEntry.model_validate(payload)


def test_equal_priority_catalog_matches_fail_closed() -> None:
    payload = load_model_catalog().entries[0].model_dump(mode="json")
    payload["catalog_key"] = "test/duplicate-one/v1"
    payload["model_pattern"] = "gpt-5\\.6"
    first = ModelCatalogEntry.model_validate(payload)
    payload["catalog_key"] = "test/duplicate-two/v1"
    second = ModelCatalogEntry.model_validate(payload)
    catalog = ModelCatalog(
        schema_version="aethergraph.model-catalog/v1",
        catalog_revision=1,
        entries=(first, second),
    )

    with pytest.raises(ValueError, match="Ambiguous model catalog entries"):
        resolve_model_catalog_entry(
            "openai",
            "gpt-5.6",
            "chat",
            "openai_responses",
            catalog=catalog,
        )


def test_test_fixture_no_longer_owns_capability_truth() -> None:
    fixture = Path(__file__).parent / "fixtures" / "llm_tool_discovery" / "provider_matrix.json"
    assert fixture.is_file()
    assert load_model_catalog().entries


def test_resolver_combines_catalog_and_adapter_with_provenance() -> None:
    profile = chat_profile_from_legacy(LLMProfile(provider="openai", model="gpt-5.6"))

    binding = resolve_chat_profile(
        profile,
        required=("native_tool_search_client",),
    )

    assert binding.valid
    assert binding.catalog_key == "openai/gpt-5.4-5.6-native-tool-search/v1"
    capability = binding.capabilities.native_tool_search_client
    assert capability.state == "supported"
    assert capability.provenance[0].source == "catalog"


def test_resolver_uses_structured_output_and_prompt_cache_catalog_domains() -> None:
    profile = chat_profile_from_legacy(LLMProfile(provider="openai", model="gpt-5.6"))

    binding = resolve_chat_profile(
        profile,
        required=("structured_output", "prompt_cache"),
    )

    assert binding.valid
    assert binding.capabilities.structured_output.state == "supported"
    assert binding.capabilities.prompt_cache.state == "supported"
    assert "openai/current-structured-output/v2" in binding.catalog_keys
    assert "openai/gpt-5.6-plus-explicit-prompt-cache/v2" in binding.catalog_keys


def test_resolver_unknown_does_not_satisfy_required_capability() -> None:
    profile = chat_profile_from_legacy(
        LLMProfile(
            provider="openai_compatible",
            model="future-model",
            base_url="http://localhost:9000/v1",
        )
    )

    binding = resolve_chat_profile(profile, required=("prompt_cache",))

    assert not binding.valid
    assert binding.capabilities.prompt_cache.state == "unknown"
    assert binding.diagnostics[0].code == "required_capability_unknown"


def test_override_supplies_unknown_model_fact_when_adapter_implements_feature() -> None:
    profile = chat_profile_from_legacy(
        LLMProfile(provider="anthropic", model="claude-sonnet-4-5-20250929")
    ).model_copy(update={"capability_overrides": ChatCapabilityOverrides(image_input="supported")})

    binding = resolve_chat_profile(profile, required=("image_input",))

    assert binding.valid
    assert binding.capabilities.image_input.state == "supported"
    assert binding.capabilities.image_input.provenance[-1].source == "override"


def test_override_is_clamped_when_adapter_explicitly_lacks_feature() -> None:
    profile = chat_profile_from_legacy(
        LLMProfile(provider="openrouter", model="openai/gpt-5.6")
    ).model_copy(
        update={
            "capability_overrides": ChatCapabilityOverrides(native_tool_search_client="supported")
        }
    )

    binding = resolve_chat_profile(
        profile,
        required=("native_tool_search_client",),
    )

    assert not binding.valid
    capability = binding.capabilities.native_tool_search_client
    assert capability.state == "unsupported"
    assert capability.provenance[-1].source == "adapter"


def test_embedding_resolution_clamps_unprojected_dimensions() -> None:
    profile = EmbeddingProfileSpec(
        connection=ProviderConnection(
            provider_id="openai",
            endpoint_id="openai_embeddings",
        ),
        model=ModelSelection(model_id="text-embedding-3-large"),
    )

    binding = resolve_embedding_profile(profile, required=("text_embeddings", "dimensions"))

    assert not binding.valid
    assert binding.catalog_key == "openai/text-embedding-3/v3"
    assert binding.capabilities.text_embeddings.state == "supported"
    assert binding.capabilities.dimensions.state == "unsupported"
    assert binding.capabilities.dimensions.provenance[0].source == "catalog"
    assert binding.capabilities.dimensions.provenance[-1].source == "adapter"
    assert binding.diagnostics[0].capability == "dimensions"


def test_embedding_override_cannot_bypass_adapter_implementation() -> None:
    profile = EmbeddingProfileSpec(
        connection=ProviderConnection(
            provider_id="openai",
            endpoint_id="openai_embeddings",
        ),
        model=ModelSelection(model_id="future-embedding-model"),
        capability_overrides=EmbeddingCapabilityOverrides(
            text_embeddings="supported",
            dimensions="supported",
        ),
    )

    binding = resolve_embedding_profile(profile, required=("text_embeddings", "dimensions"))

    assert not binding.valid
    assert binding.catalog_key is None
    assert binding.capabilities.text_embeddings.state == "supported"
    assert binding.capabilities.dimensions.state == "unsupported"
    assert [item.source for item in binding.capabilities.dimensions.provenance] == [
        "unknown",
        "override",
        "adapter",
    ]


def test_image_resolution_exposes_catalog_truth_and_adapter_clamps() -> None:
    openai = ImageGenerationProfile(
        connection=ProviderConnection(
            provider_id="openai",
            endpoint_id="openai_images",
        ),
        model=ModelSelection(model_id="gpt-image-1.5"),
    )
    google = ImageGenerationProfile(
        connection=ProviderConnection(
            provider_id="google",
            endpoint_id="gemini_image_generation",
        ),
        model=ModelSelection(model_id="gemini-3.1-flash-image"),
    )

    openai_binding = resolve_image_generation_profile(
        openai,
        required=("text_to_image", "image_editing", "multiple_outputs"),
    )
    google_binding = resolve_image_generation_profile(
        google,
        required=("text_to_image", "image_editing", "multiple_outputs"),
    )

    assert openai_binding.capabilities.text_to_image.state == "supported"
    assert openai_binding.capabilities.multiple_outputs.state == "supported"
    assert openai_binding.capabilities.image_editing.state == "unsupported"
    assert openai_binding.capabilities.image_editing.provenance[-1].source == "adapter"
    assert not openai_binding.valid
    assert google_binding.capabilities.text_to_image.state == "supported"
    assert google_binding.capabilities.image_editing.state == "supported"
    assert google_binding.capabilities.multiple_outputs.state == "unsupported"
    assert google_binding.capabilities.multiple_outputs.provenance[0].state == "unknown"
    assert google_binding.capabilities.multiple_outputs.provenance[-1].source == "adapter"
    assert not google_binding.valid


def test_operation_resolution_preserves_unknown_and_validates_endpoint_operation() -> None:
    profile = ImageGenerationProfile(
        connection=ProviderConnection(
            provider_id="openai",
            endpoint_id="openai_images",
        ),
        model=ModelSelection(model_id="future-image-model"),
        capability_overrides=ImageGenerationCapabilityOverrides(text_to_image="supported"),
    )

    binding = resolve_image_generation_profile(profile, required=("text_to_image",))

    assert binding.valid
    assert binding.catalog_key is None
    assert binding.capabilities.image_editing.state == "unsupported"
    assert binding.capabilities.image_editing.provenance[0].source == "unknown"

    without_override = resolve_image_generation_profile(
        profile.model_copy(update={"capability_overrides": ImageGenerationCapabilityOverrides()})
    )
    assert without_override.capabilities.text_to_image.state == "unknown"

    invalid_endpoint = EmbeddingProfileSpec(
        connection=ProviderConnection(
            provider_id="openai",
            endpoint_id="openai_responses",
        ),
        model=ModelSelection(model_id="text-embedding-3-small"),
    )
    with pytest.raises(ValueError, match="does not implement embeddings"):
        resolve_embedding_profile(invalid_endpoint)


def test_complete_request_resolution_combines_adapter_and_model_requirements() -> None:
    profile = chat_profile_from_legacy(LLMProfile(provider="openai", model="gpt-5.6")).model_copy(
        update={"capability_overrides": ChatCapabilityOverrides(native_tool_calling="supported")}
    )
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(
            ToolDefinition(
                name="lookup",
                description="Look up one value.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        tool_choice="auto",
    )

    resolved = resolve_model_request(profile, request)

    assert resolved.valid is True
    assert resolved.binding.endpoint_id == "openai_responses"
    assert resolved.compatibility.required_model_capabilities == ("native_tool_calling",)


def test_complete_request_resolution_never_switches_incompatible_adapter() -> None:
    profile = chat_profile_from_legacy(
        LLMProfile(provider="openrouter", model="openai/gpt-5.6")
    ).model_copy(
        update={
            "capability_overrides": ChatCapabilityOverrides(
                native_tool_calling="supported",
                native_tool_search_client="supported",
            )
        }
    )
    request = ModelRequest(
        messages=(message_from_text("user", "Find a Tool"),),
        tools=(
            ToolDefinition(
                name="lookup",
                description="Look up one value.",
                input_schema={"type": "object", "properties": {}},
                exposure="deferred",
            ),
        ),
        tool_choice="auto",
        native_tool_search=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
    )

    resolved = resolve_model_request(profile, request)

    assert resolved.valid is False
    assert resolved.binding.endpoint_id == "openai_chat_completions"
    assert resolved.compatibility.diagnostics[0].code == ("adapter_capability_unimplemented")
