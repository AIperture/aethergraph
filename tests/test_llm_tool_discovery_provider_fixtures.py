from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aethergraph.services.llm import (
    ToolDiscoveryCapabilities,
    ToolDiscoveryModeCapability,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "llm_tool_discovery" / "provider_matrix.json"
_EXPECTED_PROVIDERS = {"openai", "azure", "anthropic", "google"}
_OFFICIAL_SOURCE_HOSTS = {
    "developers.openai.com",
    "learn.microsoft.com",
    "platform.claude.com",
    "ai.google.dev",
}


def _matrix() -> dict[str, Any]:
    with _FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _record(provider: str) -> dict[str, Any]:
    return next(row for row in _matrix()["records"] if row["binding"]["provider"] == provider)


def test_provider_matrix_has_only_exact_bounded_bindings() -> None:
    matrix = _matrix()

    assert matrix["schema_version"] == "aethergraph.provider-discovery-fixture/v1"
    assert matrix["captured_at"] == "2026-08-07"
    records = matrix["records"]
    assert {row["binding"]["provider"] for row in records} == _EXPECTED_PROVIDERS

    bindings: set[tuple[str, str, str]] = set()
    for row in records:
        binding = row["binding"]
        identity = (
            binding["provider"],
            binding["model"],
            binding["endpoint_family"],
        )
        assert identity not in bindings
        bindings.add(identity)
        assert all(value.strip() == value and value for value in identity)
        assert not any(token in binding["model"] for token in ("*", "<", ">"))
        assert binding["endpoint_path"].startswith("/")

        mode_names = [mode["mode"] for mode in row["modes"]]
        assert len(mode_names) == len(set(mode_names))
        for mode in row["modes"]:
            assert mode["mode"] in {
                "native_hosted",
                "native_client",
                "engine_projected",
            }
            assert mode["replay_requirement"] in {
                "none",
                "previous_response",
                "full_history",
            }
            assert mode["result_limit_behavior"] in {
                "request_bound",
                "provider_fixed",
            }
            assert mode["protocol_version"]
            if mode["max_results"] is None:
                assert mode["bindable"] is False
                assert mode["result_limit_behavior"] == "provider_fixed"
                assert mode["blocker"] == "provider_reference_limit_not_documented"
            else:
                assert 1 <= mode["max_results"] <= 50

        assert row["sources"]
        assert all(urlparse(source).hostname in _OFFICIAL_SOURCE_HOSTS for source in row["sources"])


def test_bindable_fixture_modes_use_existing_capability_values() -> None:
    for row in _matrix()["records"]:
        bindable_modes = tuple(
            ToolDiscoveryModeCapability(
                mode=mode["mode"],
                replay_requirement=mode["replay_requirement"],
                result_limit_behavior=mode["result_limit_behavior"],
                max_results=mode["max_results"],
                protocol_version=mode["protocol_version"],
            )
            for mode in row["modes"]
            if mode["bindable"]
        )
        if not bindable_modes:
            continue

        binding = row["binding"]
        capability = ToolDiscoveryCapabilities(
            provider=binding["provider"],
            model=binding["model"],
            endpoint_family=binding["endpoint_family"],
            supported_modes=bindable_modes,
        )

        assert capability.provider == binding["provider"]
        assert capability.model == binding["model"]
        assert tuple(mode.mode for mode in capability.supported_modes) == tuple(
            mode["mode"] for mode in row["modes"] if mode["bindable"]
        )


def test_openai_and_azure_client_replay_preserves_order_and_call_identity() -> None:
    for provider in ("openai", "azure"):
        baseline = _record(provider)["baseline"]
        search_tool = next(
            tool for tool in baseline["request"]["tools"] if tool["type"] == "tool_search"
        )
        item_types = [item["type"] for item in baseline["response_items"]]
        assert search_tool["execution"] == "client"
        assert search_tool["parameters"]["required"] == ["goal"]
        assert search_tool["parameters"]["additionalProperties"] is False
        assert item_types[:2] == ["tool_search_call", "tool_search_output"]
        assert baseline["response_items"][0]["call_id"] == baseline["response_items"][1]["call_id"]
        assert baseline["continuation"]["call_id_must_match"] is True
        assert baseline["cache_observation"]["expected_catalog_prefix"] == "stable"

    azure = _record("azure")
    assert azure["binding"]["endpoint_family"] == "responses"
    assert azure["adapter_status"] == "native_client_implemented"
    assert (
        next(mode for mode in azure["modes"] if mode["mode"] == "native_client")["bindable"] is True
    )
    assert (
        next(mode for mode in azure["modes"] if mode["mode"] == "native_hosted")["bindable"]
        is False
    )


def test_anthropic_fixture_freezes_reference_limit_replay_and_cache_rules() -> None:
    row = _record("anthropic")
    hosted = next(mode for mode in row["modes"] if mode["mode"] == "native_hosted")
    baseline = row["baseline"]

    assert hosted["max_results"] == 5
    assert hosted["result_limit_behavior"] == "provider_fixed"
    assert hosted["replay_requirement"] == "full_history"
    assert row["required_headers"] == {"anthropic-version": "2023-06-01"}
    assert [item["type"] for item in baseline["response_items"]] == [
        "server_tool_use",
        "tool_search_tool_result",
        "tool_use",
    ]
    assert baseline["response_items"][0]["id"] == baseline["response_items"][1]["tool_use_id"]
    assert baseline["continuation"] == {
        "replay_assistant_content_unchanged": True,
        "send_same_tools_array": True,
        "do_not_return_server_tool_result": True,
    }
    assert baseline["cache_observation"]["expected_catalog_prefix"] == "stable"


def test_projected_gemini_fixture_records_expected_surface_rotation() -> None:
    row = _record("google")
    baseline = row["baseline"]
    before = baseline["request_before_activation"]["tools"][0]["functionDeclarations"]
    after = baseline["request_after_activation"]["tools"][0]["functionDeclarations"]

    assert [item["name"] for item in before] == ["tool_search", "tool_load"]
    assert [item["name"] for item in after] == [
        "tool_search",
        "tool_load",
        "get_weather",
    ]
    assert baseline["cache_observation"] == {
        "strategy": "projected_surface_rotates",
        "expected_catalog_prefix": "changed_after_activation",
        "provider_signal": "usageMetadata.cachedContentTokenCount",
    }
