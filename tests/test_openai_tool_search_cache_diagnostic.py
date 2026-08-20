from __future__ import annotations

import pytest

from tests.live.openai_tool_search_cache_scenario import (
    assert_completed_response,
    build_cache_scenario,
    normalize_usage,
    raw_client_search_body,
    raw_hosted_search_body,
    request_fingerprint,
)


def test_cache_diagnostic_replays_have_stable_request_identity() -> None:
    scenario = build_cache_scenario("gpt-5.6-luna")

    client_first = raw_client_search_body(scenario, cache_key="client-key")
    client_replay = raw_client_search_body(scenario, cache_key="client-key")
    hosted_first = raw_hosted_search_body(scenario, cache_key="hosted-key")
    hosted_replay = raw_hosted_search_body(scenario, cache_key="hosted-key")

    assert request_fingerprint(client_first) == request_fingerprint(client_replay)
    assert request_fingerprint(hosted_first) == request_fingerprint(hosted_replay)
    assert request_fingerprint(client_first) != request_fingerprint(hosted_first)
    assert hosted_first["tools"][-1] == {"type": "tool_search"}
    assert len(scenario.tools) == 50


def test_cache_diagnostic_normalizes_current_and_compatible_usage() -> None:
    current = normalize_usage(
        {
            "input_tokens": 2_048,
            "output_tokens": 32,
            "input_tokens_details": {"cached_tokens": 1_024},
        }
    )
    compatible = normalize_usage(
        {
            "prompt_tokens": 512,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 256},
        }
    )

    assert current.input_tokens == 2_048
    assert current.output_tokens == 32
    assert current.cache_read_tokens == 1_024
    assert compatible.input_tokens == 512
    assert compatible.output_tokens == 8
    assert compatible.cache_read_tokens == 256


def test_cache_diagnostic_rejects_incomplete_provider_response() -> None:
    with pytest.raises(AssertionError):
        assert_completed_response(
            {
                "id": "resp-incomplete",
                "status": "incomplete",
                "error": None,
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        )
