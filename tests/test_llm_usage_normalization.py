from __future__ import annotations

from aethergraph.services.llm.usage import normalize_llm_usage, normalized_usage_metrics


def test_normalize_llm_usage_maps_anthropic_cache_fields() -> None:
    raw = {
        "input_tokens": 12,
        "output_tokens": 5,
        "cache_creation_input_tokens": 30,
        "cache_read_input_tokens": 40,
    }

    normalized = normalize_llm_usage(raw)

    assert normalized["input_tokens"] == 12
    assert normalized["output_tokens"] == 5
    assert normalized["cache_write_tokens"] == 30
    assert normalized["cache_read_tokens"] == 40
    assert normalized["uncached_input_tokens"] == 12
    assert normalized["provider_usage_raw"] == raw


def test_normalize_llm_usage_maps_openai_cached_tokens() -> None:
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 64},
    }

    normalized = normalize_llm_usage(raw)

    assert normalized["input_tokens"] == 100
    assert normalized["output_tokens"] == 7
    assert normalized["cache_read_tokens"] == 64
    assert normalized["cache_write_tokens"] == 0
    assert normalized["uncached_input_tokens"] == 36


def test_normalize_llm_usage_maps_deepseek_cache_hits() -> None:
    raw = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "prompt_cache_hit_tokens": 8,
        "prompt_cache_miss_tokens": 4,
    }

    normalized = normalize_llm_usage(raw)

    assert normalized["input_tokens"] == 12
    assert normalized["output_tokens"] == 3
    assert normalized["cache_read_tokens"] == 8
    assert normalized["cache_write_tokens"] == 0
    assert normalized["uncached_input_tokens"] == 4
    assert normalized["provider_usage_raw"] == raw


def test_normalize_llm_usage_maps_nested_openai_cache_writes() -> None:
    raw = {
        "input_tokens": 100,
        "output_tokens": 7,
        "input_tokens_details": {
            "cached_tokens": 60,
            "cache_write_tokens": 25,
        },
    }

    normalized = normalize_llm_usage(raw)

    assert normalized["input_tokens"] == 100
    assert normalized["output_tokens"] == 7
    assert normalized["cache_read_tokens"] == 60
    assert normalized["cache_write_tokens"] == 25
    assert normalized["uncached_input_tokens"] == 15


def test_normalize_llm_usage_maps_deepseek_automatic_cache_fields() -> None:
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "prompt_cache_hit_tokens": 64,
        "prompt_cache_miss_tokens": 36,
    }

    normalized = normalize_llm_usage(raw)

    assert normalized["input_tokens"] == 100
    assert normalized["output_tokens"] == 7
    assert normalized["cache_read_tokens"] == 64
    assert normalized["cache_write_tokens"] == 0
    assert normalized["uncached_input_tokens"] == 36
    assert normalized["provider_usage_raw"] == raw


def test_normalized_usage_metrics_excludes_raw_provider_payload() -> None:
    normalized = normalize_llm_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    )

    metrics = normalized_usage_metrics(normalized)

    assert metrics == {
        "input_tokens": 10,
        "output_tokens": 3,
        "cache_read_tokens": 4,
        "cache_write_tokens": 0,
        "uncached_input_tokens": 6,
    }
