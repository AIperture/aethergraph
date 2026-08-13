"""Shared normalized metering projection for model operations."""

from __future__ import annotations

import logging
from typing import Any

from aethergraph.contracts.services.metering import MeteringService
from aethergraph.services.llm.usage import normalize_llm_usage, normalized_usage_metrics


async def _record_model_metering(
    metering: MeteringService | None,
    *,
    provider: str,
    model: str,
    usage: dict[str, Any],
    latency_ms: int | None,
    dimensions: dict[str, Any],
    logger: logging.Logger,
) -> None:
    if metering is None:
        return
    normalized = normalized_usage_metrics(normalize_llm_usage(usage))
    try:
        await metering.record_llm(
            user_id=dimensions.get("user_id"),
            org_id=dimensions.get("org_id"),
            run_id=dimensions.get("run_id"),
            model=model,
            provider=provider,
            prompt_tokens=normalized["input_tokens"],
            completion_tokens=normalized["output_tokens"],
            cache_read_tokens=normalized["cache_read_tokens"],
            cache_write_tokens=normalized["cache_write_tokens"],
            uncached_input_tokens=normalized["uncached_input_tokens"],
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.warning("model_metering_failed: %s", exc)
