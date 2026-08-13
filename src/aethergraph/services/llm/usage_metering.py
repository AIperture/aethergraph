"""Shared normalized metering projection for model operations."""

from __future__ import annotations

import logging
from typing import Any

from aethergraph.contracts.services.metering import MeteringService
from aethergraph.services.llm.types import EmbeddingUsage, ImageGenerationUsage
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


async def _record_embedding_metering(
    metering: MeteringService | None,
    *,
    provider: str,
    model: str,
    usage: EmbeddingUsage,
    num_texts: int,
    latency_ms: int,
    dimensions: dict[str, Any],
    logger: logging.Logger,
) -> None:
    if metering is None:
        return
    try:
        await metering.record_embedding(
            user_id=dimensions.get("user_id"),
            org_id=dimensions.get("org_id"),
            run_id=dimensions.get("run_id"),
            graph_id=dimensions.get("graph_id"),
            provider=provider,
            model=model,
            num_texts=num_texts,
            tokens=usage.input_tokens,
            usage_availability=usage.availability,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.warning("embedding_metering_failed: %s", exc)


async def _record_image_generation_metering(
    metering: MeteringService | None,
    *,
    provider: str,
    model: str,
    usage: ImageGenerationUsage,
    image_count: int,
    size: str | None,
    quality: str | None,
    latency_ms: int,
    dimensions: dict[str, Any],
    logger: logging.Logger,
) -> None:
    if metering is None:
        return
    record = getattr(metering, "record_image_generation", None)
    if record is None:
        logger.warning("image_generation_metering_unsupported")
        return
    try:
        await record(
            user_id=dimensions.get("user_id"),
            org_id=dimensions.get("org_id"),
            run_id=dimensions.get("run_id"),
            graph_id=dimensions.get("graph_id"),
            provider=provider,
            model=model,
            image_count=image_count,
            size=size,
            quality=quality,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            usage_availability=usage.availability,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.warning("image_generation_metering_failed: %s", exc)
