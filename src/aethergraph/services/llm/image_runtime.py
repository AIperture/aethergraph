"""Shared image-generation invocation lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import Any

from aethergraph.services.llm.adapters import ImageAdapterInvocation, invoke_image_adapter
from aethergraph.services.llm.types import ImageGenerationResult
from aethergraph.services.tracing import resolve_tracer

ImageUsageAccountant = Callable[[str, dict[str, Any], int], Awaitable[None]]


async def _execute_image_generation(
    host: Any,
    *,
    adapter_id: str,
    invocation: ImageAdapterInvocation,
    account_usage: ImageUsageAccountant,
    dimensions: dict[str, Any],
) -> ImageGenerationResult:
    await host._ensure_client()
    tracer = resolve_tracer()
    span = await tracer.start_span(
        service="llm",
        operation="generate_image",
        request={
            "provider": host.provider,
            "model": invocation.model,
            "prompt": invocation.prompt,
            "n": invocation.n,
            "size": invocation.size,
            "endpoint_id": adapter_id,
        },
        tags=["llm", "image"],
        metadata=dimensions,
    )
    start = time.perf_counter()
    try:
        provider_result = await host._provider_retry.execute(
            lambda: invoke_image_adapter(
                host,
                adapter_id=adapter_id,
                invocation=invocation,
            ),
            provider=host.provider,
            model=invocation.model,
            operation="image",
            rate_limit_group=host.rate_limit_group,
        )
        result = provider_result.value
        latency_ms = int((time.perf_counter() - start) * 1000)
        await account_usage(invocation.model, result.usage or {}, latency_ms)
        await span.finish(
            response={"usage": result.usage or {}, "images_count": len(result.images or [])},
            metadata=dimensions,
            metrics={**(result.usage or {}), "latency_ms": latency_ms},
        )
        return result
    except Exception as exc:
        await span.fail(
            exc,
            metadata=dimensions,
            metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
        )
        raise
