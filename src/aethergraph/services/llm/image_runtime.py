"""Shared image-generation invocation lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
from typing import Any

from aethergraph.services.llm.adapters import ImageAdapterInvocation, invoke_image_adapter
from aethergraph.services.llm.types import ImageGenerationResult, ImageGenerationUsage
from aethergraph.services.tracing import resolve_tracer

ImageUsageAccountant = Callable[
    [str, ImageGenerationUsage, int, str | None, str | None, int], Awaitable[None]
]


async def _execute_image_generation(
    host: Any,
    *,
    adapter_id: str,
    invocation: ImageAdapterInvocation,
    account_usage: ImageUsageAccountant,
    dimensions: dict[str, Any],
) -> ImageGenerationResult:
    reservation = host._operation_quota.reserve({"calls": 1, "images": invocation.n})
    start = time.perf_counter()
    span = None
    try:
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
        usage = result.usage_receipt or ImageGenerationUsage.from_provider_usage(result.usage)
        quota_error = host._operation_quota.reconcile(
            reservation,
            {
                "calls": 1,
                "images": len(result.images or []),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
            usage=usage.to_dict(),
        )
        await account_usage(
            invocation.model,
            usage,
            len(result.images or []),
            invocation.size,
            invocation.quality,
            latency_ms,
        )
        if quota_error is not None:
            raise quota_error
        usage_payload = usage.to_dict()
        await span.finish(
            response={"usage": usage_payload, "images_count": len(result.images or [])},
            metadata=dimensions,
            metrics={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "images_count": len(result.images or []),
                "latency_ms": latency_ms,
            },
        )
        return result
    except BaseException as exc:
        host._operation_quota.release(reservation)
        if span is not None:
            await span.fail(
                exc,
                metadata=dimensions,
                metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
            )
        raise
