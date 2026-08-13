"""Shared image-generation invocation lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aethergraph.services.llm.adapters import ImageAdapterInvocation, invoke_image_adapter
from aethergraph.services.llm.operation_runtime import (
    OperationTraceProjection,
    execute_model_operation,
)
from aethergraph.services.llm.types import ImageGenerationResult, ImageGenerationUsage

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
    def _usage(result: ImageGenerationResult) -> ImageGenerationUsage:
        return result.usage_receipt or ImageGenerationUsage.from_provider_usage(result.usage)

    async def _account(result: ImageGenerationResult, latency_ms: int) -> None:
        usage = _usage(result)
        await account_usage(
            invocation.model,
            usage,
            len(result.images or []),
            invocation.size,
            invocation.quality,
            latency_ms,
        )

    return await execute_model_operation(
        host,
        model=invocation.model,
        provider_operation="image",
        requested_quota={"calls": 1, "images": invocation.n},
        attempt=lambda: invoke_image_adapter(
            host,
            adapter_id=adapter_id,
            invocation=invocation,
        ),
        actual_quota=lambda result: {
            "calls": 1,
            "images": len(result.images or []),
            "input_tokens": _usage(result).input_tokens,
            "output_tokens": _usage(result).output_tokens,
            "total_tokens": _usage(result).total_tokens,
        },
        usage_payload=lambda result: _usage(result).to_dict(),
        account_usage=_account,
        trace=OperationTraceProjection(
            service="llm",
            operation="generate_image",
            request={
                "provider": host.provider,
                "model": invocation.model,
                "prompt_chars": len(invocation.prompt),
                "n": invocation.n,
                "size": invocation.size,
                "endpoint_id": adapter_id,
            },
            tags=("llm", "image"),
            response=lambda result: {
                "usage": _usage(result).to_dict(),
                "images_count": len(result.images or []),
            },
            metrics=lambda result: {
                "input_tokens": _usage(result).input_tokens,
                "output_tokens": _usage(result).output_tokens,
                "total_tokens": _usage(result).total_tokens,
                "images_count": len(result.images or []),
            },
        ),
        dimensions=dimensions,
    )
