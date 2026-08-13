"""Shared invocation lifecycle for non-Chat model operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any, Generic, TypeVar

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.llm.provider_transport import ProviderCallResult
from aethergraph.services.tracing import resolve_tracer

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class OperationTraceProjection(Generic[_ResultT]):
    """Describe sanitized tracing for one model-operation invocation."""

    service: str
    operation: str
    tags: tuple[str, ...]
    request: Mapping[str, Any]
    response: Callable[[_ResultT], Mapping[str, Any]]
    metrics: Callable[[_ResultT], Mapping[str, int | float | None]]


def model_operation_dimensions(
    *,
    profile_name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project common model-operation dimensions from the active runtime.

    Intro:
        Produces one consistent detached dimension map for operation tracing and
        metering while allowing explicit invocation values to override context.

    Examples:
        Read the active run dimensions:
            ```python
            dimensions = model_operation_dimensions()
            ```

        Supply a profile and explicit run identity:
            ```python
            dimensions = model_operation_dimensions(
                profile_name="research",
                overrides={"run_id": "run-1"},
            )
            assert dimensions["run_id"] == "run-1"
            ```

    Args:
        profile_name: Optional configured operation-profile identity.
        overrides: Optional explicit dimensions; non-`None` values replace the
            active runtime context.

    Returns:
        dict[str, Any]: Detached common dimension fields.

    Notes:
        Prompt text, embedding inputs, image bytes, and credentials are never
        included in this projection.
    """

    context = current_meter_context.get()
    explicit = dict(overrides or {})
    keys = (
        "user_id",
        "org_id",
        "run_id",
        "graph_id",
        "session_id",
        "app_id",
        "agent_id",
        "node_id",
        "trace_id",
        "span_id",
    )
    dimensions = {
        key: explicit.get(key) if explicit.get(key) is not None else context.get(key)
        for key in keys
    }
    dimensions["profile_name"] = profile_name
    return dimensions


async def execute_model_operation(
    host: Any,
    *,
    model: str,
    provider_operation: str,
    requested_quota: Mapping[str, int | None],
    attempt: Callable[[], Awaitable[ProviderCallResult[_ResultT]]],
    actual_quota: Callable[[_ResultT], Mapping[str, int | None]],
    usage_payload: Callable[[_ResultT], dict[str, Any]],
    account_usage: Callable[[_ResultT, int], Awaitable[None]],
    trace: OperationTraceProjection[_ResultT],
    dimensions: Mapping[str, Any],
) -> _ResultT:
    """Execute one complete non-Chat model-operation lifecycle.

    Intro:
        Owns atomic quota admission, lazy transport setup, sanitized tracing,
        retry/rate gating, actual-usage reconciliation, metering order, and
        cancellation-safe reservation release around one single-attempt adapter.

    Examples:
        Execute an embedding invocation:
            ```python
            result = await execute_model_operation(
                client,
                model=model,
                provider_operation="embedding",
                requested_quota={"calls": 1, "texts": len(texts)},
                attempt=attempt,
                actual_quota=actual_quota,
                usage_payload=usage_payload,
                account_usage=account_usage,
                trace=trace,
                dimensions=dimensions,
            )
            ```

        Execute image generation through the same lifecycle:
            ```python
            result = await execute_model_operation(
                client,
                model=model,
                provider_operation="image",
                requested_quota={"calls": 1, "images": count},
                attempt=attempt,
                actual_quota=actual_quota,
                usage_payload=usage_payload,
                account_usage=account_usage,
                trace=trace,
                dimensions=dimensions,
            )
            ```

    Args:
        host: Operation client owning transport, retry, rate, and quota state.
        model: Exact effective model identity for this invocation.
        provider_operation: Stable transport operation identity.
        requested_quota: Pre-dispatch logical quota reservation.
        attempt: Single provider attempt wrapped by the shared retry executor.
        actual_quota: Projection of actual result usage for reconciliation.
        usage_payload: Typed usage receipt projection for quota errors.
        account_usage: Exactly-once operation-specific metering callback.
        trace: Sanitized trace request, response, and metrics projection.
        dimensions: Detached common trace and meter dimensions.

    Returns:
        _ResultT: Exact normalized adapter result.

    Notes:
        Metering occurs after reconciliation records actual usage and before a
        post-response quota error is raised. Adapters remain single-attempt and
        never reserve quota, retry, meter, or emit terminal spans themselves.
    """

    reservation = host._operation_quota.reserve(requested_quota)
    start = time.perf_counter()
    span = None
    try:
        await host._ensure_client()
        span = await resolve_tracer().start_span(
            service=trace.service,
            operation=trace.operation,
            request=dict(trace.request),
            tags=list(trace.tags),
            metadata=dict(dimensions),
        )
        provider_result = await host._provider_retry.execute(
            attempt,
            provider=host.provider,
            model=model,
            operation=provider_operation,
            rate_limit_group=host.rate_limit_group,
        )
        result = provider_result.value
        latency_ms = int((time.perf_counter() - start) * 1000)
        quota_error = host._operation_quota.reconcile(
            reservation,
            actual_quota(result),
            usage=usage_payload(result),
        )
        await account_usage(result, latency_ms)
        if quota_error is not None:
            raise quota_error
        metrics = dict(trace.metrics(result))
        metrics["latency_ms"] = latency_ms
        await span.finish(
            response=dict(trace.response(result)),
            metadata=dict(dimensions),
            metrics=metrics,
        )
        return result
    except BaseException as exc:
        host._operation_quota.release(reservation)
        if span is not None:
            await span.fail(
                exc,
                metadata=dict(dimensions),
                metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
            )
        raise


__all__ = [
    "OperationTraceProjection",
    "execute_model_operation",
    "model_operation_dimensions",
]
