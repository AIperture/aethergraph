from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from aethergraph.config.config import (
    AppSettings,
    EmbeddingUsageQuotaSettings,
    ImageGenerationUsageQuotaSettings,
)
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient
from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.types import (
    ModelOperationRunQuotaExceededError,
    ModelOperationRunQuotaUnverifiableError,
    ModelOperationRunQuotaWouldExceedError,
)


def _response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://api.openai.com/v1/test"),
    )


def test_model_operation_quotas_default_to_unbounded() -> None:
    quota = AppSettings().model_operation_usage_quota

    assert quota.embedding.max_calls_per_run is None
    assert quota.embedding.max_texts_per_run is None
    assert quota.embedding.max_input_tokens_per_run is None
    assert quota.image_generation.max_calls_per_run is None
    assert quota.image_generation.max_images_per_run is None
    assert quota.image_generation.max_input_tokens_per_run is None
    assert quota.image_generation.max_output_tokens_per_run is None
    assert quota.image_generation.max_total_tokens_per_run is None


def test_model_operation_quotas_load_from_nested_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "AETHERGRAPH_MODEL_OPERATION_USAGE_QUOTA__EMBEDDING__MAX_TEXTS_PER_RUN",
        "7",
    )
    monkeypatch.setenv(
        "AETHERGRAPH_MODEL_OPERATION_USAGE_QUOTA__IMAGE_GENERATION__MAX_IMAGES_PER_RUN",
        "3",
    )

    quota = AppSettings().model_operation_usage_quota

    assert quota.embedding.max_texts_per_run == 7
    assert quota.image_generation.max_images_per_run == 3


@pytest.mark.asyncio
async def test_embedding_quota_reservation_is_atomic_and_released_on_cancellation() -> None:
    started = asyncio.Event()
    block = True

    class _Http:
        async def post(self, url, *, headers, json):
            if block:
                started.set()
                await asyncio.Future()
            return _response(
                {
                    "data": [{"embedding": [0.1]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )

    client = GenericEmbeddingClient(
        provider="openai",
        model="embed-test",
        api_key="test",
        operation_quota_cfg=EmbeddingUsageQuotaSettings(max_calls_per_run=1),
    )
    client._client = _Http()  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    context: dict[str, Any] = {"run_id": "run-embedding-concurrent"}
    token = current_meter_context.set(context)
    first = asyncio.create_task(client.embed_result(["first"]))
    try:
        await started.wait()
        with pytest.raises(ModelOperationRunQuotaWouldExceedError) as exc_info:
            await client.embed_result(["second"])
        assert exc_info.value.operation == "embedding"
        assert exc_info.value.quota == "calls"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        block = False
        result = await client.embed_result(["second"])
    finally:
        if not first.done():
            first.cancel()
        current_meter_context.reset(token)

    assert result.vectors == [[0.1]]
    state = context["_model_operation_usage_quota_state"]["embedding"]
    assert state["consumed"]["calls"] == 1
    assert state["reserved"]["calls"] == 0


@pytest.mark.asyncio
async def test_embedding_actual_tokens_are_metered_before_typed_quota_error() -> None:
    class _Http:
        async def post(self, url, *, headers, json):
            return _response(
                {
                    "data": [{"embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                }
            )

    class _Meter:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record_embedding(self, **record: Any) -> None:
            self.records.append(record)

    meter = _Meter()
    client = GenericEmbeddingClient(
        provider="openai",
        model="embed-test",
        api_key="test",
        metering=meter,  # type: ignore[arg-type]
        operation_quota_cfg=EmbeddingUsageQuotaSettings(max_input_tokens_per_run=2),
    )
    client._client = _Http()  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    token = current_meter_context.set({"run_id": "run-embedding-post"})
    try:
        with pytest.raises(ModelOperationRunQuotaExceededError) as exc_info:
            await client.embed_result(["hello"])
    finally:
        current_meter_context.reset(token)

    assert exc_info.value.operation == "embedding"
    assert exc_info.value.quota == "input_tokens"
    assert exc_info.value.projected == 3
    assert exc_info.value.usage is not None
    assert exc_info.value.usage["input_tokens"] == 3
    assert len(meter.records) == 1
    assert meter.records[0]["tokens"] == 3


@pytest.mark.asyncio
async def test_unavailable_embedding_tokens_fail_closed_after_metering() -> None:
    class _Http:
        async def post(self, url, *, headers, json):
            return _response({"data": [{"embedding": [0.1]}]})

    class _Meter:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record_embedding(self, **record: Any) -> None:
            self.records.append(record)

    meter = _Meter()
    client = GenericEmbeddingClient(
        provider="openai",
        model="embed-test",
        api_key="test",
        metering=meter,  # type: ignore[arg-type]
        operation_quota_cfg=EmbeddingUsageQuotaSettings(max_input_tokens_per_run=10),
    )
    client._client = _Http()  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    context: dict[str, Any] = {"run_id": "run-embedding-unavailable"}
    token = current_meter_context.set(context)
    try:
        with pytest.raises(ModelOperationRunQuotaUnverifiableError) as exc_info:
            await client.embed_result(["hello"])
    finally:
        current_meter_context.reset(token)

    assert exc_info.value.quotas == ("input_tokens",)
    assert exc_info.value.usage is not None
    assert exc_info.value.usage["availability"] == "unavailable"
    assert meter.records[0]["usage_availability"] == "unavailable"
    state = context["_model_operation_usage_quota_state"]["embedding"]
    assert state["consumed"]["calls"] == 1
    assert state["consumed"]["texts"] == 1


@pytest.mark.asyncio
async def test_image_count_quota_rejects_before_transport_creation() -> None:
    client = GenericImageGenerationClient(
        provider="openai",
        model="gpt-image-test",
        endpoint_id="openai_images",
        api_key="test",
        operation_quota_cfg=ImageGenerationUsageQuotaSettings(max_images_per_run=1),
    )
    token = current_meter_context.set({"run_id": "run-image-preflight"})
    try:
        with pytest.raises(ModelOperationRunQuotaWouldExceedError) as exc_info:
            await client.generate_image("A compass", n=2)
    finally:
        current_meter_context.reset(token)

    assert exc_info.value.operation == "image_generation"
    assert exc_info.value.quota == "images"
    assert exc_info.value.requested == 2
    assert client._client is None


@pytest.mark.asyncio
async def test_image_actual_tokens_are_metered_before_typed_quota_error() -> None:
    class _Http:
        async def post(self, url, *, headers, json):
            return _response(
                {
                    "data": [{"b64_json": "aW1hZ2U="}],
                    "usage": {"input_tokens": 4, "output_tokens": 6},
                }
            )

    class _Meter:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record_image_generation(self, **record: Any) -> None:
            self.records.append(record)

    meter = _Meter()
    client = GenericImageGenerationClient(
        provider="openai",
        model="gpt-image-test",
        endpoint_id="openai_images",
        api_key="test",
        metering=meter,  # type: ignore[arg-type]
        operation_quota_cfg=ImageGenerationUsageQuotaSettings(max_total_tokens_per_run=5),
    )
    client._client = _Http()  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    token = current_meter_context.set({"run_id": "run-image-post"})
    try:
        with pytest.raises(ModelOperationRunQuotaExceededError) as exc_info:
            await client.generate_image("A glass compass")
    finally:
        current_meter_context.reset(token)

    assert exc_info.value.operation == "image_generation"
    assert exc_info.value.quota == "total_tokens"
    assert exc_info.value.projected == 10
    assert exc_info.value.usage is not None
    assert exc_info.value.usage["total_tokens"] == 10
    assert len(meter.records) == 1
    assert meter.records[0]["total_tokens"] == 10
