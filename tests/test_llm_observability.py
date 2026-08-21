from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aethergraph.config.config import AppSettings, LLMUsageQuotaSettings
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.observability import (
    LLMObservationRecord,
    ObservabilityFacade,
    ObservationPolicy,
)
from aethergraph.observability.canonical_inspection import CanonicalInspectionReader
from aethergraph.observability.canonical_service import ProviderObservationService
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.llm import (
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from aethergraph.services.llm.correlation import current_llm_call_correlation
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.observability import ConsoleLLMObservationSink
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateLimitSnapshot,
    ProviderResponseMetadata,
    ProviderRetrySettings,
)
from aethergraph.services.llm.types import (
    LLMContextWindowExceededError,
    LLMRunQuotaExceededError,
    LLMRunQuotaWouldExceedError,
)
from aethergraph.storage.contracts import StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)


def _record(
    *,
    prompt: str = "hello",
    output: str | None = "hello back",
    run_id: str = "run-1",
) -> LLMObservationRecord:
    record = LLMObservationRecord.new(
        call_type="chat",
        provider="openai",
        model="gpt-test",
        dimensions={"run_id": run_id, "graph_id": "graph-1"},
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=None,
        max_output_tokens=64,
        output_format="text",
        json_schema=None,
        schema_name="output",
        strict_schema=True,
        validate_json=True,
        extra_params={},
        request_args={"model": "gpt-test"},
        provider_request_args={"temperature": 0},
        compatibility_notes=[],
        trace_payload={"step": 1},
    )
    record.raw_text = output
    record.usage = {"prompt_tokens": 3, "completion_tokens": 2}
    record.latency_ms = 12
    return record


@pytest.mark.asyncio
async def test_console_sink_compact_view_renders_prompt_and_output(capsys) -> None:
    sink = ConsoleLLMObservationSink(prompt_view="compact", width=60, truncation_chars=80)
    record = _record(prompt="Explain attention.", output="Attention weights relevant inputs.")

    await sink.finish_llm_call(record, capture_mode="full")

    out = capsys.readouterr().out
    assert "LLM CALL  [-] openai/gpt-test  profile=default" in out
    assert "[USER]" in out
    assert "[OUTPUT]" in out
    assert "tokens:  in=3  out=2  total=5" in out


@pytest.mark.asyncio
async def test_llm_client_records_success_and_provider_error_canonically(
    tmp_path: Path,
) -> None:
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    sink = ProviderObservationService(
        repository=LocalObservationRepository(database=database),
        owner_scope=StorageScope(project_id="project-1"),
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    reader = CanonicalInspectionReader(sink)
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=sink,
        observation_capture_mode="manifest",
        retry_settings=ProviderRetrySettings(
            base_delay_s=0.0,
            max_backoff_s=0.0,
            jitter_ratio=0.0,
        ),
    )

    dispatch_calls = 0

    async def successful_dispatch(messages, **kwargs):
        nonlocal dispatch_calls
        dispatch_calls += 1
        if dispatch_calls == 1:
            raise LLMProviderRequestError(
                provider="openai",
                model="gpt-test",
                operation="chat",
                code="provider_rate_limited",
                message="Rate limit reached.",
                retryable=True,
                status_code=429,
                metadata=ProviderResponseMetadata(
                    request_id="req-limited",
                    retry_after_s=0.598,
                    rate_limits=(
                        ProviderRateLimitSnapshot(
                            resource="tokens",
                            limit=200000,
                            remaining=34571,
                            reset_after_s=0.598,
                        ),
                    ),
                ),
            )
        return ProviderCallResult(
            ("hello back", {"prompt_tokens": 11, "completion_tokens": 7}),
            ProviderResponseMetadata(request_id="req-success"),
        )

    client._chat_dispatch = successful_dispatch  # type: ignore[method-assign]
    token = current_meter_context.set({"run_id": "run-success"})
    try:
        await client.chat([{"role": "user", "content": "hello"}])
    finally:
        current_meter_context.reset(token)

    async def failed_dispatch(messages, **kwargs):
        raise RuntimeError("boom")

    client._chat_dispatch = failed_dispatch  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        await client.chat([{"role": "user", "content": "hello"}])

    inspect_page = await reader.list_llm_calls()
    assert len(inspect_page.items) == 2
    assert {row.error_type for row in inspect_page.items} == {None, "RuntimeError"}
    assert len({row.call_id for row in inspect_page.items}) == 2
    recovered = next(row for row in inspect_page.items if row.error_type is None)
    assert recovered.attempt_count == 2
    assert recovered.retry_count == 1
    assert recovered.attempts == []
    inspect_detail = await reader.get_llm_call(recovered.call_id)
    assert len(inspect_detail.attempts) == 2
    assert inspect_detail.total_retry_wait_ms == 598
    assert inspect_detail.attempts[0].error_code == "provider_rate_limited"
    assert inspect_detail.attempts[0].rate_limits[0].resource == "tokens"
    assert inspect_detail.attempts[1].request_id == "req-success"
    correlation = current_llm_call_correlation()
    assert correlation is not None
    assert correlation.llm_call_id in {row.call_id for row in inspect_page.items}
    await database.close()


@pytest.mark.asyncio
async def test_projected_discovery_model_calls_all_reach_the_inspect_reader(
    tmp_path: Path,
) -> None:
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    sink = ProviderObservationService(
        repository=LocalObservationRepository(database=database),
        owner_scope=StorageScope(project_id="project-projected-discovery"),
        policy=ObservationPolicy(capture_mode="metadata"),
    )
    reader = CanonicalInspectionReader(sink)
    client = GenericLLMClient(
        provider="openai",
        model="example-model",
        api_key="test",
        observation_sink=sink,
        observation_capture_mode="metadata",
    )
    selections = iter(("tool_search", "tool_load", "docs_read"))

    async def projected_dispatch(messages, **kwargs):
        del messages, kwargs
        name = next(selections)
        return ProviderCallResult(
            (
                ToolCallResponse(
                    items=(ToolCall(f"call-{name}", name, {}),),
                ),
                {"prompt_tokens": 3, "completion_tokens": 1},
            )
        )

    client._chat_dispatch = projected_dispatch  # type: ignore[method-assign]
    search = ToolDefinition("tool_search", "Search Tools.", {"type": "object"})
    load = ToolDefinition("tool_load", "Load Tools.", {"type": "object"})
    docs = ToolDefinition("docs_read", "Read a document.", {"type": "object"})
    token = current_meter_context.set(
        {
            "run_id": "run-projected-discovery",
            "turn_id": "turn-projected-discovery",
        }
    )
    try:
        for call_name, definitions in (
            ("select_tool_search", (search, load)),
            ("select_tool_load", (search, load)),
            ("select_docs_read", (search, load, docs)),
        ):
            response, _usage = await client.chat(
                [{"role": "user", "content": call_name}],
                tool_request=ToolCallRequest(
                    tools=definitions,
                    active_tool_names=tuple(tool.name for tool in definitions),
                    turn_id="turn-projected-discovery",
                ),
                call_name=call_name,
            )
            assert isinstance(response, ToolCallResponse)
    finally:
        current_meter_context.reset(token)

    page = await reader.list_llm_calls(
        run_id="run-projected-discovery",
        limit=10,
    )
    assert {item.call_name for item in page.items} == {
        "select_tool_search",
        "select_tool_load",
        "select_docs_read",
    }
    await database.close()


@pytest.mark.asyncio
async def test_cancelled_llm_call_is_persisted_and_reraised(tmp_path: Path) -> None:
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    sink = ProviderObservationService(
        repository=LocalObservationRepository(database=database),
        owner_scope=StorageScope(project_id="project-cancel"),
        policy=ObservationPolicy(capture_mode="metadata"),
    )
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=sink,
        observation_capture_mode="metadata",
    )
    transport_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def blocked_dispatch(messages, **kwargs):
        transport_started.set()
        await never_finishes.wait()
        raise AssertionError("unreachable")

    client._chat_dispatch = blocked_dispatch  # type: ignore[method-assign]
    task = asyncio.create_task(client.chat([{"role": "user", "content": "cancel"}]))
    await transport_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    page = await CanonicalInspectionReader(sink).list_llm_calls()
    assert len(page.items) == 1
    assert page.items[0].lifecycle_status == "cancelled"
    assert page.items[0].error_type == "CancelledError"
    await database.close()


@pytest.mark.asyncio
async def test_observation_sink_failure_cannot_change_a_successful_llm_result() -> None:
    class FailingSink:
        async def begin_llm_call(self, record, *, capture_mode: str) -> None:
            raise RuntimeError("observation store unavailable")

        async def finish_llm_call(self, record, *, capture_mode: str) -> None:
            raise RuntimeError("observation store unavailable")

    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=FailingSink(),
        observation_capture_mode="metadata",
    )

    async def successful_dispatch(messages, **kwargs):
        return ProviderCallResult(("provider result", {"prompt_tokens": 3, "completion_tokens": 2}))

    client._chat_dispatch = successful_dispatch  # type: ignore[method-assign]
    text, usage = await client.chat([{"role": "user", "content": "hello"}])

    assert text == "provider result"
    assert usage == {"prompt_tokens": 3, "completion_tokens": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "metadata", "manifest", "full"])
async def test_metering_is_independent_of_capture_mode(mode: str) -> None:
    class FakeMetering:
        def __init__(self) -> None:
            self.records = []

        async def record_llm(self, **record) -> None:
            self.records.append(record)

    metering = FakeMetering()

    class CaptureSink:
        async def begin_llm_call(self, record, *, capture_mode: str) -> None:
            assert capture_mode == mode
            assert record.lifecycle_status == "in_progress"

        async def finish_llm_call(self, record, *, capture_mode: str) -> None:
            assert capture_mode == mode
            assert record.lifecycle_status == "completed"

    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        metering=metering,
        observation_sink=CaptureSink(),
        observation_capture_mode=mode,  # type: ignore[arg-type]
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(
            (
                "ok",
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            )
        )

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    await client.chat([{"role": "user", "content": "hello"}])

    assert len(metering.records) == 1
    assert metering.records[0]["cache_read_tokens"] == 4
    assert metering.records[0]["uncached_input_tokens"] == 6


@pytest.mark.asyncio
async def test_default_container_uses_sqlite_without_legacy_observability_sinks(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        workspace=str(tmp_path),
        deploy_mode="local",
        llm={
            "enabled": True,
            "default": {"provider": "openai", "model": "gpt-container"},
            "observability": {"capture_mode": "manifest"},
        },
    )
    container = build_default_container(root=str(tmp_path), cfg=settings)
    assert container.llm is not None
    assert container.image_service is not None
    client = container.llm.get()
    assert container.image_service.get() is not client
    assert client._assigned_image_client is container.image_service.get()

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(
            ("container output", {"prompt_tokens": 9, "completion_tokens": 1})
        )

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    await client.chat([{"role": "user", "content": "from container"}])

    assert (tmp_path / "local" / "events.sqlite3").is_file()
    assert not list(tmp_path.rglob("llm_calls.jsonl"))
    assert not hasattr(container, "tracer")
    page = await container.storage_services.inspection().list_llm_calls()
    assert page.items[0].provider == "openai"
    assert isinstance(container.observability, ObservabilityFacade)
    assert await container.observability.list_runs(limit=10_000, offset=0) == []
    assert await container.observability.list_suppressed_scopes() == {
        "session_id": set(),
        "run_id": set(),
        "trace_id": set(),
    }
    assert await container.metering.get_overview() == {
        "llm_calls": 1,
        "llm_prompt_tokens": 9,
        "llm_completion_tokens": 1,
        "llm_cache_read_tokens": 0,
        "llm_cache_write_tokens": 0,
        "llm_uncached_input_tokens": 9,
        "embedding_calls": 0,
        "embedding_texts": 0,
        "embedding_tokens": 0,
        "image_generation_calls": 0,
        "images_generated": 0,
        "image_generation_tokens": 0,
        "runs": 0,
        "runs_succeeded": 0,
        "runs_failed": 0,
        "artifacts": 0,
        "artifact_bytes": 0,
        "events": 0,
    }
    await container.close_storage()


def test_llm_observability_defaults_to_manifest_capture() -> None:
    settings = AppSettings()
    policy = ObservationPolicy()

    assert settings.llm.observability.capture_mode == "manifest"
    assert policy.capture_mode == "manifest"


@pytest.mark.asyncio
async def test_llm_chat_preflight_uses_explicit_usage_quota_before_dispatch() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        usage_quota_cfg=LLMUsageQuotaSettings(max_total_tokens_per_run=200),
    )
    dispatched = False

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched = True
        return ProviderCallResult(("unexpected", {"prompt_tokens": 1, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    token = current_meter_context.set(
        {
            "run_id": "run-preflight-tight",
            "_llm_usage_quota_state": {
                "calls": 1,
                "input_tokens": 170,
                "output_tokens": 10,
            },
        }
    )
    try:
        with pytest.raises(LLMRunQuotaWouldExceedError) as exc_info:
            await client.chat(
                [{"role": "user", "content": "x" * 120}],
                max_output_tokens=40,
            )
    finally:
        current_meter_context.reset(token)

    assert dispatched is False
    assert exc_info.value.limit == 200
    assert exc_info.value.projected > 200
    assert exc_info.value.consumed == 180


@pytest.mark.asyncio
async def test_llm_post_call_quota_violation_raises_typed_error() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        usage_quota_cfg=LLMUsageQuotaSettings(max_total_tokens_per_run=50),
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(("hello back", {"prompt_tokens": 30, "completion_tokens": 25}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    token = current_meter_context.set({"run_id": "run-post"})
    try:
        with pytest.raises(LLMRunQuotaExceededError) as exc_info:
            await client.chat([{"role": "user", "content": "hello"}])
    finally:
        current_meter_context.reset(token)

    assert exc_info.value.projected == 55
    assert exc_info.value.limit == 50


@pytest.mark.asyncio
async def test_llm_quota_reservation_prevents_concurrent_run_oversubscription() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        usage_quota_cfg=LLMUsageQuotaSettings(max_calls_per_run=1),
    )
    dispatched = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched += 1
        if dispatched > 1:
            raise AssertionError("concurrent call reached provider dispatch")
        first_started.set()
        await release_first.wait()
        return ProviderCallResult(("ok", {"prompt_tokens": 1, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    context = {"run_id": "run-concurrent-reservation"}
    token = current_meter_context.set(context)
    first = asyncio.create_task(client.chat([{"role": "user", "content": "first"}]))
    try:
        await first_started.wait()
        with pytest.raises(LLMRunQuotaWouldExceedError):
            await client.chat([{"role": "user", "content": "second"}])
        assert dispatched == 1
        release_first.set()
        assert await first == ("ok", {"prompt_tokens": 1, "completion_tokens": 1})
    finally:
        release_first.set()
        if not first.done():
            first.cancel()
        current_meter_context.reset(token)

    state = context["_llm_usage_quota_state"]
    assert state["calls"] == 1
    assert state["reserved_calls"] == 0


@pytest.mark.asyncio
async def test_llm_quota_reservation_is_released_after_provider_failure() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        usage_quota_cfg=LLMUsageQuotaSettings(max_calls_per_run=1),
    )
    dispatched = 0

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched += 1
        if dispatched == 1:
            raise ValueError("provider failed")
        return ProviderCallResult(("ok", {"prompt_tokens": 1, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    context = {"run_id": "run-release-reservation"}
    token = current_meter_context.set(context)
    try:
        with pytest.raises(ValueError, match="provider failed"):
            await client.chat([{"role": "user", "content": "first"}])
        result = await client.chat([{"role": "user", "content": "second"}])
    finally:
        current_meter_context.reset(token)

    assert result == ("ok", {"prompt_tokens": 1, "completion_tokens": 1})
    state = context["_llm_usage_quota_state"]
    assert state["calls"] == 1
    assert state["reserved_calls"] == 0


@pytest.mark.asyncio
async def test_llm_quota_reservation_is_released_after_cancellation() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        usage_quota_cfg=LLMUsageQuotaSettings(max_calls_per_run=1),
    )
    first_started = asyncio.Event()
    block_dispatch = True

    async def fake_chat_dispatch(messages, **kwargs):
        if block_dispatch:
            first_started.set()
            await asyncio.Event().wait()
        return ProviderCallResult(("ok", {"prompt_tokens": 1, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    context = {"run_id": "run-cancel-reservation"}
    token = current_meter_context.set(context)
    first = asyncio.create_task(client.chat([{"role": "user", "content": "first"}]))
    try:
        await first_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        block_dispatch = False
        result = await client.chat([{"role": "user", "content": "second"}])
    finally:
        if not first.done():
            first.cancel()
        current_meter_context.reset(token)

    assert result == ("ok", {"prompt_tokens": 1, "completion_tokens": 1})
    state = context["_llm_usage_quota_state"]
    assert state["calls"] == 1
    assert state["reserved_calls"] == 0


@pytest.mark.asyncio
async def test_llm_context_window_is_current_request_only() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        context_window_tokens=20,
    )
    dispatched = False

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched = True
        return ProviderCallResult(("unexpected", {"prompt_tokens": 1, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    with pytest.raises(LLMContextWindowExceededError) as exc_info:
        await client.chat(
            [{"role": "user", "content": "x" * 80}],
            max_output_tokens=10,
        )

    assert dispatched is False
    assert exc_info.value.estimated_total_tokens > 20
    assert exc_info.value.limit == 20


def test_llm_usage_quota_defaults_to_unbounded() -> None:
    quota = AppSettings().llm_usage_quota

    assert quota.max_calls_per_run is None
    assert quota.max_input_tokens_per_run is None
    assert quota.max_output_tokens_per_run is None
    assert quota.max_total_tokens_per_run is None
