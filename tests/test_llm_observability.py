from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

import pytest

from aethergraph.config.config import AppSettings, LLMUsageQuotaSettings
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.observability import (
    LLMObservationRecord,
    ObservabilityFacade,
    ObservationPolicy,
    SQLiteObservationStore,
)
from aethergraph.observability.redaction import canonical_json
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.llm.correlation import current_llm_call_correlation
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.observability import ConsoleLLMObservationSink
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateLimitSnapshot,
    ProviderResponseMetadata,
    ProviderRetrySettings,
    ProviderTransportAttempt,
)
from aethergraph.services.llm.types import (
    LLMContextWindowExceededError,
    LLMRunQuotaExceededError,
    LLMRunQuotaWouldExceedError,
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

    await sink.emit(record, capture_mode="full")

    out = capsys.readouterr().out
    assert "LLM CALL  [-] openai/gpt-test  profile=default" in out
    assert "[USER]" in out
    assert "[OUTPUT]" in out
    assert "tokens:  in=3  out=2  total=5" in out


@pytest.mark.asyncio
async def test_manifest_reconstructs_exact_provider_request_and_deduplicates_fragments(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    first = _record(run_id="run-1")
    first.attempts = (
        ProviderTransportAttempt(
            attempt_number=1,
            elapsed_s=0.01,
            outcome="success",
            retryable=False,
        ),
    )
    second = _record(run_id="run-2")

    await store.append_llm_call(first)
    first_stats = await store.get_storage_stats()
    await store.append_llm_call(second)
    second_stats = await store.get_storage_stats()
    detail = await store.get_llm_call(first.llm_call_id)

    assert detail is not None
    manifest = detail["prompt_manifest"]
    request = manifest["provider_request"]
    assert (
        sha256(canonical_json(request).encode()).hexdigest() == manifest["assembled_request_hash"]
    )
    assert request["messages"] == [{"content": "hello", "role": "user"}]
    assert second_stats.fragments == first_stats.fragments


@pytest.mark.asyncio
async def test_full_capture_stores_one_prompt_body_without_duplicate_preview(
    tmp_path: Path,
) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="full"),
    )
    record = _record(prompt="unique-secret-prompt")

    await store.append_llm_call(record)
    detail = await store.get_llm_call(record.llm_call_id)

    assert detail is not None
    assert detail["messages"] == [{"content": "unique-secret-prompt", "role": "user"}]
    assert "preview" not in detail["messages_preview"]
    with store._connect() as conn:
        bodies = [row[0] for row in conn.execute("SELECT body FROM content_fragments")]
    assert sum(body.count("unique-secret-prompt") for body in bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "metadata"])
async def test_non_payload_capture_modes_store_no_content_fragments(
    tmp_path: Path, mode: str
) -> None:
    store = SQLiteObservationStore(
        tmp_path / f"{mode}.db",
        policy=ObservationPolicy(capture_mode=mode),  # type: ignore[arg-type]
    )
    record = _record(prompt="secret prompt", output="secret answer")

    await store.append_llm_call(record)
    stats = await store.get_storage_stats()
    detail = await store.get_llm_call(record.llm_call_id)

    assert stats.fragments == 0
    assert detail is not None
    assert detail["messages"] is None
    assert detail["raw_text"] is None
    if mode == "off":
        assert detail["messages_preview"] is None
        assert stats.manifests == 0
    else:
        assert detail["messages_preview"]["count"] == 1
        assert stats.manifests == 1


@pytest.mark.asyncio
async def test_deletion_retains_shared_fragments_until_final_reference(tmp_path: Path) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    first = _record(run_id="run-1")
    second = _record(run_id="run-2")
    await store.append_llm_call(first)
    await store.append_llm_call(second)
    before = await store.get_storage_stats()

    preview = await store.delete_run_observations("run-1", dry_run=True)
    deleted_first = await store.delete_run_observations("run-1")
    middle = await store.get_storage_stats()
    deleted_second = await store.delete_run_observations("run-2")
    after = await store.get_storage_stats()

    assert preview.shared_fragment_bytes_retained > 0
    assert deleted_first.deleted_observations == 1
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_call_attempts").fetchone()[0] == 0
    assert middle.fragments == before.fragments
    assert deleted_second.deleted_fragments == before.fragments
    assert after.fragments == 0


@pytest.mark.asyncio
async def test_read_only_store_hydrates_attempts_after_schema_creation(tmp_path: Path) -> None:
    path = tmp_path / "observability.db"
    writable = SQLiteObservationStore(
        path,
        policy=ObservationPolicy(capture_mode="metadata"),
    )
    record = _record(run_id="run-read-only")
    record.attempts = (
        ProviderTransportAttempt(
            attempt_number=1,
            elapsed_s=0.02,
            outcome="success",
            retryable=False,
            request_id="req-read-only",
        ),
    )
    await writable.append_llm_call(record)

    read_only = SQLiteObservationStore(path, read_only=True)
    detail = await read_only.get_llm_call(record.llm_call_id)

    assert detail is not None
    assert detail["attempt_count"] == 1
    assert detail["attempts"][0]["request_id"] == "req-read-only"


@pytest.mark.asyncio
async def test_read_only_store_projects_empty_attempts_for_older_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observability.db"
    writable = SQLiteObservationStore(
        path,
        policy=ObservationPolicy(capture_mode="metadata"),
    )
    record = _record(run_id="run-before-attempt-schema")
    await writable.append_llm_call(record)
    with writable._connect() as conn:
        conn.execute("DROP TABLE llm_call_attempts")

    read_only = SQLiteObservationStore(path, read_only=True)
    listed = await read_only.query_llm_calls(run_id=record.scope.run_id)
    detail = await read_only.get_llm_call(record.llm_call_id)

    assert listed[0]["attempt_count"] == 0
    assert listed[0]["retry_count"] == 0
    assert listed[0]["total_retry_wait_ms"] == 0
    assert detail is not None
    assert detail["attempts"] == []


@pytest.mark.asyncio
async def test_concurrent_writes_and_historical_reads_are_safe(tmp_path: Path) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    records = [_record(run_id=f"run-{index}") for index in range(20)]

    await asyncio.gather(
        *(store.append_llm_call(record) for record in records),
        *(store.query_llm_calls(limit=None) for _ in range(5)),
    )

    assert len(await store.query_llm_calls(limit=None)) == 20


@pytest.mark.asyncio
async def test_llm_client_records_success_and_provider_error_in_sqlite(tmp_path: Path) -> None:
    store = SQLiteObservationStore(
        tmp_path / "observability.db",
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    sink = ObservabilityFacade(store)
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

    rows = await store.query_llm_calls(limit=None)
    assert len(rows) == 2
    assert {row["error_type"] for row in rows} == {None, "RuntimeError"}
    assert len({row["llm_call_id"] for row in rows}) == 2
    recovered = next(row for row in rows if row["error_type"] is None)
    assert recovered["attempt_count"] == 2
    assert recovered["retry_count"] == 1
    assert recovered["total_retry_wait_ms"] == 598
    detail = await store.get_llm_call(recovered["llm_call_id"])
    assert detail is not None
    assert detail["attempts"][0]["error_code"] == "provider_rate_limited"
    assert detail["attempts"][0]["rate_limits"] == [
        {
            "resource": "tokens",
            "limit": 200000,
            "remaining": 34571,
            "reset_after_s": 0.598,
        }
    ]
    assert detail["attempts"][1]["request_id"] == "req-success"
    inspect_page = await sink.list_inspect_llm_calls(run_id="run-success")
    assert inspect_page.items[0].attempt_count == 2
    assert inspect_page.items[0].retry_count == 1
    assert inspect_page.items[0].attempts == []
    inspect_detail = await sink.get_inspect_llm_call(recovered["llm_call_id"])
    assert len(inspect_detail.attempts) == 2
    assert inspect_detail.attempts[0].rate_limits[0].resource == "tokens"
    correlation = current_llm_call_correlation()
    assert correlation is not None
    assert correlation.llm_call_id in {row["llm_call_id"] for row in rows}


@pytest.mark.asyncio
async def test_observation_sink_failure_cannot_change_a_successful_llm_result() -> None:
    class FailingSink:
        async def emit(self, record, *, capture_mode: str) -> None:
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
async def test_metering_is_independent_of_capture_mode(tmp_path: Path, mode: str) -> None:
    class FakeMetering:
        def __init__(self) -> None:
            self.records = []

        async def record_llm(self, **record) -> None:
            self.records.append(record)

    metering = FakeMetering()
    store = SQLiteObservationStore(
        tmp_path / f"{mode}.db",
        policy=ObservationPolicy(capture_mode=mode),  # type: ignore[arg-type]
    )
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        metering=metering,
        observation_sink=ObservabilityFacade(store),
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
async def test_default_container_uses_sqlite_and_no_jsonl_or_persisted_tracer(
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

    assert (tmp_path / "events" / "observability.db").is_file()
    assert not list(tmp_path.rglob("llm_calls.jsonl"))
    assert container.tracer.__class__.__name__ == "NoopTracer"
    rows = await container.observability.list_llm_calls(limit=None)
    assert rows[0]["capture_mode"] == "manifest"


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
