from __future__ import annotations

import asyncio

import pytest

from aethergraph.services.inspect import open_inspection_facade
from aethergraph.services.observability import (
    LLMObservationRecord,
    ObservationPolicy,
    SQLiteObservationStore,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog


def test_workspace_facade_reads_active_and_historical_records(tmp_path) -> None:
    async def exercise() -> None:
        event_path = tmp_path / "events" / "events.db"
        writer = SqliteEventLog(str(event_path))
        await writer.append(
            {
                "id": "trace-1",
                "ts": 1.0,
                "scope_id": "trace:run/run-1",
                "kind": "trace",
                "run_id": "run-1",
                "payload": {
                    "trace_id": "operational-trace-1",
                    "span_id": "span-1",
                    "service": "runner",
                    "operation": "submit",
                    "phase": "finish",
                    "status": "ok",
                    "run_id": "run-1",
                },
            }
        )
        observation_store = SQLiteObservationStore(
            tmp_path / "events" / "observability.db",
            policy=ObservationPolicy(capture_mode="full"),
        )
        record = LLMObservationRecord.new(
            call_type="chat",
            provider="openai",
            model="gpt-test",
            dimensions={"run_id": "run-1"},
            messages=[{"role": "user", "content": "hello"}],
            reasoning_effort=None,
            max_output_tokens=None,
            output_format="text",
            json_schema=None,
            schema_name="output",
            strict_schema=True,
            validate_json=True,
            extra_params={},
            request_args={},
            provider_request_args={},
            compatibility_notes=[],
            trace_payload=None,
        )
        await observation_store.append_llm_call(record)

        facade = open_inspection_facade(tmp_path)
        assert [item.id for item in (await facade.list_traces(run_id="run-1")).items] == ["trace-1"]
        assert [item.call_id for item in (await facade.list_llm_calls(run_id="run-1")).items] == [
            record.llm_call_id
        ]
        assert (
            await facade.get_llm_call(record.llm_call_id, required_run_id="run-1")
        ).messages == [{"content": "hello", "role": "user"}]

        await writer.append(
            {
                "id": "trace-2",
                "ts": 2.0,
                "scope_id": "trace:run/run-1",
                "kind": "trace",
                "run_id": "run-1",
                "payload": {
                    "trace_id": "operational-trace-1",
                    "span_id": "span-2",
                    "service": "runner",
                    "operation": "submit",
                    "phase": "finish",
                    "status": "ok",
                    "run_id": "run-1",
                },
            }
        )
        assert [item.id for item in (await facade.list_traces(run_id="run-1")).items] == [
            "trace-2",
            "trace-1",
        ]
        with pytest.raises(RuntimeError, match="read-only"):
            await facade.event_log.append({"kind": "trace"})
        await facade.close()
        await facade.close()
        await writer.close()

    asyncio.run(exercise())
