from __future__ import annotations

import asyncio
import json

import pytest

from aethergraph.services.inspect import open_inspection_facade
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
        llm_path = tmp_path / "events" / "llm" / "llm_calls.jsonl"
        llm_path.parent.mkdir(parents=True)
        llm_path.write_text(
            json.dumps(
                {
                    "call_id": "call-1",
                    "created_at": "2026-07-15T00:00:00+00:00",
                    "call_type": "chat",
                    "provider": "openai",
                    "model": "gpt-test",
                    "run_id": "run-1",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
            + "\n"
            + '{"call_id":"incomplete"',
            encoding="utf-8",
        )

        facade = open_inspection_facade(tmp_path)
        assert [item.id for item in (await facade.list_traces(run_id="run-1")).items] == ["trace-1"]
        assert [item.call_id for item in (await facade.list_llm_calls(run_id="run-1")).items] == [
            "call-1"
        ]
        assert (await facade.get_llm_call("call-1", required_run_id="run-1")).messages == [
            {"role": "user", "content": "hello"}
        ]

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
