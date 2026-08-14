from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.observability import (
    LLMObservationRecord,
    ObservabilityFacade,
    ObservationPolicy,
    ObservationRecord,
    ObservationScope,
    SQLiteObservationStore,
    open_observability_facade,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.runs.sqlite_run_store import SQLiteRunStore


def test_workspace_facade_reads_active_and_historical_records(tmp_path) -> None:
    async def exercise() -> None:
        event_path = tmp_path / "events" / "events.db"
        writer = SqliteEventLog(str(event_path))
        engine_writer = SqliteEventLog(str(tmp_path / "memory_events" / "events.db"))
        run_store = SQLiteRunStore(str(tmp_path / "runs" / "runs.db"))
        await run_store.create(
            RunRecord(
                run_id="run-1",
                graph_id="graph-1",
                kind="graphfn",
                status=RunStatus.succeeded,
                started_at=datetime(2026, 3, 11, tzinfo=UTC),
                session_id="session-1",
            )
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
        await observation_store.append_observation(
            ObservationRecord(
                observation_id="trace-1",
                category="service_operation",
                name="submit",
                summary="runner/submit finished",
                scope=ObservationScope(run_id="run-1", trace_id="operational-trace-1"),
                attributes={
                    "service": "runner",
                    "operation": "submit",
                    "phase": "finish",
                },
            )
        )

        facade = open_observability_facade(tmp_path)
        assert [item.id for item in (await facade.list_inspect_traces(run_id="run-1")).items] == [
            "trace-1"
        ]
        assert [
            item.call_id for item in (await facade.list_inspect_llm_calls(run_id="run-1")).items
        ] == [record.llm_call_id]
        assert (
            await facade.get_inspect_llm_call(record.llm_call_id, required_run_id="run-1")
        ).messages == [{"content": "hello", "role": "user"}]

        await observation_store.append_observation(
            ObservationRecord(
                observation_id="trace-2",
                category="service_operation",
                name="submit",
                summary="runner/submit finished again",
                scope=ObservationScope(run_id="run-1", trace_id="operational-trace-1"),
                attributes={"service": "runner", "operation": "submit"},
            )
        )
        assert [item.id for item in (await facade.list_inspect_traces(run_id="run-1")).items] == [
            "trace-2",
            "trace-1",
        ]
        with pytest.raises(RuntimeError, match="read-only"):
            await facade.event_log.append({"kind": "trace"})
        await facade.close()
        await facade.close()
        await observation_store.close()
        await writer.close()
        await engine_writer.close()

    asyncio.run(exercise())


def test_public_reader_methods_hide_observability_store_implementations() -> None:
    class Store:
        async def list_suppressed_scopes(self) -> dict[str, set[str]]:
            return {"run_id": {"hidden"}, "trace_id": set(), "session_id": set()}

        async def hydrate_prompt_manifest(self, manifest_id: str):
            return {"manifest_id": manifest_id}

    class Runs:
        async def list(self, *, limit: int, offset: int):
            assert (limit, offset) == (25, 5)
            return [{"run_id": "run-1"}]

        async def get(self, run_id: str):
            return {"run_id": run_id}

    class EngineEvents:
        async def query(self, **filters):
            assert filters == {
                "tags": ["agent_engine"],
                "run_id": "run-1",
                "limit": None,
                "order_dir": "asc",
            }
            return [{"id": "event-1"}]

    async def exercise() -> None:
        facade = ObservabilityFacade(
            Store(),  # type: ignore[arg-type]
            engine_event_log=EngineEvents(),
            run_store=Runs(),
            owns_store=False,
        )

        assert await facade.list_suppressed_scopes() == {
            "run_id": {"hidden"},
            "trace_id": set(),
            "session_id": set(),
        }
        assert await facade.list_runs(limit=25, offset=5) == [{"run_id": "run-1"}]
        assert await facade.get_run("run-1") == {"run_id": "run-1"}
        assert await facade.list_engine_events(run_id="run-1") == [{"id": "event-1"}]
        assert await facade.hydrate_prompt_manifest("manifest-1") == {"manifest_id": "manifest-1"}

    asyncio.run(exercise())
