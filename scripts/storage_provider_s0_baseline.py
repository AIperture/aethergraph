"""Capture the pre-provider local-storage performance baseline.

This script intentionally exercises the current public/local implementations. It is
not a target-provider benchmark and must remain runnable until the migration's final
baseline comparison is recorded.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
from time import perf_counter, time
import tracemalloc
from typing import Any

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.storage.event_log import StateSnapshotConflictError
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.observability.models import (
    LLMObservationRecord,
    ObservationFilter,
    ObservationRecord,
    ObservationScope,
)
from aethergraph.observability.sqlite_store import SQLiteObservationStore
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex
from aethergraph.storage.continuation_store.kvdoc_cont import KVDocContinuationStore
from aethergraph.storage.docstore.sqlite_doc import SqliteDocStore
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.kv.sqlite_kv import SqliteKV
from aethergraph.storage.lexical_index.sqlite_lexical_index import SQLiteLexicalIndex
from aethergraph.storage.runs.sqlite_run_store import SQLiteRunStore
from aethergraph.storage.search_backend.generic_backend import GenericSearchBackend
from aethergraph.storage.vector_index.sqlite_index import SQLiteVectorIndex

AsyncOperation = Callable[[], Awaitable[Any]]


class DeterministicEmbedder:
    """Small dependency-free embedder used only for repeatable search timings."""

    async def embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [(digest[index] - 127.5) / 127.5 for index in range(16)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _distribution(values: list[float], *, elapsed_s: float, operations: int) -> dict[str, Any]:
    return {
        "operations": operations,
        "elapsed_s": round(elapsed_s, 6),
        "throughput_ops_s": round(operations / elapsed_s, 3) if elapsed_s else None,
        "latency_ms": {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": round(max(values), 3) if values else None,
        },
    }


async def _measure(operation: AsyncOperation, count: int) -> tuple[list[float], Counter[str], float]:
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    started = perf_counter()
    for _ in range(count):
        item_started = perf_counter()
        try:
            await operation()
        except Exception as exc:  # baseline records current failure behavior
            errors[type(exc).__name__] += 1
        else:
            latencies.append((perf_counter() - item_started) * 1_000)
    return latencies, errors, perf_counter() - started


def _database_sizes(root: Path) -> dict[str, int]:
    sizes: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = "wal" if path.name.endswith("-wal") else "shm" if path.name.endswith("-shm") else "data"
        sizes[suffix] += path.stat().st_size
    return dict(sizes)


async def _memory_baseline(root: Path, samples: int) -> dict[str, Any]:
    path = root / "memory" / "events.db"
    handles = [SqliteEventLog(str(path)) for _ in range(6)]
    latencies: dict[str, list[float]] = {"append": [], "recent_read": []}
    errors: Counter[str] = Counter()

    async def writer(worker: int) -> None:
        for index in range(samples):
            started = perf_counter()
            try:
                await handles[worker].append(
                    {
                        "id": f"memory-{worker}-{index}",
                        "scope_id": "session-baseline",
                        "session_id": "session-baseline",
                        "run_id": f"run-{worker}",
                        "kind": "memory.event",
                        "tags": ["baseline", "memory"],
                        "data": {"text": f"event {worker}-{index}"},
                        "ts": time(),
                    }
                )
            except Exception as exc:
                errors[f"append:{type(exc).__name__}"] += 1
            else:
                latencies["append"].append((perf_counter() - started) * 1_000)

    async def reader(worker: int) -> None:
        handle = handles[4 + worker]
        for _ in range(samples):
            started = perf_counter()
            try:
                await handle.query(scope_id="session-baseline", limit=25, order_dir="desc")
            except Exception as exc:
                errors[f"read:{type(exc).__name__}"] += 1
            else:
                latencies["recent_read"].append((perf_counter() - started) * 1_000)

    started = perf_counter()
    await asyncio.gather(*(writer(index) for index in range(4)), reader(0), reader(1))
    elapsed = perf_counter() - started
    for handle in handles:
        await handle.close()
    total = sum(len(values) for values in latencies.values())
    return {
        "workload": {"writers": 4, "readers": 2, "operations_per_worker": samples},
        "combined": _distribution(
            [value for values in latencies.values() for value in values],
            elapsed_s=elapsed,
            operations=total,
        ),
        "append": _distribution(latencies["append"], elapsed_s=elapsed, operations=len(latencies["append"])),
        "recent_read": _distribution(
            latencies["recent_read"], elapsed_s=elapsed, operations=len(latencies["recent_read"])
        ),
        "errors": dict(errors),
        "lock_wait_note": "Current store exposes no busy-wait duration; lock errors and tail latency are the observable proxies.",
    }


async def _state_cas_baseline(root: Path, contenders: int) -> dict[str, Any]:
    path = root / "state" / "events.db"
    handles = [SqliteEventLog(str(path)) for _ in range(contenders)]
    latencies: list[float] = []
    outcomes: Counter[str] = Counter()

    async def contend(index: int) -> None:
        started = perf_counter()
        event = {
            "id": f"state-{index}",
            "scope_id": "session-cas",
            "kind": "state.snapshot",
            "tags": ["state", "state:agent:baseline"],
            "data": {"value": {"writer": index}, "meta": {"revision": 1}},
            "ts": time(),
        }
        try:
            await handles[index].append_state_snapshot_if_revision(
                event,
                state_key="agent:baseline",
                expected_revision=0,
            )
        except StateSnapshotConflictError:
            outcomes["conflict"] += 1
        except Exception as exc:
            outcomes[type(exc).__name__] += 1
        else:
            outcomes["winner"] += 1
        finally:
            latencies.append((perf_counter() - started) * 1_000)

    started = perf_counter()
    await asyncio.gather(*(contend(index) for index in range(contenders)))
    elapsed = perf_counter() - started
    for handle in handles:
        await handle.close()
    return {
        **_distribution(latencies, elapsed_s=elapsed, operations=contenders),
        "contenders": contenders,
        "outcomes": dict(outcomes),
        "correctness": outcomes["winner"] == 1 and outcomes["conflict"] == contenders - 1,
    }


async def _continuation_baseline(root: Path, samples: int) -> dict[str, Any]:
    store = KVDocContinuationStore(
        doc_store=SqliteDocStore(str(root / "continuations" / "docs.db")),
        kv=SqliteKV(str(root / "continuations" / "tokens.db")),
        secret=b"s0-baseline-secret",
    )
    continuations = [
        Continuation(
            run_id=f"run-{index % 20}",
            node_id=f"node-{index}",
            kind="approval",
            token=f"token-{index}",
            channel="ui:session",
        )
        for index in range(samples)
    ]
    write_index = 0

    async def save_next() -> None:
        nonlocal write_index
        item = continuations[write_index]
        write_index += 1
        await store.save(item)

    read_index = 0

    async def resolve_next() -> None:
        nonlocal read_index
        item = await store.get_by_token(continuations[read_index].token)
        if item is None:
            raise AssertionError("continuation token did not resolve")
        read_index += 1

    writes, write_errors, write_elapsed = await _measure(save_next, samples)
    reads, read_errors, read_elapsed = await _measure(resolve_next, samples)
    return {
        "save": {**_distribution(writes, elapsed_s=write_elapsed, operations=len(writes)), "errors": dict(write_errors)},
        "resolve_by_token": {**_distribution(reads, elapsed_s=read_elapsed, operations=len(reads)), "errors": dict(read_errors)},
        "atomicity_note": "save currently commits document and token index in separate databases/operations.",
    }


async def _run_baseline(root: Path, samples: int) -> dict[str, Any]:
    store = SQLiteRunStore(str(root / "runs" / "runs.db"))
    records = [
        RunRecord(
            run_id=f"run-{index}",
            graph_id=f"graph-{index % 4}",
            kind="taskgraph",
            status=RunStatus.running,
            started_at=datetime.now(UTC),
            user_id=f"user-{index % 8}",
            org_id="org-baseline",
            session_id=f"session-{index % 16}",
        )
        for index in range(samples)
    ]
    create_index = 0

    async def create_next() -> None:
        nonlocal create_index
        item = records[create_index]
        create_index += 1
        await store.create(item)

    poll_index = 0

    async def poll_next() -> None:
        nonlocal poll_index
        await store.get(records[poll_index % samples].run_id)
        poll_index += 1

    list_index = 0

    async def list_next() -> None:
        nonlocal list_index
        await store.list(graph_id=f"graph-{list_index % 4}", limit=25, offset=(list_index % 4) * 10)
        list_index += 1

    update_index = 0

    async def update_next() -> None:
        nonlocal update_index
        await store.update_status(records[update_index].run_id, RunStatus.succeeded)
        update_index += 1

    result: dict[str, Any] = {}
    for name, operation, count in (
        ("create", create_next, samples),
        ("status_poll", poll_next, samples * 2),
        ("filtered_list", list_next, samples),
        ("status_update", update_next, samples),
    ):
        values, errors, elapsed = await _measure(operation, count)
        result[name] = {**_distribution(values, elapsed_s=elapsed, operations=len(values)), "errors": dict(errors)}
    return result


async def _observability_baseline(root: Path, samples: int) -> dict[str, Any]:
    store = SQLiteObservationStore(root / "observability" / "observability.db")
    for index in range(samples):
        await store.append_observation(
            ObservationRecord(
                category="log",
                name="baseline.log",
                summary=f"log {index}",
                scope=ObservationScope(
                    tenant_id="tenant-baseline",
                    project_id="project-baseline",
                    session_id=f"session-{index % 8}",
                    run_id=f"run-{index % 20}",
                    trace_id=f"trace-{index % 10}",
                ),
            )
        )
        await store.append_llm_call(
            LLMObservationRecord(
                llm_call_id=f"llm-{index}",
                created_at=datetime.now(UTC).isoformat(),
                call_type="chat",
                provider="baseline",
                model="deterministic",
                scope=ObservationScope(
                    tenant_id="tenant-baseline",
                    project_id="project-baseline",
                    run_id=f"run-{index % 20}",
                    trace_id=f"trace-{index % 10}",
                ),
                messages=[],
                reasoning_effort=None,
                max_output_tokens=None,
                output_format="text",
                json_schema=None,
                schema_name=None,
                strict_schema=None,
                validate_json=None,
                extra_params={},
                request_args={},
                provider_request_args={},
                compatibility_notes=[],
                trace_payload=None,
            )
        )

    offsets = iter((index % 8) * 25 for index in range(samples))

    async def trace_page() -> None:
        offset = next(offsets)
        await store.list_observations(ObservationFilter(trace_id="trace-1", limit=25), offset=offset)

    log_offsets = iter((index % 8) * 25 for index in range(samples))

    async def log_page() -> None:
        await store.list_observations(
            ObservationFilter(category="log", limit=25), offset=next(log_offsets)
        )

    llm_offsets = iter((index % 8) * 25 for index in range(samples))

    async def llm_page() -> None:
        await store.query_llm_calls(limit=25, offset=next(llm_offsets))

    result: dict[str, Any] = {}
    for name, operation in (("trace_page", trace_page), ("log_page", log_page), ("llm_page", llm_page)):
        values, errors, elapsed = await _measure(operation, samples)
        result[name] = {**_distribution(values, elapsed_s=elapsed, operations=len(values)), "errors": dict(errors)}
    stats = await store.get_storage_stats()
    result["store_reported_bytes"] = {
        "database": stats.database_bytes,
        "wal": stats.wal_bytes,
        "physical": stats.physical_bytes,
    }
    await store.close()
    return result


async def _artifact_baseline(root: Path, samples: int) -> dict[str, Any]:
    index = SqliteArtifactIndex(str(root / "artifacts" / "index.db"))
    artifacts = [
        Artifact(
            artifact_id=f"sha256-{index % 50}",
            run_id=f"run-{index % 10}",
            graph_id="graph-baseline",
            node_id=f"node-{index}",
            tool_name="baseline",
            tool_version="1",
            kind="text",
            sha256=f"sha256-{index % 50}",
            bytes=128,
            mime="text/plain",
            created_at=datetime.now(UTC).isoformat(),
            labels={"filename": f"artifact-{index}.txt"},
            uri=f"cas://sha256-{index % 50}",
            session_id=f"session-{index % 5}",
            occurrence_id=f"occurrence-{index}",
        )
        for index in range(samples)
    ]
    for artifact in artifacts:
        await index.upsert(artifact)
        await index.record_occurrence(artifact)
    page_index = 0

    async def list_page() -> None:
        nonlocal page_index
        await index.list_occurrences_for_run(
            f"run-{page_index % 10}", limit=20, offset=(page_index % 4) * 10
        )
        page_index += 1

    values, errors, elapsed = await _measure(list_page, samples)
    return {**_distribution(values, elapsed_s=elapsed, operations=len(values)), "errors": dict(errors)}


async def _search_baseline(root: Path, samples: int) -> dict[str, Any]:
    backend = GenericSearchBackend(
        index=SQLiteVectorIndex(str(root / "search" / "vector"), use_faiss_if_available=False),
        lexical=SQLiteLexicalIndex(str(root / "search" / "lexical")),
        embedder=DeterministicEmbedder(),
        debug=False,
    )
    write_index = 0

    async def index_next() -> None:
        nonlocal write_index
        index = write_index
        write_index += 1
        await backend.upsert(
            corpus="baseline",
            item_id=f"item-{index}",
            text=f"storage provider baseline document number {index}",
            metadata={
                "tenant_id": "tenant-baseline",
                "project_id": "project-baseline",
                "scope_id": f"scope-{index % 4}",
                "run_id": f"run-{index % 20}",
                "kind": "document",
                "created_at_ts": time() + index,
            },
        )

    values, errors, elapsed = await _measure(index_next, samples)
    result: dict[str, Any] = {
        "index": {**_distribution(values, elapsed_s=elapsed, operations=len(values)), "errors": dict(errors)}
    }
    query_count = max(25, samples // 4)
    for mode in ("semantic", "lexical", "hybrid"):
        async def search(current_mode: str = mode) -> None:
            await backend.search(
                corpus="baseline",
                query="storage provider document",
                top_k=10,
                filters={"scope_id": "scope-1"},
                mode=current_mode,
            )

        values, errors, elapsed = await _measure(search, query_count)
        result[mode] = {**_distribution(values, elapsed_s=elapsed, operations=len(values)), "errors": dict(errors)}
    return result


def _query_plans(root: Path) -> dict[str, list[str]]:
    statements = {
        "memory_recent": (root / "memory" / "events.db", "SELECT * FROM events WHERE scope_id = 'session-baseline' ORDER BY id DESC LIMIT 25"),
        "state_latest": (root / "state" / "events.db", "SELECT e.payload FROM events e JOIN event_tags t ON t.event_row_id=e.id AND t.tag='state:agent:baseline' WHERE e.scope_id='session-cas' AND e.kind='state.snapshot' ORDER BY e.id DESC LIMIT 1"),
        "run_poll": (root / "runs" / "runs.db", "SELECT data_json FROM runs WHERE run_id='run-1'"),
        "run_graph_page": (root / "runs" / "runs.db", "SELECT data_json FROM runs WHERE graph_id='graph-1' ORDER BY started_at DESC LIMIT 25 OFFSET 100"),
        "observation_trace_page": (root / "observability" / "observability.db", "SELECT * FROM observations WHERE trace_id='trace-1' ORDER BY occurred_at DESC LIMIT 25 OFFSET 100"),
        "artifact_occurrence_page": (root / "artifacts" / "index.db", "SELECT * FROM artifact_occurrences WHERE run_id='run-1' ORDER BY created_at DESC, id DESC LIMIT 20 OFFSET 40"),
        "vector_scope": (root / "search" / "vector" / "index.sqlite", "SELECT * FROM embeddings WHERE corpus_id='baseline' AND scope_id='scope-1' ORDER BY created_at_ts DESC LIMIT 30"),
    }
    result: dict[str, list[str]] = {}
    for name, (path, sql) in statements.items():
        with sqlite3.connect(path) as connection:
            result[name] = [str(row[3]) for row in connection.execute(f"EXPLAIN QUERY PLAN {sql}")]
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="ag-storage-s0-"))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit(f"baseline workspace must be empty: {root}")
    tracemalloc.start()
    results = {
        "memory": await _memory_baseline(root, args.samples),
        "state_cas": await _state_cas_baseline(root, args.contenders),
        "continuations": await _continuation_baseline(root, args.samples),
        "runs": await _run_baseline(root, args.samples),
        "observability": await _observability_baseline(root, args.samples),
        "artifacts": await _artifact_baseline(root, args.samples),
        "search": await _search_baseline(root, args.samples),
    }
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "baseline": "pre-storage-provider-local",
        "captured_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pid": os.getpid(),
        },
        "parameters": {"samples": args.samples, "cas_contenders": args.contenders},
        "workspace": str(root),
        "results": results,
        "query_plans": _query_plans(root),
        "storage_bytes": _database_sizes(root),
        "python_peak_traced_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--contenders", type=int, default=8)
    parser.add_argument("--workspace", help="Empty directory to retain the generated baseline data")
    args = parser.parse_args()
    if args.samples < 25:
        parser.error("--samples must be at least 25")
    if args.contenders < 2:
        parser.error("--contenders must be at least 2")
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
