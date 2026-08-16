"""Benchmark canonical provider-backed memory and Agent-state paths."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import platform
import tempfile
from time import perf_counter
import tracemalloc
from typing import Any

from aethergraph.services.agent_state import (
    AgentStateConflictError,
    CanonicalAgentStateFacade,
)
from aethergraph.services.memory import CanonicalMemoryFacade
from aethergraph.storage.contracts import (
    EventQuery,
    PageRequest,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://benchmarks/s4"
_SECRET = b"s4-state-memory-benchmark-secret-32b"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 10, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


def _scope() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-baseline",
        project_id="project-baseline",
        org_id="org-baseline",
        user_id="user-baseline",
        session_id="session-baseline",
        run_id="run-baseline",
        graph_id="graph-baseline",
        agent_id="agent-baseline",
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _distribution(values: list[float], *, elapsed_s: float) -> dict[str, Any]:
    return {
        "operations": len(values),
        "elapsed_s": round(elapsed_s, 6),
        "throughput_ops_s": round(len(values) / elapsed_s, 3) if elapsed_s else None,
        "latency_ms": {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": round(max(values), 3) if values else None,
        },
    }


def _database_sizes(root: Path) -> dict[str, int]:
    sizes: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        category = (
            "wal" if path.name.endswith("-wal") else "shm" if path.name.endswith("-shm") else "data"
        )
        sizes[category] += path.stat().st_size
    return dict(sizes)


def _open_bundle(root: Path):
    provider = LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    )
    return provider.open(
        StorageOpenRequest(
            workspace_id="s4-state-memory-baseline",
            workspace_root=root.resolve(),
            owner_scope=StorageScope(
                tenant_id="tenant-baseline",
                project_id="project-baseline",
            ),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=_Clock(),
            secrets=_Secrets(),
        )
    )


async def _memory_benchmark(bundle, samples: int) -> dict[str, Any]:
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
    )
    latencies: dict[str, list[float]] = {"append": [], "recent_read": []}
    errors: Counter[str] = Counter()

    async def writer(worker: int) -> None:
        for index in range(samples):
            started = perf_counter()
            try:
                await memory.append_event(
                    event_id=f"memory-{worker}-{index}",
                    occurred_at=datetime.now(UTC),
                    kind="memory.event",
                    text=f"event {worker}-{index}",
                    tags=("baseline", "memory"),
                    payload={"writer": worker, "index": index},
                )
            except Exception as exc:
                errors[f"append:{type(exc).__name__}"] += 1
            else:
                latencies["append"].append((perf_counter() - started) * 1_000)

    async def reader() -> None:
        for _index in range(samples):
            started = perf_counter()
            try:
                await memory.durable_query(EventQuery(scope=_scope(), page=PageRequest(limit=25)))
            except Exception as exc:
                errors[f"read:{type(exc).__name__}"] += 1
            else:
                latencies["recent_read"].append((perf_counter() - started) * 1_000)

    started = perf_counter()
    await asyncio.gather(*(writer(index) for index in range(4)), reader(), reader())
    elapsed = perf_counter() - started

    hot_latencies: list[float] = []
    hot_started = perf_counter()
    for _index in range(samples * 2):
        started = perf_counter()
        await memory.recent_hot(limit=25)
        hot_latencies.append((perf_counter() - started) * 1_000)
    hot_elapsed = perf_counter() - hot_started
    return {
        "workload": {"writers": 4, "readers": 2, "operations_per_worker": samples},
        "append": _distribution(latencies["append"], elapsed_s=elapsed),
        "recent_read": _distribution(latencies["recent_read"], elapsed_s=elapsed),
        "hot_read": _distribution(hot_latencies, elapsed_s=hot_elapsed),
        "errors": dict(errors),
    }


async def _state_benchmark(bundle, contenders: int) -> dict[str, Any]:
    handles = [
        CanonicalAgentStateFacade(state_store=bundle.state, scope=_scope()).bind(
            key="baseline",
            backend="memory",
            level="session",
        )
        for _index in range(contenders)
    ]
    await asyncio.gather(*(handle.load() for handle in handles))
    latencies: list[float] = []
    outcomes: Counter[str] = Counter()

    async def contend(index: int) -> None:
        started = perf_counter()
        try:
            await handles[index].commit({"writer": index}, expected_revision=0)
        except AgentStateConflictError:
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
    return {
        **_distribution(latencies, elapsed_s=elapsed),
        "contenders": contenders,
        "outcomes": dict(outcomes),
        "correctness": outcomes["winner"] == 1 and outcomes["conflict"] == contenders - 1,
    }


async def _run(samples: int, contenders: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aethergraph-s4-baseline-") as temporary:
        root = Path(temporary)
        tracemalloc.start()
        bundle = _open_bundle(root)
        try:
            memory = await _memory_benchmark(bundle, samples)
            state = await _state_benchmark(bundle, contenders)
            _current, peak = tracemalloc.get_traced_memory()
            sizes_before_checkpoint = _database_sizes(root)
            maintenance = {
                role.value: asdict(detail) for role, detail in (await bundle.checkpoint()).items()
            }
            sizes_after_checkpoint = _database_sizes(root)
        finally:
            await bundle.close()
            tracemalloc.stop()
        sizes_after_close = _database_sizes(root)
        return {
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "samples": samples,
            "contenders": contenders,
            "memory": memory,
            "state_cas": state,
            "peak_python_bytes": peak,
            "database_bytes_before_checkpoint": sizes_before_checkpoint,
            "database_bytes_after_checkpoint": sizes_after_checkpoint,
            "database_bytes_after_close": sizes_after_close,
            "checkpoint": maintenance,
        }


def main() -> None:
    """Run and print the reproducible canonical state/memory benchmark.

    Parses bounded workload sizes, creates a temporary manifested provider workspace,
    records latency/correctness/resource evidence, checkpoints it, and removes it.

    Examples:
        Run the recorded comparison shape:
            ```python
            python scripts/storage_provider_s4_state_memory_baseline.py --samples 100 --contenders 8
            ```

        Run a shorter smoke benchmark:
            ```python
            python scripts/storage_provider_s4_state_memory_baseline.py --samples 10 --contenders 4
            ```

    Args:
        None.

    Returns:
        None: JSON benchmark evidence is written to standard output.

    Notes:
        Generated storage exists only inside a managed temporary directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--contenders", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.contenders < 2:
        raise SystemExit("samples must be positive and contenders must be at least two")
    print(json.dumps(asyncio.run(_run(args.samples, args.contenders)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
