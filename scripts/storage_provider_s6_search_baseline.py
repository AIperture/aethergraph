"""Benchmark canonical indexing and every exact local search mode."""

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
import sqlite3
import tempfile
from time import perf_counter
import tracemalloc
from typing import Any

from aethergraph.storage.contracts import (
    SearchDocument,
    SearchMode,
    SearchQuery,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://benchmarks/s6"
_SECRET = b"s6-search-benchmark-secret-32bytes"
_START = datetime(2026, 8, 16, 18, tzinfo=UTC)
_TOKENS = ("canonical", "storage", "provider", "migration", "search", "document")


class _Clock:
    def __init__(self) -> None:
        self.value = _START

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


class _Embedder:
    async def embed(self, texts, **_kwargs):
        return [
            [float(text.lower().split().count(token)) for token in _TOKENS] + [1.0]
            for text in texts
        ]


def _scope() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-baseline",
        project_id="project-baseline",
        org_id="org-baseline",
        user_id="user-baseline",
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
    return LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
        embedder=_Embedder(),
    ).open(
        StorageOpenRequest(
            workspace_id="s6-search-baseline",
            workspace_root=root.resolve(),
            owner_scope=_scope(),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={
                    "continuation_token_secret_ref": _SECRET_REF,
                    "search_max_candidates": 10_000,
                },
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=_Clock(),
            secrets=_Secrets(),
        )
    )


def _document(index: int) -> SearchDocument:
    selected = index % 2 == 0
    return SearchDocument(
        corpus="baseline",
        item_id=f"item-{index}",
        text=f"canonical storage provider migration search document {index}",
        scope=_scope(),
        occurred_at=_START + timedelta(microseconds=index),
        tags=("selected", "shared") if selected else ("other", "shared"),
        metadata={"kind": "note", "bucket": index % 5},
    )


async def _measure(operation, count: int) -> tuple[list[float], Counter[str], float]:
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    started = perf_counter()
    for _index in range(count):
        item_started = perf_counter()
        try:
            await operation()
        except Exception as exc:
            errors[type(exc).__name__] += 1
        else:
            latencies.append((perf_counter() - item_started) * 1_000)
    return latencies, errors, perf_counter() - started


async def _benchmark(bundle: Any, samples: int) -> dict[str, Any]:
    documents = tuple(_document(index) for index in range(samples))
    write_index = 0
    last_cursor: str | None = None

    async def index_next() -> None:
        nonlocal last_cursor, write_index
        last_cursor = await bundle.search.upsert(documents[write_index])
        write_index += 1

    latencies, errors, elapsed = await _measure(index_next, samples)
    result: dict[str, Any] = {
        "index": {
            **_distribution(latencies, elapsed_s=elapsed),
            "errors": dict(errors),
        }
    }
    if last_cursor is None:
        raise AssertionError("benchmark indexed no documents")
    covered_cursor = await bundle.search.wait_until_indexed("baseline", last_cursor, 0.0)
    query_count = max(25, samples // 4)
    query_correctness: dict[str, bool] = {}
    for mode in SearchMode:
        query = SearchQuery(
            corpus="baseline",
            mode=mode,
            scope=_scope(),
            query="canonical storage document" if mode is not SearchMode.STRUCTURAL else "",
            top_k=10,
            tags=("shared", "selected"),
            metadata={"kind": "note"},
            require_indexed_cursor=covered_cursor,
        )

        async def search(current_query: SearchQuery = query) -> None:
            rows = await bundle.search.query(current_query)
            correct = bool(rows) and all(
                row.mode is current_query.mode
                and row.item_id.startswith("item-")
                and int(row.item_id.removeprefix("item-")) % 2 == 0
                for row in rows
            )
            query_correctness[current_query.mode.value] = (
                query_correctness.get(current_query.mode.value, True) and correct
            )
            if not correct:
                raise AssertionError("exact-mode result crossed canonical tag filters")

        latencies, errors, elapsed = await _measure(search, query_count)
        result[mode.value] = {
            **_distribution(latencies, elapsed_s=elapsed),
            "errors": dict(errors),
        }
    result["workload"] = {
        "indexed_documents": samples,
        "query_operations_per_mode": query_count,
        "top_k": 10,
        "normalized_tag_intersection": ["selected", "shared"],
    }
    result["correctness"] = not result["index"]["errors"] and all(
        query_correctness.get(mode.value, False) for mode in SearchMode
    )
    return result


def _query_plans(root: Path) -> dict[str, list[str]]:
    path = root / "local" / "search.sqlite3"
    statements = {
        "scope_time": "SELECT * FROM local_search_documents WHERE corpus='baseline' AND scope_identity='{}' ORDER BY occurred_at DESC, document_id DESC LIMIT 10",
        "tag_intersection": "SELECT d.* FROM local_search_documents d WHERE d.corpus='baseline' AND d.scope_identity='{}' AND EXISTS (SELECT 1 FROM local_search_tags t WHERE t.document_id=d.document_id AND t.tag='selected') ORDER BY d.occurred_at DESC, d.document_id DESC LIMIT 10",
    }
    connection = sqlite3.connect(path)
    try:
        return {
            name: [str(row[3]) for row in connection.execute(f"EXPLAIN QUERY PLAN {sql}")]
            for name, sql in statements.items()
        }
    finally:
        connection.close()


async def _run(samples: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aethergraph-s6-search-baseline-") as temporary:
        root = Path(temporary)
        tracemalloc.start()
        bundle = _open_bundle(root)
        try:
            search = await _benchmark(bundle, samples)
            query_plans = _query_plans(root)
            _current, peak = tracemalloc.get_traced_memory()
            sizes_before_checkpoint = _database_sizes(root)
            maintenance = {
                role.value: asdict(detail) for role, detail in (await bundle.checkpoint()).items()
            }
            sizes_after_checkpoint = _database_sizes(root)
        finally:
            await bundle.close()
            tracemalloc.stop()
        return {
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "samples": samples,
            "search": search,
            "query_plans": query_plans,
            "peak_python_bytes": peak,
            "database_bytes_before_checkpoint": sizes_before_checkpoint,
            "database_bytes_after_checkpoint": sizes_after_checkpoint,
            "database_bytes_after_close": _database_sizes(root),
            "checkpoint": maintenance,
        }


def main() -> None:
    """Run and print the reproducible canonical search benchmark.

    The benchmark creates one manifested local provider, measures synchronous exact
    indexing plus structural/semantic/lexical/hybrid queries, verifies normalized tag
    containment, records query plans, checkpoints, and removes the workspace.

    Examples:
        Run the recorded comparison shape:
            ```python
            python scripts/storage_provider_s6_search_baseline.py --samples 100
            ```

        Run a short smoke benchmark:
            ```python
            python scripts/storage_provider_s6_search_baseline.py --samples 25
            ```

    Args:
        None.

    Returns:
        None: JSON correctness, latency, memory, storage, and query-plan evidence is
        written to standard output.

    Notes:
        Generated storage exists only inside a managed temporary directory. This
        script never activates production provider routing or a fallback backend.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 25:
        raise SystemExit("samples must be at least 25")
    print(json.dumps(asyncio.run(_run(args.samples)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
