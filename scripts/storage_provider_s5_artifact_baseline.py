"""Benchmark canonical S5 occurrence queries and public artifact-page hydration."""

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

from aethergraph.services.artifacts import CanonicalArtifactFacade
from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactRecord,
    ArtifactRetentionRecord,
    PageRequest,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://benchmarks/s5"
_SECRET = b"s5-artifact-benchmark-secret-32b"
_START = datetime(2026, 8, 16, 14, tzinfo=UTC)


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


def _owner_scope() -> StorageScope:
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
    ).open(
        StorageOpenRequest(
            workspace_id="s5-artifact-baseline",
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


async def _populate(bundle: Any, samples: int) -> None:
    owner = _owner_scope()
    artifact_count = min(50, samples)
    for index in range(artifact_count):
        selected = index % 2 == 0
        artifact = ArtifactRecord(
            artifact_id=f"artifact-{index}",
            content_hash=f"{index:064x}",
            hash_algorithm="sha256",
            size_bytes=128,
            media_type="text/plain",
            kind="report" if selected else "draft",
            blob_locator=f"blob:sha256:{index:064x}",
            owner_scope=owner,
            created_at=_START + timedelta(microseconds=index),
            labels={
                "tags": ("final", "reviewed") if selected else ("draft",),
                "category": "evidence" if selected else "working",
            },
        )
        await bundle.artifacts.put(artifact)
        if selected:
            await bundle.artifacts.compare_and_set_retention(
                ArtifactRetentionRecord(
                    artifact_id=artifact.artifact_id,
                    scope=owner,
                    pinned=True,
                    revision=1,
                    updated_at=_START + timedelta(microseconds=index),
                ),
                0,
            )
    for index in range(samples):
        artifact_index = index % artifact_count
        await bundle.artifacts.record_occurrence(
            ArtifactOccurrence(
                occurrence_id=f"occurrence-{index}",
                artifact_id=f"artifact-{artifact_index}",
                scope=StorageScope(
                    **owner.as_filter(),
                    session_id=f"session-{index % 5}",
                    run_id=f"run-{index % 10}",
                    graph_id="graph-baseline",
                    node_id=f"node-{index % 20}",
                ),
                action=ArtifactAction.PRODUCED,
                occurred_at=_START + timedelta(microseconds=index),
                labels={"stage": "published"},
                metrics={"quality": float(index % 100) / 100},
            )
        )


def _filtered_query(run_id: str) -> ArtifactOccurrenceQuery:
    return ArtifactOccurrenceQuery(
        owner_scope=_owner_scope(),
        scope=StorageScope(run_id=run_id),
        page=PageRequest(limit=20),
        kind="report",
        tags=("final", "reviewed"),
        labels={"category": "evidence"},
        pinned=True,
    )


async def _benchmark(bundle: Any, samples: int) -> dict[str, Any]:
    facade = CanonicalArtifactFacade(
        blobs=bundle.blobs,
        artifacts=bundle.artifacts,
        search=bundle.search,
        runs=bundle.runs,
        sessions=bundle.sessions,
        owner_scope=_owner_scope(),
        execution_scope=StorageScope(**_owner_scope().as_filter(), run_id="run-0"),
    )
    run_page_latencies: list[float] = []
    filtered_query_latencies: list[float] = []
    hydration_latencies: list[float] = []
    errors: Counter[str] = Counter()
    for index in range(samples):
        run_id = f"run-{index % 10}"
        started = perf_counter()
        try:
            page = await bundle.artifacts.query_occurrences(
                ArtifactOccurrenceQuery(
                    owner_scope=_owner_scope(),
                    scope=StorageScope(run_id=run_id),
                    page=PageRequest(limit=20),
                )
            )
            if any(item.scope.run_id != run_id for item in page.items):
                raise AssertionError("run page crossed run scope")
        except Exception as exc:
            errors[f"run_page:{type(exc).__name__}"] += 1
        else:
            run_page_latencies.append((perf_counter() - started) * 1_000)

        filtered_run_id = f"run-{(index % 5) * 2}"
        started = perf_counter()
        try:
            page = await bundle.artifacts.query_occurrences(_filtered_query(filtered_run_id))
            if any(item.scope.run_id != filtered_run_id for item in page.items):
                raise AssertionError("filtered query crossed run scope")
        except Exception as exc:
            errors[f"filtered_query:{type(exc).__name__}"] += 1
        else:
            filtered_query_latencies.append((perf_counter() - started) * 1_000)

        started = perf_counter()
        try:
            public = await facade.query_public_artifacts(
                PageRequest(limit=20),
                scope=StorageScope(run_id=filtered_run_id),
                kind="report",
                tags=("final", "reviewed"),
                labels={"category": "evidence"},
                pinned=True,
            )
            if any(item.run_id != filtered_run_id or not item.pinned for item in public.items):
                raise AssertionError("public hydration changed query semantics")
        except Exception as exc:
            errors[f"hydrate:{type(exc).__name__}"] += 1
        else:
            hydration_latencies.append((perf_counter() - started) * 1_000)
    return {
        "workload": {
            "seed_occurrences": samples,
            "run_page_operations": samples,
            "filtered_query_operations": samples,
            "hydration_operations": samples,
            "page_limit": 20,
        },
        "occurrence_run_page": _distribution(
            run_page_latencies,
            elapsed_s=sum(run_page_latencies) / 1_000,
        ),
        "filtered_occurrence_query": _distribution(
            filtered_query_latencies,
            elapsed_s=sum(filtered_query_latencies) / 1_000,
        ),
        "public_page_hydration": _distribution(
            hydration_latencies,
            elapsed_s=sum(hydration_latencies) / 1_000,
        ),
        "errors": dict(errors),
        "correctness": not errors,
    }


async def _run(samples: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aethergraph-s5-artifact-baseline-") as temporary:
        root = Path(temporary)
        tracemalloc.start()
        bundle = _open_bundle(root)
        try:
            await _populate(bundle, samples)
            artifacts = await _benchmark(bundle, samples)
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
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "samples": samples,
            "artifacts": artifacts,
            "peak_python_bytes": peak,
            "database_bytes_before_checkpoint": sizes_before_checkpoint,
            "database_bytes_after_checkpoint": sizes_after_checkpoint,
            "database_bytes_after_close": _database_sizes(root),
            "checkpoint": maintenance,
        }


def main() -> None:
    """Run and print the reproducible S5 artifact benchmark.

    The command creates one temporary manifested local provider, seeds normalized
    content/retention/occurrence state, measures indexed queries and complete frozen
    public-page hydration, checkpoints the provider, and removes the workspace.

    Examples:
        Run the recorded comparison shape:
            ```python
            python scripts/storage_provider_s5_artifact_baseline.py --samples 100
            ```

        Run a short smoke benchmark:
            ```python
            python scripts/storage_provider_s5_artifact_baseline.py --samples 10
            ```

    Args:
        None.

    Returns:
        None: JSON correctness, latency, memory, and storage evidence is printed.

    Notes:
        Generated content exists only inside a managed temporary directory. The
        benchmark does not activate production provider routing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("samples must be positive")
    print(json.dumps(asyncio.run(_run(args.samples)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
