from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from .specs import HarnessBenchmark, HarnessBenchmarkResult, HarnessRunResult, HarnessScenario


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class HarnessExporter:
    def export_run(self, scenario: HarnessScenario, result: HarnessRunResult, root_dir: str) -> str:
        root = Path(root_dir).resolve()
        run_dir = root / scenario.id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "trace").mkdir(exist_ok=True)
        (run_dir / "llm").mkdir(exist_ok=True)
        (run_dir / "memory").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        manifest = {
            "kind": "harness_run",
            "scenario_id": scenario.id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_mode": result.target_mode,
            "run_id": result.run_id,
            "session_id": result.session_id,
            "tags": result.tags,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
        )
        (run_dir / "result.json").write_text(
            json.dumps(_json_safe(asdict(result)), indent=2),
            encoding="utf-8",
        )
        (run_dir / "trace" / "graph_events.json").write_text(
            json.dumps(_json_safe(result.trace.graph_events), indent=2),
            encoding="utf-8",
        )
        (run_dir / "trace" / "channel_events.json").write_text(
            json.dumps(_json_safe(result.trace.channel_events), indent=2),
            encoding="utf-8",
        )
        (run_dir / "trace" / "snapshots.json").write_text(
            json.dumps(_json_safe(result.trace.snapshots), indent=2),
            encoding="utf-8",
        )
        (run_dir / "llm" / "calls.jsonl").write_text(
            "\n".join(json.dumps(_json_safe(row)) for row in result.trace.llm_calls)
            + ("\n" if result.trace.llm_calls else ""),
            encoding="utf-8",
        )
        (run_dir / "memory" / "events.json").write_text(
            json.dumps(_json_safe(result.trace.memory_events), indent=2),
            encoding="utf-8",
        )
        (run_dir / "memory" / "summaries.json").write_text(
            json.dumps(_json_safe(result.trace.memory_summaries), indent=2),
            encoding="utf-8",
        )

        for attachment in scenario.attachments:
            if attachment.path and Path(attachment.path).exists():
                shutil.copy2(attachment.path, run_dir / "artifacts" / Path(attachment.path).name)

        return str(run_dir)

    def export_benchmark(
        self,
        benchmark: HarnessBenchmark,
        result: HarnessBenchmarkResult,
        root_dir: str,
    ) -> str:
        root = Path(root_dir).resolve() / benchmark.id
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": "harness_benchmark",
            "benchmark_id": benchmark.id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario_ids": [scenario.id for scenario in benchmark.scenarios],
        }
        (root / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
        )
        (root / "summary.json").write_text(
            json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8"
        )
        (root / "runs.jsonl").write_text(
            "\n".join(json.dumps(_json_safe(asdict(run))) for run in result.runs)
            + ("\n" if result.runs else ""),
            encoding="utf-8",
        )
        for scenario, run in zip(benchmark.scenarios, result.runs, strict=False):
            self.export_run(scenario, run, str(root / "runs"))
        return str(root)
