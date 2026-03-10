from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ScenarioScorer(Protocol):
    async def score(
        self, scenario: HarnessScenario, result: HarnessRunResult
    ) -> dict[str, Any]: ...


class ContinuationResolver(Protocol):
    async def resolve(
        self,
        continuation: Any,
        *,
        scenario: HarnessScenario,
        wait_index: int,
        history: list[WaitResolutionRecord],
    ) -> dict[str, Any]: ...


@dataclass
class HarnessAttachment:
    path: str | None = None
    name: str | None = None
    mimetype: str | None = None
    uri: str | None = None
    source: str = "fixture"
    labels: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_attachment_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "artifact",
            "source": self.source,
            "name": self.name or (Path(self.path).name if self.path else None),
            "mimetype": self.mimetype,
            "uri": self.uri or self.path,
            "labels": dict(self.labels),
            "meta": dict(self.meta),
        }
        if self.path:
            payload["meta"]["fixture_path"] = self.path
        return payload

    def to_resume_file(self) -> dict[str, Any]:
        return {
            "name": self.name or (Path(self.path).name if self.path else "fixture.bin"),
            "mimetype": self.mimetype or "application/octet-stream",
            "uri": self.uri or self.path,
            "path": self.path,
            "labels": dict(self.labels),
            "meta": dict(self.meta),
        }


@dataclass
class HarnessTarget:
    agent_id: str | None = None
    graph_id: str | None = None
    target: Any | None = None

    def mode(self) -> str:
        if self.agent_id:
            return "agent"
        if self.graph_id:
            return "graph"
        if self.target is not None:
            return "direct"
        raise ValueError("HarnessTarget requires agent_id, graph_id, or target")


@dataclass
class HarnessExportConfig:
    root_dir: str
    include_llm: bool = True
    include_memory: bool = True
    include_trace: bool = True
    include_artifacts: bool = True


@dataclass
class WaitResolutionRecord:
    token: str
    kind: str
    node_id: str
    payload: dict[str, Any]
    prompt: Any = None
    status: str = "resolved"
    error: str | None = None


@dataclass
class HarnessTraceBundle:
    graph_events: list[dict[str, Any]] = field(default_factory=list)
    channel_events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    memory_summaries: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HarnessRunResult:
    scenario_id: str
    benchmark_id: str | None
    status: str
    target_mode: str
    run_id: str | None
    session_id: str
    outputs: dict[str, Any] | None = None
    error: str | None = None
    waits: list[WaitResolutionRecord] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    trace: HarnessTraceBundle = field(default_factory=HarnessTraceBundle)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessBenchmarkResult:
    benchmark_id: str
    runs: list[HarnessRunResult]
    summary: dict[str, Any]
    export_dir: str | None = None


@dataclass
class HarnessScenario:
    id: str
    target: HarnessTarget
    message: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    attachments: list[HarnessAttachment] = field(default_factory=list)
    user_meta: dict[str, Any] = field(default_factory=dict)
    wait_resolver: ContinuationResolver | None = None
    scorers: list[ScenarioScorer] = field(default_factory=list)
    expected_outputs: dict[str, Any] | None = None
    operator_overrides: Any | None = None
    timeout_s: float = 30.0
    tags: list[str] = field(default_factory=list)
    shared_session_id: str | None = None
    export: HarnessExportConfig | None = None


@dataclass
class HarnessBenchmark:
    id: str
    scenarios: list[HarnessScenario]
    max_concurrency: int = 1
    export: HarnessExportConfig | None = None
