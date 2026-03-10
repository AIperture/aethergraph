from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.graph.graph_refs import resolve_any
from aethergraph.core.runtime.graph_runner import run_async
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunStatus, RunVisibility
from aethergraph.services.container.default_container import DefaultContainer, get_container

from .export import HarnessExporter
from .overrides import use_operator_overrides
from .scoring import ExactOutputScorer
from .simulation import FailOnWaitResponder
from .specs import (
    HarnessBenchmark,
    HarnessBenchmarkResult,
    HarnessRunResult,
    HarnessScenario,
    HarnessTraceBundle,
    WaitResolutionRecord,
)


class HarnessRunner:
    def __init__(
        self, *, container: DefaultContainer | None = None, exporter: HarnessExporter | None = None
    ):
        self.container = container or get_container()
        self.exporter = exporter or HarnessExporter()

    async def run_benchmark(self, benchmark: HarnessBenchmark) -> HarnessBenchmarkResult:
        semaphore = asyncio.Semaphore(max(1, benchmark.max_concurrency))

        async def _run_one(scenario: HarnessScenario) -> HarnessRunResult:
            async with semaphore:
                return await self.run_scenario(scenario, benchmark_id=benchmark.id)

        runs = await asyncio.gather(*[_run_one(scenario) for scenario in benchmark.scenarios])
        summary = self._summarize_runs(runs)
        export_dir = None
        if benchmark.export is not None:
            bench_result = HarnessBenchmarkResult(
                benchmark_id=benchmark.id, runs=runs, summary=summary
            )
            export_dir = self.exporter.export_benchmark(
                benchmark, bench_result, benchmark.export.root_dir
            )
        return HarnessBenchmarkResult(
            benchmark_id=benchmark.id,
            runs=runs,
            summary=summary,
            export_dir=export_dir,
        )

    async def run_scenario(
        self, scenario: HarnessScenario, *, benchmark_id: str | None = None
    ) -> HarnessRunResult:
        started = datetime.now(timezone.utc)
        session_id = scenario.shared_session_id or f"harness-session-{uuid4().hex[:12]}"
        run_id = f"harness-run-{uuid4().hex[:12]}"
        tags = [
            f"harness:scenario:{scenario.id}",
            f"harness:session:{session_id}",
        ]
        if benchmark_id:
            tags.append(f"harness:benchmark:{benchmark_id}")
        tags.extend(scenario.tags)
        result = HarnessRunResult(
            scenario_id=scenario.id,
            benchmark_id=benchmark_id,
            status="running",
            target_mode=scenario.target.mode(),
            run_id=run_id,
            session_id=session_id,
            started_at=started.isoformat(),
            tags=tags,
            metadata={"harness_run_id": run_id},
        )

        scorers = list(scenario.scorers)
        if scenario.expected_outputs is not None:
            scorers.append(ExactOutputScorer())
        wait_resolver = scenario.wait_resolver or FailOnWaitResponder()

        try:
            with use_operator_overrides(scenario.operator_overrides):
                if scenario.target.mode() == "direct":
                    result.outputs = await asyncio.wait_for(
                        self._run_direct_target(scenario, run_id=run_id, session_id=session_id),
                        timeout=scenario.timeout_s,
                    )
                    result.status = "succeeded"
                else:
                    outputs, waits = await asyncio.wait_for(
                        self._run_managed_target(
                            scenario,
                            run_id=run_id,
                            session_id=session_id,
                            benchmark_id=benchmark_id,
                            tags=tags,
                            wait_resolver=wait_resolver,
                        ),
                        timeout=scenario.timeout_s,
                    )
                    result.outputs = outputs
                    result.waits = waits
                    result.status = "succeeded"
        except asyncio.TimeoutError:
            if result.run_id:
                await self.container.run_manager.cancel_run(result.run_id)
            result.status = "timeout"
            result.error = f"Scenario exceeded timeout ({scenario.timeout_s}s)"
        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

        finished = datetime.now(timezone.utc)
        result.finished_at = finished.isoformat()
        result.duration_s = max(0.0, (finished - started).total_seconds())
        result.trace = await self._collect_trace(
            scenario,
            run_id=result.run_id,
            session_id=session_id,
        )
        if scorers:
            result.scores = {}
            for scorer in scorers:
                score = await scorer.score(scenario, result)
                result.scores[score.get("name", scorer.__class__.__name__)] = score
        if scenario.export is not None:
            self.exporter.export_run(scenario, result, scenario.export.root_dir)
        return result

    async def _run_direct_target(
        self, scenario: HarnessScenario, *, run_id: str, session_id: str
    ) -> dict:
        identity = RequestIdentity(user_id="local", org_id="local", mode="local")
        inputs = dict(scenario.inputs)
        if scenario.message is not None and "message" not in inputs:
            inputs["message"] = scenario.message
        if scenario.attachments and "attachments" not in inputs:
            inputs["attachments"] = [
                attachment.to_attachment_dict() for attachment in scenario.attachments
            ]
        return await run_async(
            scenario.target.target,
            inputs=inputs,
            identity=identity,
            run_id=run_id,
            session_id=session_id,
        )

    async def _run_managed_target(
        self,
        scenario: HarnessScenario,
        *,
        run_id: str,
        session_id: str,
        benchmark_id: str | None,
        tags: list[str],
        wait_resolver,
    ) -> tuple[dict | None, list[WaitResolutionRecord]]:
        rm = self.container.run_manager
        identity = RequestIdentity(user_id="local", org_id="local", mode="local")
        waits: list[WaitResolutionRecord] = []
        terminal_future = await rm._get_or_create_run_future(run_id)
        record = await self._submit_scenario_run(
            scenario,
            run_id=run_id,
            session_id=session_id,
            benchmark_id=benchmark_id,
            tags=tags,
            identity=identity,
        )
        terminal = asyncio.create_task(self._await_terminal_result(terminal_future))
        seen_tokens: set[str] = set()

        try:
            while True:
                if terminal.done():
                    final_record, outputs = await terminal
                    if final_record.status == RunStatus.failed:
                        raise RuntimeError(final_record.error or "run failed")
                    if final_record.status == RunStatus.canceled:
                        raise RuntimeError(final_record.error or "run canceled")
                    return outputs, waits

                persisted = await rm.get_record(record.run_id)
                if persisted is not None and persisted.status in {
                    RunStatus.succeeded,
                    RunStatus.failed,
                    RunStatus.canceled,
                }:
                    if persisted.status == RunStatus.failed:
                        raise RuntimeError(persisted.error or "run failed")
                    if persisted.status == RunStatus.canceled:
                        raise RuntimeError(persisted.error or "run canceled")
                    outputs = await self._recover_managed_outputs(scenario, run_id=record.run_id)
                    return outputs, waits

                continuations = await self.container.cont_store.list_cont_by_run(record.run_id)
                for cont in continuations:
                    if cont.closed or cont.token in seen_tokens:
                        continue
                    payload = await wait_resolver.resolve(
                        cont,
                        scenario=scenario,
                        wait_index=len(waits),
                        history=waits,
                    )
                    waits.append(
                        WaitResolutionRecord(
                            token=cont.token,
                            kind=cont.kind,
                            node_id=cont.node_id,
                            payload=payload,
                            prompt=cont.prompt,
                        )
                    )
                    seen_tokens.add(cont.token)
                    await self.container.resume_router.resume(
                        cont.run_id,
                        cont.node_id,
                        cont.token,
                        payload,
                    )
                await asyncio.sleep(0.05)
        finally:
            if not terminal.done():
                terminal.cancel()

    async def _await_terminal_result(self, fut: asyncio.Future) -> tuple[Any, dict | None]:
        result = await fut
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None

    async def _recover_managed_outputs(
        self,
        scenario: HarnessScenario,
        *,
        run_id: str,
    ) -> dict | None:
        if self.container.state_store is None:
            return None
        graph_id = self._resolve_managed_graph_id(scenario)
        if not graph_id:
            return None
        try:
            graph = self.container.registry.get_graph(name=graph_id, version=None)
        except KeyError:
            return None
        snapshot = await self.container.state_store.load_latest_snapshot(run_id)
        if snapshot is None:
            return None
        bindings = (graph.io_signature().get("outputs", {}) or {}).get("bindings", {}) or {}
        node_states = (snapshot.state or {}).get("nodes", {}) or {}
        outputs_by_node = {
            node_id: dict((node_state or {}).get("outputs") or {})
            for node_id, node_state in node_states.items()
            if (node_state or {}).get("outputs")
        }
        graph_inputs = dict((snapshot.state or {}).get("_bound_inputs") or scenario.inputs)
        if scenario.message is not None and "message" not in graph_inputs:
            graph_inputs["message"] = scenario.message
        if scenario.attachments and "attachments" not in graph_inputs:
            graph_inputs["attachments"] = [
                attachment.to_attachment_dict() for attachment in scenario.attachments
            ]
        try:
            return {
                key: resolve_any(
                    binding,
                    graph_inputs=graph_inputs,
                    outputs_by_node=outputs_by_node,
                )
                for key, binding in bindings.items()
            }
        except Exception:
            return None

    def _resolve_managed_graph_id(self, scenario: HarnessScenario) -> str | None:
        if scenario.target.graph_id:
            return scenario.target.graph_id
        if scenario.target.agent_id:
            meta = (
                self.container.registry.get_meta(nspace="agent", name=scenario.target.agent_id)
                or {}
            )
            backing = meta.get("backing", {}) or {}
            return backing.get("name") or meta.get("graph_id")
        return None

    async def _submit_scenario_run(
        self,
        scenario: HarnessScenario,
        *,
        run_id: str,
        session_id: str,
        benchmark_id: str | None,
        tags: list[str],
        identity: RequestIdentity,
    ):
        rm = self.container.run_manager
        user_meta = {
            **scenario.user_meta,
            "harness_run_id": run_id,
            "scenario_id": scenario.id,
        }
        if benchmark_id:
            user_meta["benchmark_id"] = benchmark_id

        if scenario.target.agent_id:
            meta = (
                self.container.registry.get_meta(nspace="agent", name=scenario.target.agent_id)
                or {}
            )
            backing = meta.get("backing", {})
            graph_id = backing.get("name") or meta.get("graph_id") or scenario.target.graph_id
            if not graph_id:
                raise RuntimeError(f"Agent {scenario.target.agent_id!r} has no graph backing")
            if self.container.eventlog is not None and (scenario.message or scenario.attachments):
                await self.container.eventlog.append(
                    {
                        "id": str(uuid4()),
                        "ts": datetime.now(timezone.utc).timestamp(),
                        "scope_id": session_id,
                        "kind": "session_chat",
                        "payload": {
                            "type": "user.message",
                            "text": scenario.message or "",
                            "files": [],
                            "attachments": [
                                attachment.to_attachment_dict()
                                for attachment in scenario.attachments
                            ],
                            "meta": {"direction": "inbound", "role": "user", **user_meta},
                        },
                    }
                )
            return await rm.submit_run(
                graph_id=graph_id,
                inputs={
                    "message": scenario.message or scenario.inputs.get("message", ""),
                    "attachments": [
                        attachment.to_attachment_dict() for attachment in scenario.attachments
                    ],
                    "session_id": session_id,
                    "user_meta": user_meta,
                },
                run_id=run_id,
                session_id=session_id,
                tags=tags,
                identity=identity,
                origin=RunOrigin.chat,
                visibility=RunVisibility.inline,
                importance=RunImportance.ephemeral,
                agent_id=scenario.target.agent_id,
                app_id=meta.get("app_id"),
            )

        inputs = dict(scenario.inputs)
        if scenario.message is not None and "message" not in inputs:
            inputs["message"] = scenario.message
        if scenario.attachments and "attachments" not in inputs:
            inputs["attachments"] = [
                attachment.to_attachment_dict() for attachment in scenario.attachments
            ]
        inputs.setdefault("user_meta", user_meta)
        return await rm.submit_run(
            graph_id=scenario.target.graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=session_id,
            tags=tags,
            identity=identity,
            origin=RunOrigin.local,
            visibility=RunVisibility.normal,
            importance=RunImportance.normal,
        )

    async def _collect_trace(
        self, scenario: HarnessScenario, *, run_id: str | None, session_id: str
    ) -> HarnessTraceBundle:
        bundle = HarnessTraceBundle()
        if run_id and self.container.state_store is not None:
            bundle.graph_events = [
                asdict(ev) for ev in await self.container.state_store.load_events_since(run_id, -1)
            ]
            snapshot = await self.container.state_store.load_latest_snapshot(run_id)
            if snapshot is not None:
                bundle.snapshots = [asdict(snapshot)]
        if self.container.eventlog is not None:
            bundle.channel_events = await self.container.eventlog.query(
                scope_id=session_id,
                kinds=["session_chat"],
                limit=500,
            )
        bundle.llm_calls = await self._read_llm_rows(run_id=run_id, session_id=session_id)
        if run_id:
            bundle.memory_events, bundle.memory_summaries = await self._read_memory_rows(
                scenario=scenario,
                run_id=run_id,
                session_id=session_id,
            )
        return bundle

    async def _read_llm_rows(self, *, run_id: str | None, session_id: str) -> list[dict]:
        obs_path = getattr(self.container, "llm_observation_path", None)
        if not obs_path:
            return []
        path = Path(obs_path)
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id and row.get("run_id") == run_id or row.get("session_id") == session_id:
                rows.append(row)
        return rows

    async def _read_memory_rows(
        self, *, scenario: HarnessScenario, run_id: str, session_id: str
    ) -> tuple[list[dict], list[dict]]:
        graph_id = scenario.target.graph_id
        if graph_id is None and scenario.target.agent_id:
            meta = (
                self.container.registry.get_meta(nspace="agent", name=scenario.target.agent_id)
                or {}
            )
            graph_id = (meta.get("backing") or {}).get("name")
        graph_id = graph_id or "harness"
        scope = None
        if self.container.scope_factory is not None:
            scope = self.container.scope_factory.for_memory(
                identity=RequestIdentity(user_id="local", org_id="local", mode="local"),
                run_id=run_id,
                graph_id=graph_id,
                node_id="harness",
                session_id=session_id,
                app_id=None,
                agent_id=scenario.target.agent_id,
                level="session" if scenario.target.agent_id else "run",
                custom_scope_id=None,
            )
        mem = self.container.memory_factory.for_session(
            run_id=run_id,
            graph_id=graph_id,
            node_id="harness",
            session_id=session_id,
            scope=scope,
            scoped_indices=None,
        )
        hot_events = await self.container.memory_factory.hotlog.query(
            mem.timeline_id, session_id=session_id, limit=200
        )
        persisted = await self.container.memory_factory.persistence.query_events(
            mem.timeline_id, session_id=session_id, limit=200
        )
        summaries = await self.container.memory_factory.persistence.query_summaries(
            scope_id=mem.memory_scope_id,
            timeline_id=mem.timeline_id,
            limit=50,
        )
        merged: dict[str, dict] = {}
        for event in [*hot_events, *persisted]:
            payload = asdict(event)
            merged[payload["event_id"]] = payload
        return list(merged.values()), summaries

    def _summarize_runs(self, runs: list[HarnessRunResult]) -> dict:
        succeeded = sum(1 for run in runs if run.status == "succeeded")
        failed = sum(1 for run in runs if run.status != "succeeded")
        durations = sorted([run.duration_s for run in runs if run.duration_s is not None])
        p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else 0.0
        return {
            "runs": len(runs),
            "succeeded": succeeded,
            "failed": failed,
            "p95_duration_s": p95,
            "statuses": {run.scenario_id: run.status for run in runs},
        }
