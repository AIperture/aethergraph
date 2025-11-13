from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from aethergraph.contracts.services.resume import ResumeEvent  # we’ll extend usage to include run_id
from aethergraph.contracts.services.wakeup import WakeupEvent
from ..graph.node_spec import NodeEvent
from ..graph.node_state import TERMINAL_STATES, WAITING_STATES, NodeStatus
from ..graph.graph_refs import GRAPH_INPUTS_NODE_ID
from ..graph.task_node import TaskNodeRuntime

from .retry_policy import RetryPolicy

if TYPE_CHECKING:
    from ..runtime.runtime_env import RuntimeEnv
    from ..graph.task_graph import TaskGraph
    from aethergraph.services.schedulers.registry import SchedulerRegistry


# --------- Global control events tagged with run_id ---------
@dataclass
class GlobalResumeEvent:
    run_id: str
    node_id: str
    payload: Dict[str, Any]

@dataclass
class GlobalWakeupEvent:
    run_id: str
    node_id: str


# --------- Per-run state ---------
@dataclass
class RunSettings:
    max_concurrency: int = 4
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    stop_on_first_error: bool = False
    skip_dependents_on_failure: bool = True

@dataclass
class RunState:
    run_id: str
    graph: "TaskGraph"
    env: "RuntimeEnv"
    settings: RunSettings

    # bookkeeping
    running_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)                    # node_id -> task
    resume_payloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)               # node_id -> payload
    resume_pending: Set[str] = field(default_factory=set)                                   # node_ids awaiting capacity
    ready_pending: Set[str] = field(default_factory=set)                                    # nodes explicitly enqueued
    backoff_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)                    # node_id -> sleeper task
    terminated: bool = False

    def capacity(self) -> int:
        return max(0, self.settings.max_concurrency - len(self.running_tasks))

    def any_waiting(self) -> bool:
        return any((n.spec.type != "plan") and (n.state.status in WAITING_STATES)
                   for n in self.graph.nodes)

    def all_terminal(self) -> bool:
        for n in self.graph.nodes:
            if n.spec.type == "plan":
                continue
            if n.state.status not in TERMINAL_STATES:
                return False
        return True


@dataclass
class RunEvent:
    run_id: str
    status: str  # "SUCCESS" | "FAILED" | "CANCELLED"
    timestamp: float

# --------- Global Forward Scheduler ---------
class GlobalForwardScheduler:
    """
    A global event-driven DAG scheduler that coordinates execution across many graphs (runs)
    in a single asyncio event loop.

    • One global control plane (queue) carrying (run_id, node_id, payload) events
    • Resumed nodes (WAITING_* -> RUNNING) are prioritized globally
    • Each run has its own capacity; also a global cap can be applied if desired
    """

    def __init__(self,
                 *,
                 registry: "SchedulerRegistry",
                 global_max_concurrency: Optional[int] = None,
                 logger: Optional[Any] = None):
        self._runs: Dict[str, RunState] = {}
        self._listeners: List[Callable[[NodeEvent], Awaitable[None]]] = []
        self._events: asyncio.Queue = asyncio.Queue()
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminated = False

        # Optional global cap across all runs
        self._global_max_concurrency = global_max_concurrency  # None => unlimited
        self._logger = logger

        # registry for MultiSchedulerResumeBus routing
        self._registry = registry

        # convenience: track our loop (used by ResumeBus cross-thread dispatch)
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

        self._run_listeners: list[Callable[[RunEvent], Awaitable[None]]] = []

    # ----- public hooks -----
    def add_listener(self, cb: Callable[[NodeEvent], Awaitable[None]]):
        if not inspect.iscoroutinefunction(cb):
            raise ValueError("Listener must be an async function")
        self._listeners.append(cb)

    async def submit(self,
                     *,
                     run_id: str,
                     graph: "TaskGraph",
                     env: "RuntimeEnv",
                     settings: Optional[RunSettings] = None):
        """Register a new run (graph+env) with optional per-run settings."""
        if run_id in self._runs:
            raise ValueError(f"run_id already submitted: {run_id}")
        rs = RunState(run_id=run_id, graph=graph, env=env, settings=settings or RunSettings())
        self._runs[run_id] = rs
        self._registry.register(run_id, self)  # so MultiSchedulerResumeBus can find us

    async def run_until_all_done(self):
        """Drive the global loop until all runs are terminal."""
        await self._drive_loop(block_until="all_done")

    async def run_until_complete(self, run_id: str):
        """Drive the global loop until the specified run is terminal."""
        await self._drive_loop(block_until=("run_done", run_id))

    async def run_forever(self):
        """Service-mode: keep running until shutdown() is called."""
        await self._drive_loop(block_until="forever")

    async def shutdown(self):
        if self._terminated:
            return
        self._terminated = True

        # mark runs as terminated & cancel sleepers/runners
        for rs in self._runs.values():
            rs.terminated = True
            for t in list(rs.backoff_tasks.values()):
                t.cancel()
            for t in list(rs.running_tasks.values()):
                t.cancel()

        # wake the driver if it's blocked on events.get()
        try:
            await self._events.put(GlobalWakeupEvent(run_id="__shutdown__", node_id="__shutdown__"))
        except RuntimeError:
            # queue may be closing; best-effort
            pass

        # also ensure the pause gate isn’t closed
        if hasattr(self, "_pause_event"):
            self._pause_event.set()


    async def terminate_run(self, run_id: str):
        rs = self._runs.get(run_id)
        if not rs:
            return
        rs.terminated = True
        for t in list(rs.backoff_tasks.values()):
            t.cancel()
        for t in list(rs.running_tasks.values()):
            t.cancel()

    # external resume/wakeup API (called by ResumeBus)
    async def on_resume_event(self, run_id: str, node_id: str, payload: Dict[str, Any]):
        await self._events.put(GlobalResumeEvent(run_id=run_id, node_id=node_id, payload=payload))

    async def on_wakeup_event(self, run_id: str, node_id: str):
        await self._events.put(GlobalWakeupEvent(run_id=run_id, node_id=node_id))

    # ----- main loop -----
    async def _drive_loop(self, *, block_until: str | Tuple[str, str]):
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        MAX_DRAIN = 200
        while not self._terminated:
            await self._pause_event.wait()

            # 1) Drain control events (non-blocking)
            drained = 0
            while drained < MAX_DRAIN:
                try:
                    ev = self._events.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained += 1
                await self._handle_event(ev)

            # 2) Attempt to schedule work
            scheduled_any = await self._schedule_global()

            # 3) Check termination conditions
            if block_until == "all_done":
                if all(rs.all_terminal() and not rs.running_tasks and not rs.backoff_tasks and not rs.resume_pending
                       for rs in self._runs.values()):
                    break

            elif isinstance(block_until, tuple) and block_until[0] == "run_done":
                tgt = self._runs.get(block_until[1])
                if tgt and tgt.all_terminal() and not tgt.running_tasks and not tgt.backoff_tasks and not tgt.resume_pending:
                    # compute a simple status
                    status = "SUCCESS"
                    for n in tgt.graph.nodes:
                        if n.spec.type == "plan": 
                            continue
                        if n.state.status == NodeStatus.FAILED:
                            status = "FAILED"; break
                    await self._emit_run(RunEvent(run_id=tgt.run_id, status=status, timestamp=datetime.utcnow().timestamp()))
                    break

            # 4) If nothing is running anywhere and nothing scheduled, decide how to wait
            any_running = any(rs.running_tasks for rs in self._runs.values())
            if not any_running and not scheduled_any:
                # if any run has waiting nodes, block for a global resume/wakeup
                if any(rs.any_waiting() for rs in self._runs.values()):
                    ev = await self._events.get()
                    await self._handle_event(ev)
                    continue

                # if all runs are terminal, and we’re not in 'forever' mode, the outer loop will exit next tick
                if block_until == "forever":
                    # idle until next event
                    ev = await self._events.get()
                    await self._handle_event(ev)
                    continue

            # 5) Wait for either any task to finish OR a control event
            running_tasks = [t for rs in self._runs.values() for t in rs.running_tasks.values()]
            ctrl = asyncio.create_task(self._events.get())
            try:
                if running_tasks:
                    done, _ = await asyncio.wait(running_tasks + [ctrl], return_when=asyncio.FIRST_COMPLETED)
                    if ctrl in done:
                        ev = ctrl.result()
                        await self._handle_event(ev)
                else:
                    # No running tasks; wait for the next control event
                    ev = await ctrl
                    await self._handle_event(ev)
            finally:
                if not ctrl.done():
                    ctrl.cancel()

    # ----- scheduling -----
    async def _schedule_global(self) -> bool:
        """
        Global scheduling:
          1) Start resumed waiters across all runs (respect per-run capacity and optional global cap)
          2) Start any explicitly pending nodes
          3) Compute new ready sets (round-robin across runs)
        """
        scheduled = 0

        def global_capacity_left() -> int:
            if self._global_max_concurrency is None:
                return 10**9
            total_running = sum(len(rs.running_tasks) for rs in self._runs.values())
            return max(0, self._global_max_concurrency - total_running)

        # phase 1: resumed waiters first (global)
        for rs in self._runs.values():
            while rs.resume_pending and rs.capacity() > 0 and global_capacity_left() > 0:
                nid = rs.resume_pending.pop()
                node = rs.graph.node(nid)
                if node and node.state.status in WAITING_STATES and nid not in rs.running_tasks:
                    await self._start_node(rs, node)
                    scheduled += 1

        # phase 2: explicit pending (from run_one-style requests)
        for rs in self._runs.values():
            while rs.ready_pending and rs.capacity() > 0 and global_capacity_left() > 0:
                nid = rs.ready_pending.pop()
                node = rs.graph.node(nid)
                if node and nid not in rs.running_tasks and node.state.status not in TERMINAL_STATES:
                    if self._deps_satisfied(rs, node):
                        await self._start_node(rs, node)
                        scheduled += 1

        # phase 3: normal ready nodes (round-robin for fairness)
        any_capacity = any(rs.capacity() > 0 for rs in self._runs.values())
        if any_capacity and global_capacity_left() > 0:
            # simple round-robin by iterating runs and taking up to run capacity
            for rs in self._runs.values():
                if rs.capacity() <= 0:
                    continue
                ready = list(self._compute_ready(rs))
                take = min(len(ready), rs.capacity(), global_capacity_left())
                for nid in ready[:take]:
                    await self._start_node(rs, rs.graph.node(nid))
                    scheduled += 1

        return scheduled > 0

    def _compute_ready(self, rs: RunState) -> Set[str]:
        ready: Set[str] = set()
        for node in rs.graph.nodes:
            node_id = node.node_id
            if node.spec.type == "plan":
                continue
            st = node.state.status
            if st in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.SKIPPED, *WAITING_STATES):
                continue
            if node_id in rs.running_tasks:
                continue
            if self._deps_satisfied(rs, node):
                ready.add(node_id)
        return ready

    def _deps_satisfied(self, rs: RunState, node: TaskNodeRuntime) -> bool:
        for dep in node.spec.dependencies or []:
            if dep == GRAPH_INPUTS_NODE_ID:
                continue
            dn = rs.graph.node(dep)
            if dn is None or dn.state.status != NodeStatus.DONE:
                return False
        return True

    # ----- event handling -----
    async def _handle_event(self, ev: GlobalResumeEvent | GlobalWakeupEvent | ResumeEvent | WakeupEvent):
        # Back-compat: if someone still enqueues a plain ResumeEvent without run_id, ignore (we’re global now).
        if isinstance(ev, ResumeEvent):
            if self._logger:
                self._logger.warning("Ignored legacy ResumeEvent without run_id in GlobalForwardScheduler")
            return
        if isinstance(ev, WakeupEvent):
            if self._logger:
                self._logger.warning("Ignored legacy WakeupEvent without run_id in GlobalForwardScheduler")
            return

        if isinstance(ev, GlobalResumeEvent):
            rs = self._runs.get(ev.run_id)
            if not rs or rs.terminated:
                return
            rs.resume_payloads[ev.node_id] = ev.payload
            # cancel any backoff
            t = rs.backoff_tasks.pop(ev.node_id, None)
            if t:
                t.cancel()
            # try immediate start
            started = await self._try_start_immediately(rs, ev.node_id)
            if not started:
                rs.resume_pending.add(ev.node_id)
            return

        if isinstance(ev, GlobalWakeupEvent):
            rs = self._runs.get(ev.run_id)
            if not rs or rs.terminated:
                return
            await self._try_start_immediately(rs, ev.node_id)
            return

    async def _try_start_immediately(self, rs: RunState, node_id: str) -> bool:
        if rs.capacity() <= 0:
            return False
        node = rs.graph.node(node_id)
        if not node:
            return False
        if node.state.status not in WAITING_STATES:
            return False
        await self._start_node(rs, node)
        return True

    # ----- node execution -----
    async def _start_node(self, rs: RunState, node: TaskNodeRuntime):
        from .step_forward import step_forward

        if rs.terminated:
            return
        node_id = node.node_id
        resume_payload = rs.resume_payloads.pop(node_id, None)

        if node.state.status in WAITING_STATES and resume_payload is None:
            # no payload yet; keep pending
            rs.resume_pending.add(node_id)
            return

        async def _runner():
            try:
                await rs.graph.set_node_status(node_id, NodeStatus.RUNNING)
                ctx = rs.env.make_ctx(node=node, resume_payload=resume_payload)
                result = await step_forward(node=node, ctx=ctx, retry_policy=rs.settings.retry_policy)

                if result.status == NodeStatus.DONE:
                    outs = result.outputs or {}
                    await rs.graph.set_node_outputs(node_id, outs)
                    await rs.graph.set_node_status(node_id, NodeStatus.DONE)
                    rs.env.outputs_by_node[node.node_id] = outs
                    await self._emit(NodeEvent(run_id=rs.env.run_id,
                                               graph_id=getattr(rs.graph.spec, "graph_id", "inline"),
                                               node_id=node.node_id,
                                               status=str(NodeStatus.DONE),
                                               outputs=outs,
                                               timestamp=datetime.utcnow().timestamp()))
                elif result.status.startswith("WAITING_"):
                    await rs.graph.set_node_status(node_id, result.status)
                    await self._emit(NodeEvent(run_id=rs.env.run_id,
                                               graph_id=getattr(rs.graph.spec, "graph_id", "inline"),
                                               node_id=node.node_id,
                                               status=result.status,
                                               outputs=node.outputs or {},
                                               timestamp=datetime.utcnow().timestamp()))
                elif result.status == NodeStatus.FAILED:
                    await rs.graph.set_node_status(node_id, NodeStatus.FAILED)
                    await self._emit(NodeEvent(run_id=rs.env.run_id,
                                               graph_id=getattr(rs.graph.spec, "graph_id", "inline"),
                                               node_id=node.node_id,
                                               status=str(NodeStatus.FAILED),
                                               outputs=node.outputs or {},
                                               timestamp=datetime.utcnow().timestamp()))
                    attempts = getattr(node, "attempts", 0)
                    if attempts > 0 and attempts < rs.settings.retry_policy.max_attempts:
                        delay = rs.settings.retry_policy.backoff(attempts - 1).total_seconds()
                        rs.backoff_tasks[node.node_id] = asyncio.create_task(self._sleep_and_requeue(rs, node, delay))
                    else:
                        if rs.settings.skip_dependents_on_failure:
                            await self._skip_dependents(rs, node_id)
                        if rs.settings.stop_on_first_error:
                            rs.terminated = True
                elif result.status == NodeStatus.SKIPPED:
                    await rs.graph.set_node_status(node_id, NodeStatus.SKIPPED)
                    await self._emit(NodeEvent(run_id=rs.env.run_id,
                                               graph_id=getattr(rs.graph.spec, "graph_id", "inline"),
                                               node_id=node.node_id,
                                               status=str(NodeStatus.SKIPPED),
                                               outputs=node.outputs or {},
                                               timestamp=datetime.utcnow().timestamp()))
            except asyncio.CancelledError:
                try:
                    await rs.graph.set_node_status(node_id, NodeStatus.FAILED)
                except Exception:
                    pass
            finally:
                pass

        task = asyncio.create_task(_runner())
        rs.running_tasks[node_id] = task
        task.add_done_callback(lambda t, nid=node_id, r=rs: r.running_tasks.pop(nid, None))

    async def _sleep_and_requeue(self, rs: RunState, node: TaskNodeRuntime, delay: float):
        try:
            await asyncio.sleep(delay)
            if not rs.terminated:
                await self._start_node(rs, node)
        except asyncio.CancelledError:
            pass
        finally:
            rs.backoff_tasks.pop(node.node_id, None)

    async def _skip_dependents(self, rs: RunState, failed_node_id: str):
        q = [failed_node_id]
        seen = set()
        while q:
            cur = q.pop(0)
            for n in rs.graph.nodes:
                if cur in (n.spec.dependencies or []):
                    if n.node_id in seen:
                        continue
                    seen.add(n.node_id)
                    node = rs.graph.node(n.node_id)
                    if node.state.status not in TERMINAL_STATES and n.node_id not in rs.running_tasks:
                        await rs.graph.set_node_status(n.node_id, NodeStatus.SKIPPED)
                    q.append(n.node_id)

    async def _emit(self, event: NodeEvent):
        for cb in self._listeners:
            try:
                await cb(event)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"[GlobalForwardScheduler._emit] listener error: {e}")
                else:
                    print(f"[GlobalForwardScheduler._emit] listener error: {e}")


    def _get_run(self, run_id: str) -> RunState:
        rs = self._runs.get(run_id)
        if not rs:
            raise KeyError(f"Unknown run_id: {run_id}")
        return rs

    async def enqueue_ready(self, run_id: str, node_id: str) -> None:
        """Mark a node in this run as explicitly pending (like old run_one)."""
        rs = self._get_run(run_id)
        rs.ready_pending.add(node_id)
        # nudge the loop (a no-op wakeup is fine)
        await self._events.put(GlobalWakeupEvent(run_id=run_id, node_id=node_id))

    async def wait_for_node_terminal(self, run_id: str, node_id: str) -> Dict[str, Any]:
        """Resolve when node reaches a terminal status; return its outputs (may be {})."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        async def _once(ev):
            if ev.run_id == run_id and ev.node_id == node_id:
                if ev.status in (str(NodeStatus.DONE), str(NodeStatus.FAILED), str(NodeStatus.SKIPPED)):
                    if not fut.done():
                        fut.set_result(ev.outputs or {})

        # one-shot listener
        self.add_listener(_once)
        try:
            return await fut
        finally:
            # best-effort: remove listener by rebuilding list (small scale)
            self._listeners = [cb for cb in self._listeners if cb is not _once]


    def post_resume_event_threadsafe(self, run_id: str, node_id: str, payload: dict) -> None:
        if self.loop is None:
            raise RuntimeError("GlobalForwardScheduler.loop is not set yet")
        self.loop.call_soon_threadsafe(
            self._events.put_nowait,
            GlobalResumeEvent(run_id=run_id, node_id=node_id, payload=payload)
        )

    def get_status(self) -> dict:
        runs = {}
        for run_id, rs in self._runs.items():
            waiting = sum(1 for n in rs.graph.nodes if n.state.status in WAITING_STATES)
            runs[run_id] = {
                "running": len(rs.running_tasks),
                "resume_pending": len(rs.resume_pending),
                "backoff_sleepers": len(rs.backoff_tasks),
                "waiting": waiting,
                "terminated": rs.terminated,
                "capacity": rs.capacity(),
            }
        total_running = sum(r["running"] for r in runs.values())
        idle = (total_running == 0) and any(r["waiting"] > 0 for r in runs.values())
        return {"idle": idle, "runs": runs}
        

    def add_run_listener(self, cb):
        if not inspect.iscoroutinefunction(cb):
            raise ValueError("Listener must be async")
        self._run_listeners.append(cb)

    async def _emit_run(self, ev: RunEvent):
        for cb in list(self._run_listeners):
            try:
                await cb(ev)
            except Exception:
                if self._logger:
                    self._logger.warning(f"run listener error: {e}")