from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
import inspect
from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.resume import ResumeEvent
from aethergraph.contracts.services.wakeup import WakeupEvent

from ..graph.graph_refs import GRAPH_INPUTS_NODE_ID
from ..graph.node_spec import NodeEvent
from ..graph.node_state import TERMINAL_STATES, WAITING_STATES, NodeStatus
from ..graph.task_node import TaskNodeRuntime
from .base_scheduler import BaseScheduler
from .retry_policy import RetryPolicy
from .step_forward import step_forward

if TYPE_CHECKING:
    from ..graph.task_graph import TaskGraph
    from ..runtime.runtime_env import RuntimeEnv


def _is_plan(node) -> bool:
    return getattr(node, "node_type", getattr(node, "type", None)) == "plan"


class ForwardScheduler(BaseScheduler):
    """
    Event-driven DAG scheduler for Aethergraph.

    Overview
    --------
    The ForwardScheduler executes a TaskGraph in "forward" mode: it starts runnable
    nodes as soon as their dependencies are DONE, up to a configurable concurrency
    limit. It is fully event-driven (no busy polling) and reacts to:
      • Task completions
      • External resumes (human/robot/time/event) delivered via a control queue
      • Backoff timers for retries

    Responsibilities
    ----------------
    • Determine runnable nodes (deps satisfied, not terminal, not running)
    • Start nodes (async) and invoke `step_forward(...)` for the work
    • Transition node state to DONE, SKIPPED, FAILED, or WAITING_*
    • Persist and publish Continuations when a node requests a wait
    • Handle Resume/Wakeup events and re-start waiting nodes with a payload
    • Enforce max concurrency and apply retry/backoff policy
    • Optionally: stop early on the first terminal failure and/or mark dependents SKIPPED

    Key Concepts
    ------------
    • Terminal states: {DONE, FAILED, SKIPPED} (plus any custom terminal states)
    • Waiting states: {WAITING_HUMAN, WAITING_ROBOT, WAITING_TIME, WAITING_EVENT}
    • Control events:
        - ResumeEvent(node_id, payload): resume a WAITING_* node with payload
        - WakeupEvent(node_id): resume due to timer/poll (payload supplied upstream)
    • Concurrency: bounded by `max_concurrency`; resumed nodes are prioritized
    • Retries: delegated to RetryPolicy (attempts, backoff); backoff sleepers are tracked

    Data Structures
    ---------------
    • running_tasks: {node_id -> asyncio.Task}
    • _events: asyncio.Queue[ResumeEvent | WakeupEvent]  (control plane)
    • _resume_payloads: {node_id -> dict}                (payload stash until start)
    • _resume_pending: set[node_id]                      (resumed but awaiting capacity)
    • _backoff_tasks: {node_id -> asyncio.Task}          (sleepers before retry)

    Run Loop (high level)
    ---------------------
    1) Drain any control events currently in `_events` (non-blocking) and handle them.
    2) Schedule work up to capacity:
         a) Start resumed waiters in `_resume_pending` first.
         b) Start newly "ready" nodes (deps DONE).
    3) If nothing is running or scheduled:
         a) If graph is effectively terminal (no running, no waiters, no pending/backoffs), exit.
         b) If any nodes are WAITING_*, block on `_events.get()` until a resume/wakeup arrives.
         c) Otherwise, the graph is stalled (likely unmet deps or failures); raise.
    4) If there is running work, wait for FIRST_COMPLETED of:
         - any running task, or
         - a new control event from `_events`.
       Then loop back to (1).

    State Transitions (per node)
    ----------------------------
    • RUNNING → DONE:
        - Persist outputs
        - Emit NodeEvent(DONE)
    • RUNNING → WAITING_*:
        - Continuation already saved and notified by `step_forward`
        - Emit NodeEvent(WAITING_*)
    • RUNNING → FAILED:
        - Set FAILED; emit NodeEvent(FAILED)
        - If retry eligible: schedule backoff sleeper and requeue later
        - If retries exhausted:
            * If `skip_dependents_on_failure=True`: mark dependents SKIPPED (transitively)
            * If `stop_on_first_error=True`: set `_terminated=True` to end the run
    • (External) WAITING_* + Resume/Wakeup:
        - Store payload, cancel backoff for that node (if any)
        - Start immediately if capacity allows; else add to `_resume_pending`

    Scheduling Order
    ----------------
    1) Resumed waiters (capacity permitting)
    2) Newly ready nodes (dependencies satisfied)
    This keeps the system responsive to external signals.

    Termination Conditions
    ----------------------
    • Natural completion: all non-plan nodes are in terminal states and
      there are no running tasks, backoffs, or pending resumes.
    • Early stop: first terminal failure with `stop_on_first_error=True`.
    • Stalled graph: no running tasks, no waiters, not terminal → raises RuntimeError.

    Configuration
    -------------
    • max_concurrency: int = 4
    • retry_policy: RetryPolicy
    • stop_on_first_error: bool = False
    • skip_dependents_on_failure: bool = True

    Performance & Safety Notes
    --------------------------
    • The loop is idle when there is no work: it blocks on either task completion
      or `_events.get()`. There is no busy waiting.
    • `_events` is drained non-blockingly at the start of each iteration to reduce
      resume latency and coalesce multiple resumes.
    • All resume paths are capacity-aware; if full, node IDs sit in `_resume_pending`.
    • Backoff timers are lightweight asyncio sleep tasks; they wake only when due.

    Extension Points
    ----------------
    • add_listener(cb): subscribe to NodeEvent emissions for metrics/telemetry.
    • _compute_ready(): override to implement custom gating/priority.
    • _skip_dependents(failed_id): override if you need custom skip rules.

    Typical Usage
    -------------
        env = RuntimeEnv(...); sched = ForwardScheduler(graph, env, max_concurrency=2)
        result = await sched.run()   # returns when the graph is effectively terminal
        # External systems call: await sched.on_resume_event(node_id, payload)

    Invariants
    ----------
    • A node is started at most once concurrently.
    • Resumes are idempotent: last payload wins before the node (re)starts.
    • Continuations are persisted before WAITING_* is reported.
    """

    def __init__(
        self,
        graph: TaskGraph,
        env: RuntimeEnv,
        retry_policy: RetryPolicy | None = None,
        *,
        max_concurrency: int = 4,
        stop_on_first_error: bool = False,
        skip_dep_on_failure: bool = True,
        logger: Any | None = None,
    ):
        """ForwardScheduler executes nodes in a forward manner, scheduling ready nodes as soon as their dependencies are met.
        It supports waiting nodes (WAITING_HUMAN, WAITING_EXTERNAL, etc.) and can resume them upon external events.

        Args:
         - graph: TaskGraph to execute.
         - env: RuntimeEnv providing runtime services and context.
         - retry_policy: RetryPolicy defining retry behavior for failed nodes.
         - max_concurrency: Maximum number of concurrent running tasks.
         - stop_on_first_error: If True, stops the entire graph execution on the first node failure.
         - skip_dep_on_failure: If True, skips downstream dependents of a failed node, but continues executing other independent nodes.
        """

        super().__init__(graph, mode="forward")
        self.env = env
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_concurrency = max_concurrency
        self.stop_on_first_error = stop_on_first_error
        self.skip_dep_on_failure = skip_dep_on_failure

        # bookkeeping
        self._resume_payloads: dict[str, dict] = {}  # node_id -> resume payload
        self._backoff_tasks: dict[str, asyncio.Task] = {}  # node_id -> backoff task
        self._resume_pending: set[str] = set()  # node_ids with resume pending but not yet started
        self._ready_pending: set[str] = set()  # node_ids that became ready but not yet started

        # event to pause/resume execution
        self._events: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = (
            None  # used by MultiSchedulerResumeBus with cross-thread calls
        )
        self._nudge = asyncio.Event()
        self._resume_tokens: set[str] = set()  # for logging/debugging

        # listeners and callbacks
        self._listeners: list[
            Callable[[NodeEvent], Awaitable[None]]
        ] = []  # Placeholder for event listeners

        # logger
        self.logger = logger

        # termination flag
        self._cancelled = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None):
        """Bind an event loop to this scheduler (for cross-thread resume calls)."""
        self.loop = loop or asyncio.get_running_loop()

    # --------- event listeners ---------
    def add_listener(self, listener: Callable[[NodeEvent], Awaitable[None]]):
        """Add an event listener that will be called on node events."""
        if not inspect.iscoroutinefunction(listener):
            raise ValueError("Listener must be an async function")
        self._listeners.append(listener)

    def _capacity(self) -> int:
        """Return available capacity for new tasks."""
        return self.max_concurrency - len(self.get_running_task_node_ids())

    async def _try_start_immediately(self, node_id: str) -> bool:
        """Try to start node now if waiting + capacity available; return True if started."""
        if self._capacity() <= 0 or node_id in self.running_tasks:
            return False
        node = self._runtime(node_id)
        if not node:
            return False
        if node.state.status not in WAITING_STATES:
            return False
        await self._start_node(node)
        return True

    async def _emit(self, event: NodeEvent):
        """Emit an event to all listeners. Should not kill the scheduler if a listener fails."""
        for cb in self._listeners:
            try:
                await cb(event)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[ForwardScheduler._emit] Error in event listener: {e}")
                else:
                    print(f"[ForwardScheduler._emit] Error in event listener: {e}")

    # --------- public API ---------
    async def deliver_resume(self, token: str):
        """
        Wake the engine: a continuation with `token` has been resolved.
        Typically this means:
         - mark the relevant node WAITING->READY
         - schedule a tick
        """
        # Implementation choices:
        # - if you keep a WaitRegistry in env, this might just trigger a run loop tick
        # - if you keep an asyncio.Event per run, set() it
        self._resume_tokens.add(token)  # optional: track for logging
        self._nudge.set()  # asyncio.Event the main loop awaits

    async def run(self):
        """Main run loop. Schedules ready nodes, handles events, and manages concurrency.

        The loop works as follows:
        - Drain any pending control events (e.g. resume and wakeup).
        - Schedule ready nodes up to max_concurrency.
        - If no tasks are running and none were scheduled:
            - If all nodes are terminal, exit.
            - If any node is WAITING_*, block for a resume event.
            - Otherwise, raise error (stalled graph).
        - If tasks are running, wait for either a task to complete or a control event.
        - Repeat until terminated.
        """
        self.loop = asyncio.get_running_loop()

        dirty = True  # something changed; try scheduling
        MAX_DRAIN = 100  # max control events to drain in one go (to avoid starvation)
        while not self._terminated:
            await self._pause_event.wait()

            # clear nudge
            self._nudge.clear()

            if dirty:
                # 1) drain already-queued control events (non-blocking)
                for _ in range(MAX_DRAIN):  # optional MAX_DRAIN guard
                    try:
                        ev = self._events.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self._handle_events(ev)
                # 2) try to schedule work
                scheduled = await self._schedule_ready()
                dirty = False
            else:
                scheduled = 0

            running = list(self.running_tasks.values())

            # 3) no work currently running or scheduled
            if not running and scheduled == 0:
                nothing_pending = (not self._backoff_tasks) and (not self._resume_pending)
                if nothing_pending and not self._any_waiting():
                    # graph is effectively terminal (DONE/FAILED/SKIPPED only)
                    self._terminated = True
                    break

                if self._any_waiting():
                    # 4) BLOCK until a resume/wakeup arrives (no CPU spin)
                    ev = await self._events.get()
                    await self._handle_events(ev)
                    dirty = True
                    continue

                # stalled: neither running nor waiting nor terminal (likely unmet deps)
                raise RuntimeError("stalled")

            # 5) We have running tasks; wait for either a task to finish OR a control event
            ctrl = asyncio.create_task(self._events.get())
            try:
                done, _ = await asyncio.wait(running + [ctrl], return_when=asyncio.FIRST_COMPLETED)
                if ctrl in done:
                    ev = ctrl.result()
                    await self._handle_events(ev)
                # either a task completed or an event arrived → state changed
                dirty = True
            finally:
                if not ctrl.done():
                    ctrl.cancel()

        if self._cancelled:
            # propagate an explicit cancellation upwards
            raise asyncio.CancelledError(
                f"ForwardScheduler for run_id={self.env.run_id} was terminated"
            )

    async def run_from(self, node_ids: list[str]):
        """Run starting from specific nodes (e.g. after external event)."""
        for nid in node_ids:
            node = self.graph.node(nid)
            if node.state.status in (
                NodeStatus.WAITING_HUMAN,
                NodeStatus.WAITING_ROBOT,
                NodeStatus.WAITING_EXTERNAL,
                NodeStatus.WAITING_TIME,
                NodeStatus.WAITING_EVENT,
            ):
                # will be executed when resume_payload arrives
                continue
            await self._start_node(node)

    async def terminate(self):
        """Terminate execution; running tasks will complete but no new tasks will be started."""
        self._terminated = True
        self._cancelled = True

        # cancel backoff tasks
        for task in self._backoff_tasks.values():
            task.cancel()
        # cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()

    async def run_node(self, node):
        """Explicitly run a specific node (e.g. for testing)."""
        await self._start_node(node)

    # ENFORCE capacity in run_one()
    async def run_one_old(self, node: TaskNodeRuntime) -> dict[str, Any]:
        # deps must be DONE (except inputs node)
        for dep in node.dependencies or []:
            if dep == GRAPH_INPUTS_NODE_ID:
                continue
            dep_node = self.graph.node(dep)
            if dep_node is None or dep_node.state.status != NodeStatus.DONE:
                raise RuntimeError(f"Cannot run node {node.node_id}: dependency {dep} not DONE")

        # If we're already at capacity, wait until any running task completes
        while self._capacity() <= 0:
            # Wait for FIRST_COMPLETED among running tasks
            running = list(self.running_tasks.values())
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

        # Start this node now that a slot is available
        await self._start_node(node)

        # Wait for this specific node to finish
        task = self.running_tasks.get(node.node_id)
        if task:
            await task
        return node.outputs or {}

    async def _wait_until_terminal(self, target_id: str):
        """Drive the scheduler event loop just enough to bring target_id to a terminal state."""
        while True:
            node = self.graph.node(target_id)
            if node.state.status in TERMINAL_STATES:
                return node.state.status

            # Prioritize resume events
            try:
                ev = self._events.get_nowait()
                await self._handle_events(ev)
            except asyncio.QueueEmpty:
                pass

            # Try to (re)start anything that became runnable
            await self._schedule_ready()

            # If nothing running, block on the next control event (resume/wakeup), then loop
            running = list(self.running_tasks.values())
            if not running:
                ev = await self._events.get()
                await self._handle_events(ev)
            else:
                # Either a running task finishes or a control event arrives
                ctrl = asyncio.create_task(self._events.get())
                try:
                    done, _ = await asyncio.wait(
                        running + [ctrl], return_when=asyncio.FIRST_COMPLETED
                    )
                    if ctrl in done:
                        await self._handle_events(ctrl.result())
                finally:
                    if not ctrl.done():
                        ctrl.cancel()

    async def run_one(self, node: TaskNodeRuntime) -> dict[str, Any]:
        """Run a single node by ID, return its outputs."""
        self.loop = asyncio.get_running_loop()  # ensure loop is set
        # deps DONE check (kept as-is) ...
        while self._capacity() <= 0:
            running = list(self.running_tasks.values())
            if not running:
                break
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

        await self._start_node(node)

        # Wait for the first execution round to finish
        task = self.running_tasks.get(node.node_id)
        if task:
            await task

        # If the node is WAITING_*, drive the loop until it becomes terminal
        n = self.graph.node(node.node_id)
        if n.state.status in WAITING_STATES:
            await self._wait_until_terminal(node.node_id)

        # Terminal: return outputs (or {} if failed/skipped)
        n = self.graph.node(node.node_id)
        if n.state.status == NodeStatus.DONE:
            return n.outputs or {}
        if n.state.status == NodeStatus.FAILED:
            # optionally raise an error here
            return n.outputs or {}
        # SKIPPED or others:
        return n.outputs or {}

    async def step_next(self):
        """Run exactly one step (for step-by-step execution)."""
        r = self._compute_ready()
        if r:
            nid = next(iter(r))
            await self._start_node(self.graph.node(nid))

    # called by ResumeRouter when external/human resumes a waiting node
    async def on_resume_event(self, run_id: str, node_id: str, payload: dict[str, Any]):
        """Called by external event trigger to resume a waiting node.
        We use async queue to schedule the resume event.
        """
        # NOTE: run_id is not needed for local scheduler,  but we need it to match the signature with GlobalScheduler
        await self._events.put(ResumeEvent(run_id, node_id, payload))

    # --------- internal methods ---------
    async def _schedule_ready(self) -> int:
        available = self._capacity()
        if available <= 0:
            return 0
        scheduled = 0

        # 1) resumed waiters first
        while available > 0 and self._resume_pending:
            nid = self._resume_pending.pop()
            node = self.graph.node(nid)
            if node and node.state.status in WAITING_STATES and nid not in self.running_tasks:
                await self._start_node(node)
                scheduled += 1
                available -= 1

        # 2) explicit-start ready nodes (from run_one) next
        while available > 0 and self._ready_pending:
            nid = self._ready_pending.pop()
            node = self.graph.node(nid)
            if (
                node
                and node.node_id not in self.running_tasks
                and node.state.status not in TERMINAL_STATES
            ):
                # still ensure deps satisfied
                if all(
                    (dep == GRAPH_INPUTS_NODE_ID)
                    or (self.graph.node(dep).state.status == NodeStatus.DONE)
                    for dep in (node.spec.dependencies or [])
                ):
                    await self._start_node(node)
                    scheduled += 1
                    available -= 1
                else:
                    pass  # deps not satisfied; skip

        # 3) normal ready nodes
        if available > 0:
            for nid in list(self._compute_ready())[:available]:
                await self._start_node(self.graph.node(nid))
                scheduled += 1

        return scheduled

    async def _skip_dependents(self, failed_node_id: str):
        """Mark all downstream dependents of failed_node_id as SKIPPED if not already terminal/running."""
        # breadth-first over reverse edges
        q = [failed_node_id]
        seen = set()
        while q:
            cur = q.pop(0)
            for n in self.graph.nodes:
                if cur in (n.spec.dependencies or []):
                    if n.node_id in seen:
                        continue
                    seen.add(n.node_id)
                    node = self.graph.node(n.node_id)
                    if (
                        node.state.status not in TERMINAL_STATES
                        and n.node_id not in self.running_tasks
                    ):
                        await self.graph.set_node_status(n.node_id, NodeStatus.SKIPPED)
                    q.append(n.node_id)

    def _compute_ready(self) -> set[str]:
        """Nodes whose deps are completed/skipped and that are not running/waiting/failed.
        Returns set of node_ids.
        The function works as follows:
        - Iterate over all nodes in the graph.
        - Skip plan nodes and nodes that are already done, failed, skipped, or waiting.
        - Skip nodes that are already running.
        - Check if all dependencies of the node are satisfied (i.e., in DONE).
        - If dependencies are satisfied, add the node_id to the ready set.
        """

        ready: set[str] = set()
        for node in self.graph.nodes:  # runtime nodes
            node_id = node.node_id
            node_status = node.state.status
            node_type = node.spec.type

            if node_type == "plan":
                continue  # skip plan nodes; TODO: we may deprecate plan node later
            if node_status in (
                NodeStatus.DONE,
                NodeStatus.FAILED,
                NodeStatus.SKIPPED,
                NodeStatus.WAITING_HUMAN,
                NodeStatus.WAITING_ROBOT,
                NodeStatus.WAITING_EXTERNAL,
                NodeStatus.WAITING_TIME,
                NodeStatus.WAITING_EVENT,
            ):
                # already done/waiting/failed
                continue

            if node_id in self.running_tasks:
                # already running
                continue

            # dependencies satisfied?
            deps_ok = True
            for dep in node.spec.dependencies or []:
                if dep == GRAPH_INPUTS_NODE_ID:
                    continue  # inputs node is always satisfied
                dep_node = self._runtime(dep)
                if dep_node is None:
                    if self.logger:
                        self.logger.warning(
                            f"Node {node_id} has missing dependency {dep}; skipping"
                        )
                    else:
                        print(
                            f"[ForwardScheduler] Node {node_id} has missing dependency {dep}; skipping"
                        )
                    deps_ok = False
                    break
                if dep_node.state.status not in [NodeStatus.DONE]:
                    deps_ok = False
                    break
            if deps_ok:
                ready.add(node_id)

        return ready

    def _runtime(self, node_id: str) -> TaskNodeRuntime:
        # get runtime node by id
        node = self.graph.node(node_id)
        return node

    async def _start_node(self, node: TaskNodeRuntime):
        node_id = node.node_id

        # attach resume payload if any (WAITING_* -> RUNNING)
        resume_payload = self._resume_payloads.pop(node_id, None)

        if node.state.status in WAITING_STATES and resume_payload is None:
            # keep it pending; it will be scheduled once a payload arrives
            self._resume_pending.add(node_id)
            return

        async def _runner():
            try:
                await self.graph.set_node_status(node_id, NodeStatus.RUNNING)
                ctx = self.env.make_ctx(
                    node=node, resume_payload=resume_payload
                )  # ExecutionContext
                result = await step_forward(node=node, ctx=ctx, retry_policy=self.retry_policy)

                if result.status == NodeStatus.DONE:
                    # normalize between output/outputs
                    outs = result.outputs or {}

                    await self.graph.set_node_outputs(node_id, outs)
                    await self.graph.set_node_status(node_id, NodeStatus.DONE)

                    # publish outputs to env for downstream consumption
                    self.env.outputs_by_node[node.node_id] = outs

                    # emit event
                    event = NodeEvent(
                        run_id=self.env.run_id,
                        graph_id=getattr(self.graph.spec, "graph_id", "inline"),
                        node_id=node.node_id,
                        status=str(NodeStatus.DONE),
                        outputs=node.outputs or {},
                        timestamp=datetime.utcnow().timestamp(),
                    )
                    await self._emit(event)

                elif result.status.startswith("WAITING_"):
                    # no outputs yet; continuation already persisted by ctx.storage via step_forward
                    # scheduler idles until on_resume() or wakeup queue triggers
                    await self.graph.set_node_status(node_id, result.status)

                    # emit event
                    event = NodeEvent(
                        run_id=self.env.run_id,
                        graph_id=getattr(self.graph.spec, "graph_id", "inline"),
                        node_id=node.node_id,
                        status=result.status,
                        outputs=node.outputs or {},
                        timestamp=datetime.utcnow().timestamp(),
                    )
                    await self._emit(event)

                elif result.status == NodeStatus.FAILED:
                    # step_forward already incremented attempts (if policy applies)
                    # If retry allowed, schedule backoff sleeper:
                    await self.graph.set_node_status(node_id, NodeStatus.FAILED)

                    # emit event
                    event = NodeEvent(
                        run_id=self.env.run_id,
                        graph_id=getattr(self.graph.spec, "graph_id", "inline"),
                        node_id=node.node_id,
                        status=str(NodeStatus.FAILED),
                        outputs=node.outputs or {},
                        timestamp=datetime.utcnow().timestamp(),
                    )
                    await self._emit(event)

                    attempts = getattr(node, "attempts", 0)
                    if attempts > 0 and attempts < self.retry_policy.max_attempts:
                        delay = self.retry_policy.backoff(
                            attempts - 1
                        ).total_seconds()  # attempts was incremented in step_forward
                        self._backoff_tasks[node.node_id] = asyncio.create_task(
                            self._sleep_and_requeue(node, delay)
                        )
                    else:
                        # retries exhausted: optionally stop or skip dependents
                        if self.skip_dep_on_failure:
                            await self._skip_dependents(node_id)
                        if self.stop_on_first_error:
                            # flip the master switch to stop the main loop

                            self._terminated = True

                elif result.status == NodeStatus.SKIPPED:
                    await self.graph.set_node_status(node_id, NodeStatus.SKIPPED)

                    # emit event
                    event = NodeEvent(
                        run_id=self.env.run_id,
                        graph_id=getattr(self.graph.spec, "graph_id", "inline"),
                        node_id=node.node_id,
                        status=str(NodeStatus.SKIPPED),
                        outputs=node.outputs or {},
                        timestamp=datetime.utcnow().timestamp(),
                    )
                    await self._emit(event)

                # record memory after step
                # record_after_step(self.env, node, result)
                # TODO: optionally map selected outputs into domain memory here

            except NotImplementedError:
                # subgraph logic not handled here; escalate to orchestrator
                await self.graph.set_node_status(node_id, NodeStatus.FAILED)
            except asyncio.CancelledError:
                # task cancelled (e.g. on terminate);
                await self.graph.set_node_status(node_id, NodeStatus.CANCELLED)
            finally:
                # remove from running tasks in caller
                pass

        task = asyncio.create_task(_runner())
        self.running_tasks[node_id] = task
        # cleanup when done
        task.add_done_callback(lambda t, nid=node_id: self.running_tasks.pop(nid, None))

    async def _sleep_and_requeue(self, node: TaskNodeRuntime, delay: float):
        try:
            await asyncio.sleep(delay)
            if not self._terminated:
                await self._start_node(node)
        except asyncio.CancelledError:
            pass
        finally:
            self._backoff_tasks.pop(node.node_id, None)

    async def _handle_events(self, ev):
        """Handle control events (e.g., resume, wakeup).
        The function works as follows:
        - If the event is a ResumeEvent:
            - Store the resume payload.
            - Cancel any backoff task for the node.
            - If the node is already running or max concurrency reached, mark it as pending and return.
            - Otherwise, start the node.
        - If the event is a WakeupEvent:
            - If the node is not running and max concurrency not reached, start the node.
        NOTE: This function assumes that the event queue is drained before scheduling new nodes.
        """
        # resume event for WAITING_* nodes
        if isinstance(ev, ResumeEvent):
            # store payload (idempotent; last write wins)
            self._resume_payloads[ev.node_id] = ev.payload

            # cancel any pending backoff for this node
            task = self._backoff_tasks.pop(ev.node_id, None)
            if task:
                task.cancel()

            # try start now, else mark pending
            started = await self._try_start_immediately(ev.node_id)
            if not started:
                self._resume_pending.add(ev.node_id)
            return

        elif isinstance(ev, WakeupEvent):
            started = await self._try_start_immediately(ev.node_id)
            # If capacity is full, nothing else to do. When a slot frees, _schedule_ready will pick it.
            return

    def _all_nodes_terminal(self) -> bool:
        # treat plan nodes as ignorable for completion
        for node in self.graph.nodes:
            if _is_plan(node):
                continue
            if node.state.status not in TERMINAL_STATES:
                return False
        return True

    def _any_waiting(self) -> bool:
        return any(
            (not _is_plan(n)) and (n.state.status in WAITING_STATES) for n in self.graph.nodes
        )

    def post_resume_event_threadsafe(self, run_id: str, node_id: str, payload: dict):
        if not self.loop or not self.loop.is_running():
            # no-op or log; bus will warn
            return
        asyncio.run_coroutine_threadsafe(self.on_resume_event(run_id, node_id, payload), self.loop)
