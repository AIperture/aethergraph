from __future__ import annotations

import asyncio
import threading
from typing import Any
import uuid

from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.contracts.services.state_stores import GraphSnapshot
from aethergraph.core.graph.task_graph import TaskGraph
from aethergraph.core.runtime.recovery import hash_spec, recover_graph_run
from aethergraph.services.container.default_container import build_default_container  # adjust path
from aethergraph.services.state_stores.graph_observer import PersistenceObserver
from aethergraph.services.state_stores.resume_policy import (
    assert_snapshot_json_only,
)
from aethergraph.services.state_stores.utils import snapshot_from_graph

from ..execution.forward_scheduler import ForwardScheduler
from ..execution.retry_policy import RetryPolicy
from ..graph.graph_fn import GraphFunction
from ..graph.graph_refs import resolve_any as _resolve_any
from ..runtime.runtime_env import RuntimeEnv
from ..runtime.runtime_metering import current_meter_context
from ..runtime.runtime_services import ensure_services_installed
from .run_registration import RunRegistrationGuard


# ---------- env helpers ----------
def _get_container():
    # install once if not installed by sidecar/server
    return ensure_services_installed(build_default_container)


async def _attach_persistence(graph, env, spec, snapshot_every=1) -> PersistenceObserver:
    """
    Wire the centralized state_store to the graph via PersistenceObserver.
    Returns the observer instance so caller can optionally force a final snapshot.
    """
    store = getattr(env.container, "state_store", None) or getattr(env, "state_store", None)
    if not store:
        # Safe no-op: resumability won't work but run still executes.
        return None

    obs = PersistenceObserver(
        store=store,
        artifact_store=getattr(env.container, "artifacts", None),
        spec_hash=hash_spec(spec),
        snapshot_every=snapshot_every,
    )
    graph.add_observer(obs)
    return obs


async def _build_env(
    owner, inputs: dict[str, Any], **rt_overrides
) -> tuple[RuntimeEnv, RetryPolicy, int]:
    container = _get_container()
    # apply optional overrides onto the container instance
    for k, v in rt_overrides.items():
        if v is not None and hasattr(container, k):
            setattr(container, k, v)

    run_id = rt_overrides.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"
    session_id = rt_overrides.get("session_id")
    graph_id = getattr(owner, "graph_id", None) or getattr(owner, "name", None)
    env = RuntimeEnv(
        run_id=run_id,
        graph_id=graph_id,
        session_id=session_id,
        graph_inputs=inputs,
        outputs_by_node={},
        container=container,
    )
    retry = rt_overrides.get("retry") or RetryPolicy()
    max_conc = rt_overrides.get("max_concurrency", getattr(owner, "max_concurrency", 4))
    return env, retry, max_conc


# ---------- materialization ----------
def _materialize_task_graph(target) -> Any:
    """
    Accept:
      - TaskGraph instance (has io_signature attr)
      - graph builder object with .build()
      - a callable builder that returns a TaskGraph when invoked with no args
    """
    # already a TaskGraph
    if hasattr(target, "io_signature"):
        return target

    # builder pattern with .build()
    if hasattr(target, "build") and callable(target.build):
        g = target.build()
        if hasattr(g, "io_signature"):
            return g

    # callable builder that returns a TaskGraph
    if callable(target):
        g = target()
        if hasattr(g, "io_signature"):
            return g

    raise TypeError(
        "run_async: target must be a TaskGraph instance, a TaskGraph builder, "
        "or a callable returning a TaskGraph."
    )


def _resolve_graph_outputs(
    graph,
    inputs: dict[str, Any],
    env: RuntimeEnv,
):
    bindings = graph.io_signature().get("outputs", {}).get("bindings", {})

    def _res(b):
        return _resolve_any(b, graph_inputs=inputs, outputs_by_node=env.outputs_by_node)

    try:
        result = {k: _res(v) for k, v in bindings.items()}
    except KeyError as e:
        waiting = [
            nid
            for nid, n in graph.state.nodes.items()
            if getattr(n, "status", "").startswith("WAITING_")
        ]
        continuations = []
        if env.continuation_store and hasattr(env.continuation_store, "get"):
            for nid in waiting:
                cont = env.continuation_store.get(run_id=env.run_id, node_id=nid)
                if cont:
                    continuations.append(
                        {
                            "node_id": nid,
                            "kind": cont.kind,
                            "token": cont.token,
                            "channel": cont.channel,
                            "deadline": getattr(cont.deadline, "isoformat", lambda: None)(),
                        }
                    )
        raise GraphHasPendingWaits(
            "Graph quiesced with pending waits; outputs are not yet resolvable.",
            waiting_nodes=waiting,
            continuations=continuations,
        ) from e

    return next(iter(result.values())) if len(result) == 1 else result


def _resolve_graph_outputs_or_waits(graph, inputs, env, *, raise_on_waits: bool = True):
    try:
        return _resolve_graph_outputs(graph, inputs, env)
    except GraphHasPendingWaits as e:
        if raise_on_waits:
            raise
        return {
            "status": "waiting",
            "waiting_nodes": e.waiting_nodes,
            "continuations": e.continuations,
        }


def _seed_outputs_from_snapshot(env, snap: GraphSnapshot):
    env.outputs_by_node = env.outputs_by_node or {}
    nodes = snap.state.get("nodes", {})
    for nid, ns in nodes.items():
        outs = (ns or {}).get("outputs") or {}
        if outs:
            env.outputs_by_node[nid] = outs


def _is_graph_complete(snap: GraphSnapshot) -> bool:
    nodes = snap.state.get("nodes", {})
    if not nodes:
        return False

    # completed if every node is DONE/SKIPPED or has outputs matching spec
    def doneish(st):
        s = (st or {}).get("status", "")
        return s in ("DONE", "SKIPPED")

    return all(doneish(ns) for ns in nodes.values())


async def load_latest_snapshot_json(store, run_id: str) -> dict[str, Any] | None:
    """
    Returns the raw JSON dict of the latest snapshot (or None).
    """
    snap = await store.load_latest_snapshot(run_id)
    if not snap:
        return None
    # JsonGraphStateStore serializes GraphSnapshot via snap.__dict__
    # load_latest_snapshot already returns a GraphSnapshot(**jsondict).
    # Convert back to plain JSON-ish dict:

    return {
        "run_id": snap.run_id,
        "graph_id": snap.graph_id,
        "rev": snap.rev,
        "spec_hash": snap.spec_hash,
        "state": snap.state,
    }


def _register_metering_context(
    env: RuntimeEnv, target: GraphFunction | TaskGraph | Any
) -> dict[str, Any]:
    """
    Build a metering context dict from the RuntimeEnv.
    """
    run_id = env.run_id
    # derive graph_id from target if possible - GraphFunction.name or TaskGraph.spec.graph_id
    graph_id = getattr(target, "name", None) or getattr(
        getattr(target, "spec", None), "graph_id", None
    )

    # user id etc through auth context if available
    user_id = getattr(getattr(env, "auth", None), "user_id", None)
    org_id = getattr(getattr(env, "auth", None), "org_id", None)

    token = current_meter_context.set(
        {
            "run_id": run_id,
            "graph_id": graph_id,
            "user_id": user_id,
            "org_id": org_id,
        }
    )
    return token


# ---------- public API ----------
async def run_async(target, inputs: dict[str, Any] | None = None, **rt_overrides):
    """
    Generic async runner for TaskGraph or GraphFunction.
    - GraphFunction → delegates to gf.run(env=..., **inputs)
    - TaskGraph/builder → schedules and resolves graph-level outputs
    """
    inputs = inputs or {}
    # GraphFunction path
    if isinstance(target, GraphFunction):
        env, retry, max_conc = await _build_env(target, inputs, **rt_overrides)
        token = _register_metering_context(env, target)  # set metering context
        try:
            return await target.run(env=env, max_concurrency=max_conc, **inputs)
        finally:
            # reset metering context
            current_meter_context.reset(token)

    # TaskGraph path
    graph = _materialize_task_graph(target)
    env, retry, max_conc = await _build_env(graph, inputs, **rt_overrides)

    # Extract spec for run/recovery ...
    spec = getattr(graph, "spec", None) or getattr(graph, "get_spec", lambda: None)()
    if spec is None:
        spec = graph.spec

    store = getattr(env.container, "state_store", None)
    snap = None
    assert store is None or hasattr(
        store, "load_latest_snapshot"
    ), "state_store must implement lo  ad_latest_snapshot(run_id)"

    if store:
        # 1) Attempt cold-resume (build a graph with hydrated state)
        graph = await recover_graph_run(spec=spec, run_id=env.run_id, store=store)

        # 2) Load raw JSON snapshot and ENFORCE strict policy
        snap_json = await load_latest_snapshot_json(store, env.run_id)
        if snap_json:
            # keep for short-circuit + seeding
            snap = await store.load_latest_snapshot(env.run_id)
            # Short-circuit if already complete
            if snap:
                _seed_outputs_from_snapshot(env, snap)
                if _is_graph_complete(snap):
                    return _resolve_graph_outputs(graph, inputs, env)

            # strict policy: block resume if any non-JSON / __aether_ref__ is present
            assert_snapshot_json_only(env.run_id, snap_json, mode="reuse_only")
    else:
        graph = _materialize_task_graph(target)

    # Bind/validate inputs
    graph._validate_and_bind_inputs(inputs)

    # Attach persistence observer + run (unchanged) ...
    obs = await _attach_persistence(graph, env, spec, snapshot_every=1)

    # get logger from env's container
    from ..runtime.runtime_services import current_logger_factory

    logger = current_logger_factory().for_scheduler()

    sched = ForwardScheduler(
        graph,
        env,
        retry_policy=retry,
        max_concurrency=max_conc,
        skip_dep_on_failure=True,
        stop_on_first_error=True,
        logger=logger,
    )

    # Register for resumes and run
    token = _register_metering_context(env, target)  # set metering context
    try:
        with RunRegistrationGuard(run_id=env.run_id, scheduler=sched, container=env.container):
            try:
                await sched.run()
            except asyncio.CancelledError:
                raise
            finally:
                # FINAL SNAPSHOT on normal or cancelled exit (if store exists)
                if store and obs:
                    artifacts = getattr(env.container, "artifacts", None)
                    snap = await snapshot_from_graph(
                        run_id=graph.state.run_id or env.run_id,
                        graph_id=graph.graph_id,
                        rev=graph.state.rev,
                        spec_hash=hash_spec(spec),
                        state_obj=graph.state,
                        artifacts=artifacts,
                        allow_externalize=False,  # FIXME: artifact writer async loop error; set False to *avoid* writing artifacts during snapshot
                        include_wait_spec=True,
                    )
                    await store.save_snapshot(snap)

        # Resolve graph-level outputs (will raise  if waits)
        return _resolve_graph_outputs_or_waits(graph, inputs, env, raise_on_waits=True)
    finally:
        # reset metering context
        current_meter_context.reset(token)


async def run_or_resume_async(
    target,
    inputs: dict[str, Any],
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    **rt_overrides,
):
    """
    If state exists for run_id → cold resume, else fresh run.
    Exactly the same signature as run_async plus optional run_id.
    """
    if run_id is not None:
        rt_overrides = dict(rt_overrides or {}, run_id=run_id)
    if session_id is not None:
        rt_overrides = dict(rt_overrides or {}, session_id=session_id)
    return await run_async(target, inputs, **rt_overrides)


# sync adapter (optional, safe in notebooks/servers)
class _LoopThread:
    def __init__(self):
        self._ev = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._loop = None
        self._thread.start()
        self._ev.wait()

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ev.set()
        loop.run_forever()

    def submit_old(self, coro):
        # this will block terminal until coro is done
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def submit(self, coro):
        # this will allow KeyboardInterrupt to propagate -> still not perfect. Use async main if possible.
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result()
        except KeyboardInterrupt:
            # cancel the task in the loop thread and wait for cleanup
            fut.cancel()

            def _cancel_all():
                for t in asyncio.all_tasks(self._loop):
                    t.cancel()

            self._loop.call_soon_threadsafe(_cancel_all)
            raise


_LOOP = _LoopThread()


def run(target, inputs: dict[str, Any] | None = None, **rt_overrides):
    inputs = inputs or {}
    return _LOOP.submit(run_async(target, inputs, **rt_overrides))
