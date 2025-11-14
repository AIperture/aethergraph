from datetime import datetime, timedelta
import functools
import inspect
from typing import Any

from aethergraph.services.continuations.continuation import Continuation

from ..graph.graph_refs import RESERVED_INJECTABLES  # {"context", "resume", "self"}
from ..graph.task_node import NodeStatus, TaskNodeRuntime
from ..runtime.execution_context import ExecutionContext
from ..runtime.node_context import NodeContext
from .retry_policy import RetryPolicy
from .step_result import StepResult
from .wait_types import WaitRequested


async def maybe_await(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


def _normalize_result(res):
    if res is None:
        return {}
    if isinstance(res, dict):
        return res
    if isinstance(res, tuple):
        return {f"out{i}": v for i, v in enumerate(res)}
    return {"result": res}


def _waiting_status(kind: str) -> str:
    return NodeStatus.from_kind(kind) if kind else NodeStatus.WAITING_EXTERNAL  # maps to WAITING_*


def unwrap_callable(fn):
    """Unwrap a callable from various wrapper types.
        This includes:
        - functions decorated with @tool or @waitable_tool (have __aether_impl__)
        - functools.partial
        - bound methods (unwrap to function)
        Returns the innermost callable.

        The function works as follows:
        - If the callable has already been seen (to prevent infinite loops), return it as is.
        - If the callable has an attribute __aether_impl__, unwrap it to that attribute.
        - If the callable is a functools.partial, unwrap it to its func attribute.
        - If the callable is a bound method, unwrap it to its __func__ attribute.
        - If none of the above, return the callable as is.
        This function is useful for extracting the core logic function from various
        wrappers that may have been applied to it.
    Args:
        fn: The callable to unwrap.
    """
    seen = set()
    while True:
        if id(fn) in seen:
            return fn
        seen.add(id(fn))
        if hasattr(fn, "__aether_impl__"):
            fn = fn.__aether_impl__
            continue
        if isinstance(fn, functools.partial):
            fn = fn.func
            continue
        if inspect.ismethod(fn):
            fn = fn.__func__
            continue
        return fn


def _flatten_inputs(resolved_inputs: dict[str, Any]) -> dict[str, Any]:
    """Copy, then expand nested 'kwargs' dict into top-level keys."""
    out = dict(resolved_inputs) if resolved_inputs else {}
    nested = out.pop("kwargs", None)
    if isinstance(nested, dict):
        # only fill missing keys to let explicit top-level override nested
        for k, v in nested.items():
            out.setdefault(k, v)
    return out


def build_call_kwargs(
    logic_fn,
    resolved_inputs: dict[str, Any],
    *,
    node_ctx: NodeContext,
    runtime_ctx: "ExecutionContext" = None,
) -> dict[str, Any]:
    """Build kwargs to call a logic function:
    - flatten resolved_inputs (expand nested kwargs)
    - inject framework args by name (node/context/logger/resume)
    - validate required args
    Returns a dict of kwargs to call the logic function.

    NOTE: the input context is the full ExecutionContext, not a limited NodeContext. The 'context' param in the output kwargs
    will be a NodeContext if the callee wants it. NodeContext is used when calling the logic function if it
    accepts a 'context' parameter.

    Raises TypeError if required args are missing.
    """
    import inspect

    if runtime_ctx is None or node_ctx is None:
        raise RuntimeError("build_call_kwargs: node_ctx and runtime_ctx are required")

    sig = inspect.signature(logic_fn)
    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    flat = _flatten_inputs(resolved_inputs)
    if "kwargs" in flat and isinstance(flat["kwargs"], dict):
        flat = {**flat, **flat["kwargs"]}
        flat.pop("kwargs", None)

    # Framework injectables (authoritative)
    inject_pool = {
        "context": node_ctx,  # always NodeContext
        "resume": getattr(runtime_ctx, "resume_payload", None),
    }

    merged = dict(flat)
    for k in RESERVED_INJECTABLES:
        if k == "self":
            continue
        if k in params or has_var_kw:
            merged[k] = inject_pool.get(k)

    if not has_var_kw:
        merged = {k: v for k, v in merged.items() if k in params}
    merged.pop("self", None)
    merged.pop("kwargs", None)

    required = [
        name
        for name, p in params.items()
        if name != "self"
        and p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    missing = [k for k in required if k not in merged]
    if missing:
        raise TypeError(
            f"{getattr(logic_fn, '__name__', type(logic_fn).__name__)} missing required arguments: {missing}. "
            f"Provided keys: {sorted(merged.keys())}"
        )
    return merged


async def step_forward(
    *, node: "TaskNodeRuntime", ctx: "ExecutionContext", retry_policy: "RetryPolicy"
) -> StepResult:
    """
    Execute one node forward:
      - resolve & inject kwargs (node/context/memory/logger/resume)
      - await async logic
      - apply should_run gate
      - route subgraph to a dedicated handler (NotImplemented here)
      - distinguish waits vs failures
      - persist Continuation on wait (token, deadline/poll, channel)
    Returns a StepResult; the runner is responsible for mutating node state.
    """
    lg = None
    if getattr(ctx, "logger_factory", None) and hasattr(ctx.logger_factory, "for_node_ctx"):
        lg = ctx.logger_factory.for_node_ctx(
            run_id=ctx.run_id, node_id=node.node_id, graph_id=getattr(ctx, "graph_id", None)
        )
    attempts = getattr(node, "attempts", 0)

    logic_fn = unwrap_callable(ctx.get_logic(node.logic))

    # Resolve graph inputs
    try:
        resolved_inputs = await ctx.resolve_inputs(node)
    except Exception as e:
        if lg:
            lg.exception("input resolution error")
        return StepResult(status=NodeStatus.FAILED, error=e)

    # should_run gate (unchanged) ...
    should = True
    if hasattr(ctx, "should_run") and callable(ctx.should_run):
        try:
            should = (
                await ctx.should_run(node, resolved_inputs)
                if inspect.iscoroutinefunction(ctx.should_run)
                else ctx.should_run(node, resolved_inputs)
            )
        except Exception as e:
            if lg:
                lg.warning(f"should_run raised {e!r}; defaulting to run=True")
    if not should:
        return StepResult(
            status=getattr(NodeStatus, "SKIPPED", "SKIPPED"), outputs={"skipped": True}
        )

    # create NodeContext once
    node_ctx = ctx.create_node_context(node)

    # Build kwargs with node_ctx as 'context' and the full ctx as 'runtime'
    kwargs = build_call_kwargs(
        logic_fn,
        resolved_inputs=resolved_inputs,
        node_ctx=node_ctx,  # <-- pass node_ctx explicitly for convenience
        runtime_ctx=ctx,  # <-- pass runtime explicitly to resolve resume payload
    )
    try:
        result = (
            await logic_fn(**kwargs)
            if inspect.iscoroutinefunction(logic_fn)
            or (callable(logic_fn) and inspect.iscoroutinefunction(logic_fn.__call__))
            else logic_fn(**kwargs)
        )

        outputs = _normalize_result(result)
        if lg:
            lg.info("done")
        return StepResult(status=NodeStatus.DONE, outputs=outputs)

    except WaitRequested as w:
        # persist a Continuation and return StepResult with WAITING_*
        if lg:
            lg.info("wait requested: %s", getattr(w, "kind", None))
        return await _enter_wait(
            node=node, ctx=ctx, node_ctx=node_ctx, lg=lg, spec=w.to_dict(), attempts=attempts
        )

    except Exception as e:
        if lg:
            lg.exception("tool error")
        if attempts < retry_policy.max_attempts and retry_policy.should_retry(e):
            backoff = retry_policy.backoff(attempts)
            if lg:
                lg.warning(f"retry scheduled in {backoff}")
            node.attempts = attempts + 1
        # import traceback; traceback.print_exc()
        return StepResult(status=NodeStatus.FAILED, error=e)


# ---- wait path ---------------------------------------------------------------
def _parse_deadline(deadline: Any, now_fn) -> datetime | None:
    if not deadline:
        return None
    if isinstance(deadline, datetime):
        return deadline
    try:
        return datetime.fromisoformat(deadline)
    except Exception:
        # allow "in N seconds" style if ever passed
        try:
            sec = int(deadline)
            return now_fn() + timedelta(seconds=sec)
        except Exception:
            return None


def normalize_wait_spec(spec: dict[str, Any], *, node_ctx: "NodeContext") -> dict[str, Any]:
    """Normalize wait spec from WaitRequested to a canonical dict that used in channel/continuation:
    In WaitSpec, we allow:
        - kind: str e.g.  "approval" | "user_input" | "human" | "robot" | "external" | "time" | "event" | ...
        - prompt: str | dict
        - resume_schema: dict
        - channel: str | None (it may be None)
        - deadline: datetime | str (ISO) | int (seconds from now)
        - poll: dict


    In the normalized dict, we ensure:
        - kind: str (default "external")
        - prompt: str | dict | None
        - resume_schema: dict | None
        - channel: str (default from node_ctx or "console:stdin")
        - deadline: datetime | None
        - poll: dict | None

    NOTE: in channel, we only allow kind to be "approval" or "user_input" for external interaction. Other kinds will
    simply push a notification without expecting a user response.
    """
    from datetime import datetime, timezone

    out = dict(spec or {})
    out["kind"] = out.get("kind") or "external"
    out["prompt"] = out.get("prompt")
    out["resume_schema"] = out.get("resume_schema")

    # Channel resolution via node_ctx
    ch = out.get("channel")
    if isinstance(ch, dict):
        ch = None
    if not ch:
        ch = node_ctx.channel()._resolve_default_key() or "console:stdin"
    out["channel"] = ch

    # Deadline
    now_fn = getattr(node_ctx, "_now", None)
    if now_fn is None:

        def now_fn():
            return datetime.now(timezone.utc)

    out["deadline"] = _parse_deadline(out.get("deadline"), now_fn)

    # Poll
    poll = out.get("poll")
    if poll:
        try:
            poll["interval_sec"] = int(poll.get("interval_sec", 30))
        except Exception:
            poll["interval_sec"] = 30
        out["poll"] = poll
    return out


async def _enter_wait(
    *, node, ctx, node_ctx, lg, spec: dict[str, Any], attempts: int
) -> StepResult:
    spec = normalize_wait_spec(spec, node_ctx=node_ctx)

    # 1) Reuse token if present
    token = spec.get("token")
    store = ctx.services.continuation_store

    # Add wait spec in node state for reference -> This has not been used anywhere yet, We need save it with TaskGraph when state changes to WAITING_*
    node.state.wait_spec = {
        "kind": spec["kind"],  # "text" | "approval" | "files" | ...
        "channel": spec.get("channel"),
        "prompt": spec.get("prompt"),
        "options": spec.get("options"),
        "meta": spec.get("meta", {}),
    }

    cont = None
    if token:
        try:
            cont = await store.get_by_token(token)
        except Exception:
            cont = None

    if cont is None:
        # fall back to minting (legacy path)
        token = token or await store.mint_token(ctx.run_id, node.node_id, attempts)
        cont = Continuation(
            run_id=ctx.run_id,
            node_id=node.node_id,
            kind=spec["kind"],
            token=token,
            prompt=spec.get("prompt"),
            resume_schema=spec.get("resume_schema"),
            channel=spec["channel"],
            deadline=spec.get("deadline"),
            poll=spec.get("poll"),
            next_wakeup_at=None,
            created_at=ctx.now(),
            attempts=attempts,
        )
    else:
        # update mutable fields
        cont.kind = spec.get("kind", cont.kind)
        cont.prompt = spec.get("prompt", cont.prompt)
        cont.resume_schema = spec.get("resume_schema", cont.resume_schema)
        cont.channel = spec.get("channel", cont.channel)
        cont.deadline = spec.get("deadline", cont.deadline)
        cont.poll = spec.get("poll", cont.poll)
        cont.attempts = attempts

    # schedule next wakeup
    if cont.poll and "interval_sec" in cont.poll:
        from datetime import timedelta

        cont.next_wakeup_at = ctx.now() + timedelta(seconds=int(cont.poll["interval_sec"]))
    elif cont.deadline:
        cont.next_wakeup_at = cont.deadline
    else:
        cont.next_wakeup_at = None

    # persist (create or update)
    await store.save(cont)

    # 2) If inline payload was captured during setup, resume immediately
    inline = spec.get("inline_payload")
    if inline is not None:
        try:
            await ctx.resume_router.resume(cont.run_id, cont.node_id, cont.token, inline)
            if lg:
                lg.debug("inline resume dispatched for token=%s", cont.token)
            # No need to notify again
            return StepResult(
                status=_waiting_status(cont.kind),
                continuation=cont,
                next_wakeup_at=cont.next_wakeup_at,
            )
        except Exception as e:
            if lg:
                lg.warning(f"inline resume failed: {e!r}; will proceed without it")

    # 3) Notify only if the tool hasn't already done it
    if not spec.get("notified", False):
        try:
            await ctx.channels.notify(cont)
            if lg:
                lg.debug("notified channel=%s", cont.channel)
        except Exception as e:
            if lg:
                lg.error(f"notify failed: {e}")

    return StepResult(
        status=_waiting_status(cont.kind),
        continuation=cont,
        next_wakeup_at=cont.next_wakeup_at,
    )
