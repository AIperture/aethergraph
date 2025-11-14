from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import uuid

from aethergraph.core.runtime.execution_context import ExecutionContext
from aethergraph.core.runtime.graph_runner import _build_env


# Ad-hoc node for temporary tasks
@dataclass
class _AdhocNode:
    node_id: str = "adhoc"
    tool_name: str | None = None
    tool_version: str | None = None


async def build_adhoc_context(
    *,
    run_id: str | None = None,
    graph_id: str = "adhoc",
    node_id: str = "adhoc",
    **rt_overrides,
) -> ExecutionContext:
    # Owner can be anything with max_concurrency; we won't really schedule
    class _Owner:
        max_concurrency = rt_overrides.get("max_concurrency", 1)

    env, retry, max_conc = await _build_env(_Owner(), inputs={}, **rt_overrides)

    env.run_id = run_id or f"adhoc-{uuid.uuid4().hex[:8]}"
    env.graph_id = graph_id

    node = _AdhocNode(node_id=node_id)
    exe_ctx = env.make_ctx(node=node, resume_payload=None)
    node_ctx = exe_ctx.create_node_context(node)

    return node_ctx


@asynccontextmanager
async def open_session(
    *,
    run_id: str | None = None,
    graph_id: str = "adhoc",
    node_id: str = "adhoc",
    **rt_overrides,
):
    """
    Open an 'adhoc' context that behaves like a NodeContext, without a real graph run.
    Advanced / scripting use only.
    """
    ctx = await build_adhoc_context(
        run_id=run_id, graph_id=graph_id, node_id=node_id, **rt_overrides
    )
    try:
        yield ctx
    finally:
        # optional: flush / close memory, artifacts, etc.
        pass
