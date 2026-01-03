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
    run_id: str | None = None,
    graph_id: str = "adhoc",
    node_id: str = "adhoc",
    **rt_overrides,
) -> ExecutionContext:
    """
    Build an ad-hoc execution context for running a single node outside of a scheduled graph.
    This function creates a minimal runtime environment suitable for quick, one-off executions,
    such as testing or interactive exploration. It generates a temporary run and graph context,
    instantiates an ad-hoc node, and returns a node-specific execution context.
    Examples:
        Basic usage with default parameters:
        ```python
        node_ctx = await build_adhoc_context()
        ```
        Customizing the run and session IDs:
        ```python
        node_ctx = await build_adhoc_context(run_id="test-run", session_id="dev-session")
        ```
        Overriding runtime parameters:
        ```python
        node_ctx = await build_adhoc_context(max_concurrency=4)
        ```
    Args:
        run_id: Optional string to uniquely identify this run. If not provided,
            a random ID is generated.
        session_id: Optional string to associate this context with a session.
        graph_id: Identifier for the graph. Defaults to `"adhoc"`.
        node_id: Identifier for the node. Defaults to `"adhoc"`.
        **rt_overrides: Additional runtime overrides, such as `max_concurrency`.
    Returns:
        NodeExecutionContext: The execution context for the ad-hoc node, ready for use.
    """

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
    run_id: str | None = None,
    graph_id: str = "adhoc",
    node_id: str = "adhoc",
    **rt_overrides,
):
    """
    Open an ad-hoc session context for advanced or scripting use.

    This asynchronous context manager yields a temporary context that mimics a `NodeContext`
    without requiring a real graph run. It is intended for advanced scenarios where a lightweight,
    ephemeral execution environment is needed, such as scripting, testing, or prototyping.

    Examples:
        Basic usage to open an ad-hoc session:
        ```python
        async with open_session() as ctx:
            # Use ctx as you would a NodeContext
            ...
        ```

        Overriding runtime parameters:
        ```python
        async with open_session(graph_id="mygraph", node_id="customnode", foo="bar") as ctx:
            ...
        ```

    Args:
        run_id: Optional unique identifier for the run. If None, a random or default value is used.
        graph_id: Identifier for the graph context. Defaults to "adhoc".
        node_id: Identifier for the node context. Defaults to "adhoc".
        **rt_overrides: Arbitrary keyword arguments to override runtime context parameters.

    Yields:
        NodeContext: An ad-hoc context object suitable for advanced scripting or testing.

    Raises:
        Any exception raised during context construction or teardown will propagate.

    Note:
        This context does not persist state or artifacts beyond its lifetime. Use only for
        non-production, ephemeral tasks.
    """
    ctx = await build_adhoc_context(
        run_id=run_id, graph_id=graph_id, node_id=node_id, **rt_overrides
    )
    try:
        yield ctx
    finally:
        # optional: flush / close memory, artifacts, etc.
        pass
