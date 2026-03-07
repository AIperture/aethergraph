"""
Convenience module for running graphs locally in scripts.

Usage:
    from aethergraph import graph_fn, NodeContext, start_server
    from aethergraph.runner import run

    @graph_fn(name="my_graph")
    async def my_graph(text: str, *, context: NodeContext) -> str:
        return {"result": text.upper()}

    # Option A: run with a server (for UI visibility)
    url = start_server()
    result = run(my_graph, inputs={"text": "hello"})

    # Option B: run without a server (lightweight, no UI)
    result = run("my_graph", inputs={"text": "hello"})
"""

from __future__ import annotations

import asyncio
from typing import Any


def run(
    target: Any,
    *,
    inputs: dict[str, Any] | None = None,
    workspace: str = "./aethergraph_data",
    origin: str = "local",
) -> dict[str, Any] | None:
    """
    Run a graph synchronously and return its outputs.

    Args:
        target: Either a graph name (str) or a decorated graph function/object.
        inputs: Dict of inputs to pass to the graph.
        workspace: Workspace directory (used if services aren't already installed).
        origin: Run origin label (default "local").

    Returns:
        The graph outputs dict, or None if the run failed.

    Raises:
        KeyError: If the graph name is not found in the registry.
        RuntimeError: If the run fails.
    """
    return asyncio.run(_run_async(target, inputs=inputs, workspace=workspace, origin=origin))


async def run_async(
    target: Any,
    *,
    inputs: dict[str, Any] | None = None,
    workspace: str = "./aethergraph_data",
    origin: str = "local",
) -> dict[str, Any] | None:
    """Async version of run()."""
    return await _run_async(target, inputs=inputs, workspace=workspace, origin=origin)


async def _run_async(
    target: Any,
    *,
    inputs: dict[str, Any] | None = None,
    workspace: str = "./aethergraph_data",
    origin: str = "local",
) -> dict[str, Any] | None:
    from aethergraph.api.v1.deps import RequestIdentity
    from aethergraph.core.runtime.run_types import RunOrigin
    from aethergraph.core.runtime.runtime_services import current_services

    # Resolve graph_id from target
    if isinstance(target, str):
        graph_id = target
    else:
        # Decorated graph function or TaskGraph — extract name
        graph_id = getattr(target, "name", None) or getattr(target, "__name__", None)
        if graph_id is None:
            raise ValueError(f"Cannot determine graph name from {target!r}")

    # Ensure services are installed (start_server does this; but if running
    # without a server we need to bootstrap manually)
    try:
        container = current_services()
    except Exception:
        from aethergraph.config.context import set_current_settings
        from aethergraph.config.loader import load_settings
        from aethergraph.core.runtime.runtime_services import install_services
        from aethergraph.services.container.default_container import build_default_container

        cfg = load_settings()
        set_current_settings(cfg)
        container = build_default_container(root=workspace, cfg=cfg)
        install_services(container)

    rm = container.run_manager
    identity = RequestIdentity(user_id="local", org_id="local", mode="local")

    try:
        run_origin = RunOrigin(origin)
    except ValueError:
        run_origin = RunOrigin.local

    record, outputs, has_waits, continuations = await rm.run_and_wait(
        graph_id,
        inputs=inputs or {},
        identity=identity,
        origin=run_origin,
    )

    if record.status.value == "failed":
        raise RuntimeError(f"Graph run failed: {record.error}")

    return outputs
