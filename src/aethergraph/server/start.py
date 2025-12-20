from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI
import uvicorn

from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.server.loading import GraphLoader, LoadSpec
from aethergraph.server.server_state import (
    get_running_url_if_any,
    pick_free_port,
    workspace_lock,
    write_server_state,
)

from .app_factory import create_app

_started = False
_server_thread: threading.Thread | None = None
_url: str | None = None
_uvicorn_server: uvicorn.Server | None = None
_loader = GraphLoader()


@dataclass
class ServerHandle:
    url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self, timeout_s: float = 2.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout_s)

    def block(self) -> None:
        self.thread.join()


def _make_uvicorn_server(app: FastAPI, host: str, port: int, log_level: str) -> uvicorn.Server:
    """
    Create a uvicorn.Server we can stop via server.should_exit = True.
    (Safe for background thread.)
    """
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=log_level,
        lifespan="on",
        loop="asyncio",
    )
    server = uvicorn.Server(config=config)
    server.install_signal_handlers = lambda: None  # type: ignore
    return server


def start_server(
    *,
    workspace: str = "./aethergraph_data",
    host: str = "127.0.0.1",
    port: int = 8000,  # 0 = auto free port
    log_level: str = "warning",
    unvicorn_log_level: str = "warning",
    return_container: bool = False,
    return_handle: bool = False,
    load_modules: list[str] | None = None,
    load_paths: list[str] | None = None,
    project_root: str | None = None,
    strict_load: bool = False,
) -> str | tuple[str, Any] | tuple[str, ServerHandle] | tuple[str, Any, ServerHandle]:
    """
    Start (or reuse) the AetherGraph sidecar server.

    Core idea:
      - The sidecar is the long-lived backend process that hosts the API + services
        (runs/artifacts/memory/sessions/apps/agents).
      - A "workspace" is the persistent storage root. Data survives restarts because
        stores live under `workspace/`.
      - Graphs/apps/agents are discovered by IMPORTING user code that defines decorated
        graphs (@graph_fn/@graphify/as_app/as_agent). Import triggers registration.

    Server discovery / reuse:
      - We keep one server per workspace by using:
          * workspace_lock(workspace)
          * workspace/.aethergraph/server.json
      - If an existing server for this workspace is already running, we reuse it and
        return its URL (so the UI/Electron can reconnect without caring about ports).

    Parameters:
      workspace:
        Persistent storage directory (runs/artifacts/memory/logs/etc). Use a stable path
        so the UI can show historical runs after restart.
      host:
        Bind host, usually 127.0.0.1 for local desktop.
      port:
        If 0, we auto-pick a free port (recommended for desktop/Electron).
        If non-zero, uses that fixed port (useful for dev).
      log_level / unvicorn_log_level:
        App logging vs uvicorn logging verbosity.

      load_modules:
        Optional list of Python modules to import BEFORE the server starts, e.g.
          ["my_project.graphs"]
        Importing them registers decorated graphs/apps/agents so the UI sees them immediately.
      load_paths:
        Optional list of Python file paths to import BEFORE the server starts, e.g.
          ["./graphs.py", "./more_graphs.py"]
        Use this for single-file "quick start" scripts.
      project_root:
        Optional path temporarily added to sys.path while loading modules/paths.
        Use this if the loaded files import local helpers/packages.
      strict_load:
        If True, raise immediately on import/load errors. If False, record errors in
        loader report (recommended for interactive use).

      return_container:
        If True (and we started in-process), also return the service container
        (app.state.container). NOTE: not available if we reused a server from another process.
      return_handle:
        If True (and we started in-process), return a ServerHandle with .block() and .stop().

    Typical usage patterns:

      A) "Make graphs visible to frontend" (recommended: load BEFORE start)
        url = start_server(
          workspace="./aethergraph_data",
          port=0,
          load_paths=["./_local/0_quick_start/1_local_registry.py"],
          project_root=".",   # so local imports resolve if needed
        )

      B) Notebook/script mode (keep server alive by blocking)
        url, handle = start_server(workspace="./aethergraph_data", port=0, return_handle=True)
        print("Server:", url)
        handle.block()  # keeps process alive; UI can connect repeatedly

      C) Reuse already-running server
        url = start_server(workspace="./aethergraph_data")  # returns existing URL if running

    Returns:
      - URL string in most cases, e.g. "http://127.0.0.1:53421"
      - Optionally (url, container) and/or (url, handle) if requested and started in-process.
    """
    global _started, _server_thread, _url, _uvicorn_server

    # In-process fast path
    if _started and _url:
        if return_container or return_handle:
            # We can return these because we're in-process
            # (container is attached to app only when we start it; see below)
            pass
        else:
            return _url

    # Cross-process coordination: one workspace => one server
    with workspace_lock(workspace):
        running_url = get_running_url_if_any(workspace)
        if running_url:
            # Reuse the already-running sidecar for this workspace
            _started = True
            _url = running_url
            # Cross-process: we cannot return container/handle
            return running_url

        # Load graphs BEFORE server start so /apps, /agents are populated immediately
        spec = LoadSpec(
            modules=load_modules or [],
            paths=load_paths or [],
            project_root=project_root,
            strict=strict_load,
        )
        if spec.modules or spec.paths:
            report = _loader.load(spec)
            # Optional: stash report for debugging. We'll attach it to app below.
            _loader.last_report = report

        # Build app (installs services inside create_app)
        cfg = load_settings()
        set_current_settings(cfg)
        app = create_app(workspace=workspace, cfg=cfg, log_level=log_level)

        # Optional debug info
        app.state.last_load_report = getattr(_loader, "last_report", None)

        picked_port = pick_free_port(port)
        url = f"http://{host}:{picked_port}"

        # Create stoppable server object
        server = _make_uvicorn_server(app, host, picked_port, unvicorn_log_level)

        def _target():
            server.run()

        t = threading.Thread(
            target=_target,
            name="aethergraph-sidecar",
            daemon=True,
        )
        t.start()

        # Update globals
        _server_thread = t
        _uvicorn_server = server
        _started = True
        _url = url

        # Write server.json for discovery
        write_server_state(
            workspace,
            {
                "pid": os.getpid(),
                "host": host,
                "port": picked_port,
                "url": url,
                "workspace": str(Path(workspace).resolve()),
                "started_at": time.time(),
            },
        )

        handle = ServerHandle(url=url, server=server, thread=t)

        if return_container and return_handle:
            return url, app.state.container, handle
        if return_container:
            return url, app.state.container
        if return_handle:
            return url, handle
        return url


async def start_server_async(**kw) -> str:
    # Async-friendly wrapper; still uses a thread to avoid clashing with caller loop
    return start_server(**kw)  # type: ignore[return-value]


def stop_server():
    """Stop the in-process background server (useful in tests/notebooks)."""
    global _started, _server_thread, _url, _uvicorn_server
    if not _started:
        return

    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True

    if _server_thread and _server_thread.is_alive():
        with contextlib.suppress(Exception):
            _server_thread.join(timeout=5)

    _started = False
    _server_thread = None
    _uvicorn_server = None
    _url = None


# backward compatibility
start = start_server
stop = stop_server
start_async = start_server_async
