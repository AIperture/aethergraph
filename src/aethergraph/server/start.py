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
    Start (or reuse) the AetherGraph sidecar server in a normalized and flexible way.

    This method manages server lifecycle, workspace locking, and dynamic loading of user code.
    It supports both in-process and cross-process server reuse, and can return handles for
    advanced control or integration.

    Examples:
      Basic usage to start a server and get its URL:
      ```python
      url = start_server(workspace="./aethergraph_data", port=0)
      ```

      Loading user graphs before starting:
      ```python
      url = start_server(
        workspace="./aethergraph_data",
        port=0,
        load_paths=["./my_graphs.py"],
        project_root=".",
      )
      ```

      Starting and blocking until server exit (notebook/script mode):
      ```python
      url, handle = start_server(workspace="./aethergraph_data", port=0, return_handle=True)
      print("Server running at", url)
      try:
        handle.block()
      except KeyboardInterrupt:
        print("Stopping server...")
        handle.stop()
      ```

      Returning the dependency injection container for advanced use:
      ```python
      url, container = start_server(workspace="./aethergraph_data", return_container=True)
      ```

      Returning both container and handle:
      ```python
      url, container, handle = start_server(
        workspace="./aethergraph_data",
        return_container=True,
        return_handle=True,
      )
      ```

    Args:
      workspace: Persistent storage directory for server state and data.
      host: Host address to bind the server (default "127.0.0.1").
      port: Port to bind the server (0 for auto-pick, or specify a fixed port).
      log_level: Logging level for the application.
      unvicorn_log_level: Logging level for the Uvicorn server.
      return_container: If True, also return the app's dependency injection container.
      return_handle: If True, return a ServerHandle for programmatic control (block/stop).
      load_modules: List of Python modules to import before server start.
      load_paths: List of Python file paths to import before server start.
      project_root: Path to add to sys.path for module resolution during loading.
      strict_load: If True, raise on import/load errors; otherwise, record errors in loader report.

    Returns:
      str: The server URL (e.g., "http://127.0.0.1:53421").
      tuple: Optionally, (url, container), (url, handle), or (url, container, handle)
        depending on the flags set and whether the server was started in-process.

    Notes:
      - Workspace is a dedicated directory for server data, including logs, caches, and runtime state; multiple processes using the same
        workspace will coordinate to reuse a single server instance. Delete the workspace to reset state.
      - Use handle.block() to wait for server exit when you need to keep the server running in a script or notebook. This is typically not needed
        when using the server in client mode.
      - When you are using Aethergraph UI, use handle.block() to keep the server running so that the UI can connect to it and discover agents/apps.
    """
    global _started, _server_thread, _url, _uvicorn_server

    # In-process fast path
    if _started and _url:
        if return_container or return_handle:
            # We can return these because we're in-process
            # (container is attached to app only when we start it; see below)
            pass
        else:
            print(" - reusing existing in-process server at", _url)
            return _url

    print(" - acquiring workspace lock...")
    # Cross-process coordination: one workspace => one server
    with workspace_lock(workspace):
        running_url = get_running_url_if_any(workspace)
        if running_url:
            # Reuse the already-running sidecar for this workspace
            print(" - reusing existing sidecar server at", running_url)
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

        print(" Loading user graphs with spec:", spec)
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

        print("\n" + "=" * 50)
        print(f"[AetherGraph] 🚀 Server started at: {url}")
        print(f"[AetherGraph] 🖥️  UI:        {url}/ui (if built)")
        print(f"[AetherGraph] 📡 API:       {url}/api/v1/")
        print(f"[AetherGraph] 📂 Workspace:  {workspace}")
        print("=" * 50 + "\n")

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
