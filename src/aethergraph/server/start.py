# aethergraph/start.py
from __future__ import annotations
import asyncio, threading, socket, contextlib
from typing import Optional
import uvicorn

from .app_factory import create_app
from aethergraph.config.config import AppSettings
from aethergraph.config.loader import load_settings
from aethergraph.config.context import set_current_settings
from ..plugins.channel.routes.webui_routes import install_web_channel, webui_router

_started = False
_server_thread: Optional[threading.Thread] = None
_shutdown_flag = threading.Event()
_url: Optional[str] = None

def _pick_free_port(p: int) -> int:
    if p:
        return p
    s = socket.socket(); s.bind(("", 0))
    port = s.getsockname()[1]; s.close()
    return port

def _run_uvicorn_in_thread(app, host: str, port: int, log_level: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=log_level, loop="asyncio"))

    async def runner():
        task = asyncio.create_task(server.serve())
        while not _shutdown_flag.is_set():
            await asyncio.sleep(0.2)
        if not server.should_exit:
            server.should_exit = True
        await task

    try:
        loop.run_until_complete(runner())
    finally:
        loop.stop()
        loop.close()

def start_server(
    *,
    workspace: str = "./aethergraph_data",
    host: str = "127.0.0.1",
    port: int = 8000,                              # 0 = auto free port
    log_level: str = "warning",
    unvicorn_log_level: str = "warning",
    return_container: bool = False,
) -> str:
    """
    Start the AetherGraph sidecar server in a background thread and install
    services using the given workspace. Safe to call at top of any script
    or notebook cell (no main() wrapper needed). Returns base URL.
    """
    global _started, _server_thread, _url
    if _started:
        return _url  # type: ignore

    # Build app (installs services inside create_app)
    cfg = load_settings()
    set_current_settings(cfg)

    app = create_app(
        workspace=workspace,
        cfg=cfg,
        log_level=log_level
    )

    picked_port = _pick_free_port(port)
    t = threading.Thread(
        target=_run_uvicorn_in_thread,
        args=(app, host, picked_port, unvicorn_log_level),
        name="aethergraph-sidecar",
        daemon=True,
    )
    t.start()

    _server_thread = t
    _started = True
    _url = f"http://{host}:{picked_port}"

    install_web_channel(app)

    if return_container:
        return _url, app.state.container
    return _url


async def start_server_async(**kw) -> str:
    # Async-friendly wrapper; still uses a thread to avoid clashing with caller loop
    return start_server(**kw)

def stop_server():
    """Optional: stop the background server (useful in tests)."""
    global _started, _server_thread, _url
    if not _started:
        return
    _shutdown_flag.set()
    if _server_thread and _server_thread.is_alive():
        with contextlib.suppress(Exception):
            _server_thread.join(timeout=5)
    _started = False
    _server_thread = None
    _url = None
    _shutdown_flag.clear()


# backward compatibility
start = start_server
stop = stop_server
start_async = start_server_async