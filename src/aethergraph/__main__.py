# aethergraph/__main__.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import uvicorn

from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.server.app_factory import create_app
from aethergraph.server.loading import GraphLoader, LoadSpec
from aethergraph.server.server_state import (
    get_running_url_if_any,
    pick_free_port,
    workspace_lock,
    write_server_state,
)

"""
AetherGraph CLI (Phase 1)

Goal: run the sidecar persistently as a long-lived process.

Why:
  - Your workspace stores persistent data (runs/artifacts/memory/sessions).
  - The server process must stay alive for the frontend/Electron to call the API repeatedly.
  - When port=0 (auto free port), the actual URL changes per start.
    We write workspace/.aethergraph/server.json so the UI can discover the URL without
    hardcoding ports or parsing stdout.

Commands:

  1) Start the sidecar (blocking, recommended for "always-on" local server)
       python -m aethergraph serve --workspace ./aethergraph_data --port 0 \
         --project-root . \
         --load-path ./graphs.py

     Notes:
       - --port 0 auto-picks a free port and prints the resulting URL.
       - --load-path / --load-module imports user code BEFORE the server starts,
         so decorated graphs/apps/agents appear immediately in the UI.
       - --project-root is temporarily added to sys.path during loading (for local imports).
       - server.json is written under the workspace for discovery.

  2) Reuse detection (avoid starting multiple servers for the same workspace)
       python -m aethergraph serve --workspace ./aethergraph_data --reuse

     Behavior:
       - If a server for this workspace is already running, print its URL and exit 0.
       - If not running, starts a new server.

Recommended desktop/Electron workflow:
  - Electron chooses a workspace folder.
  - Electron checks workspace/.aethergraph/server.json and tries to connect.
  - If missing/dead, Electron spawns:
      python -m aethergraph serve --workspace <workspace> --port 0 --load-path <graphs.py> ...
  - Electron reads server.json to get the URL and connects.
"""


def main(argv: list[str] | None = None) -> int:
    """
    Start the AetherGraph server via CLI.

    This entrypoint launches the persistent sidecar server for your workspace,
    enabling API access for frontend/UI clients. It supports automatic
    port selection, workspace isolation, and dynamic loading of user graphs/apps.

    Examples:
        Basic usage with default workspace and port:
        ```bash
        python -m aethergraph serve --workspace # only default agents/apps show up
        ```

        load user graphs from a file and autoreload on changes:
        ```bash
        python -m aethergraph serve --load-path ./graphs.py --reload
        ```

        Load multiple modules and set a custom project root:
        ```bash
        python -m aethergraph serve --load-module mygraphs --project-root .
        ```

        Reuse detection (print URL if already running):
        ```bash
        python -m aethergraph serve --reuse
        ```

        Customize workspace and port:
        ```bash
        python -m aethergraph serve --workspace ./my_workspace --port 8000  # this will not show previous runs/artifacts unless reused
        ```

    Args:
        argv: Optional list of CLI arguments. If None, uses sys.argv[1:].

    Required keywords:
        - `serve`: Command to start the AetherGraph server. If no other command is given, the server will only load default built-in agents/apps.

    Optional keywords:
        - `workspace`: Path to the workspace folder (default: ./aethergraph_data).
        - `host`: Host address to bind (default: 127.0.0.1).
        - `port`: Port to bind (default: 8745; use 0 for auto-pick).
        - `log-level`: App log level (default: warning).
        - `uvicorn-log-level`: Uvicorn log level (default: warning).
        - `project-root`: Temporarily added to sys.path for local imports.
        - `load-module`: Python module(s) to import before server starts (repeatable).
        - `load-path`: Python file(s) to load before server starts (repeatable).
        - `strict-load`: Raise error if graph loading fails.
        - `reuse`: If server already running for workspace, print URL and exit.
        - `reload`: Enable auto-reload (dev mode).

    Returns:
        int: Exit code (0 for success, 2 for unknown command).

    Notes:
        - Launching the server via CLI keeps it running persistently for API clients to connect like AetherGraph UI.
        - In local mode, the server port will automatically be consistent with UI connections.
        - use `--reload` for development to auto-restart on code changes. This will use uvicorn's reload feature.
        - When switching ports, the UI will not show previous runs/artifacts unless the server is reused. This is
            because the server URL is tied to the frontend hash. Keep the server in a same port (default 8745) for local dev.
            Later the UI can support dynamic port discovery via server.json.
    """
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(prog="aethergraph")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the AetherGraph sidecar (blocking).")
    serve.add_argument("--workspace", default="./aethergraph_data")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8745, help="0 = auto free port")
    serve.add_argument("--log-level", default="warning")
    serve.add_argument("--uvicorn-log-level", default="info")

    serve.add_argument(
        "--project-root",
        default=".",
        help="Root directory for the project. Added to sys.path while loading user graphs.",
    )
    serve.add_argument(
        "--load-module", action="append", default=[], help="Module to import (repeatable)."
    )
    serve.add_argument(
        "--load-path", action="append", default=[], help="Python file path to load (repeatable)."
    )
    serve.add_argument("--strict-load", action="store_true", help="Raise if graph loading fails.")

    serve.add_argument(
        "--reuse",
        action="store_true",
        help="If server already running for workspace, print URL and exit 0.",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (dev only).",
    )

    args = parser.parse_args(argv)
    print(args)

    if args.cmd == "serve":
        loader = GraphLoader()

        # Ensure one workspace => one server process
        with workspace_lock(args.workspace):
            running = get_running_url_if_any(args.workspace)
            if running:
                if args.reuse:
                    print(running)
                    return 0
                print(f"Already running for workspace: {running}")
                return 0

            # Load graphs BEFORE app starts
            project_root = args.project_root
            modules = list(args.load_module or [])
            paths = list(args.load_path or [])
            spec = LoadSpec(
                modules=list(args.load_module or []),
                paths=list(args.load_path or []),
                project_root=args.project_root,
                strict=bool(args.strict_load),
            )

            # Export them to environment so the worker factory can read them
            os.environ["AETHERGRAPH_WORKSPACE"] = args.workspace
            os.environ.setdefault(
                "AETHERGRAPH_ROOT", args.workspace
            )  # AETHERGRAPH_ROOT is the workspace root in env
            os.environ["AETHERGRAPH_PROJECT_ROOT"] = str(project_root)
            os.environ["AETHERGRAPH_LOAD_MODULES"] = ",".join(modules)
            os.environ["AETHERGRAPH_LOAD_PATHS"] = os.pathsep.join(paths)
            os.environ["AETHERGRAPH_STRICT_LOAD"] = "1" if args.strict_load else "0"
            os.environ["AETHERGRAPH_LOG_LEVEL"] = args.log_level

            print("=" * 50)
            print("🔄 Loading graphs and agents...")
            if spec.modules or spec.paths:
                print(
                    "➕ Importing modules:",
                    spec.modules,
                    "and paths:",
                    spec.paths,
                    "at project root:",
                    spec.project_root,
                )
                report = loader.load(spec)
                # Optional: print load errors but still continue if not strict
                if report.errors and not args.strict_load:
                    for e in report.errors:
                        print(f"⚠️ [load error]  {e.source}: {e.error}")
                        print("   (continuing despite load error; use --strict-load to fail)")
                        if e.traceback:
                            print(e.traceback)
            print("✅ Graph/agents loading complete.")
            print("=" * 50)

            cfg = load_settings()
            set_current_settings(cfg)

            app = create_app(workspace=args.workspace, cfg=cfg, log_level=args.log_level)
            app.state.last_load_report = getattr(loader, "last_report", None)

            port = pick_free_port(int(args.port))
            url = f"http://{args.host}:{port}"

            # Write discovery file while we still hold the lock
            write_server_state(
                args.workspace,
                {
                    "pid": os.getpid(),
                    "host": args.host,
                    "port": port,
                    "url": url,
                    "workspace": str(Path(args.workspace).resolve()),
                    "started_at": time.time(),
                },
            )

        if not args.reload:
            # Run blocking server (lock released so others can read server.json)
            print("\n" + "=" * 50)
            # We align the labels to 18 characters (the length of the longest label)
            print(f"[AetherGraph] 🚀 {'Server started at:':<18} {url}")
            print(
                f"[AetherGraph] 🖥️  {'UI:':<18} {url}/ui   (if built)"
            )  # strangly, this needs two spaces unlike the rest
            print(f"[AetherGraph] 📡 {'API:':<18} {url}/api/v1/")
            print(f"[AetherGraph] 📂 {'Workspace:':<18} {args.workspace}")
            print("=" * 50 + "\n")
            uvicorn.run(
                app,
                host=args.host,
                port=port,
                log_level=args.uvicorn_log_level,
            )
            return 0

        # When --reload is on:
        if args.reload:
            print("\n" + "=" * 50)
            print(f"[AetherGraph] 🚀 {'Server started at:':<18} {url}")
            print(f"[AetherGraph] 🖥️  {'UI:':<18} {url}/ui   (if built)")
            print(f"[AetherGraph] 📡 {'API:':<18} {url}/api/v1/")
            print(f"[AetherGraph] 📂 {'Workspace:':<18} {args.workspace}")
            print(f"[AetherGraph] ♻️  {'Auto-reload:':<18} enabled (uvicorn)")
            print("=" * 50 + "\n")

            reload_dirs: list[str] = [str(project_root)]
            for p in paths:
                reload_dirs.append(str(Path(p).parent))

            # Use import string + factory=True here
            uvicorn.run(
                "aethergraph.server.app_factory:create_app_from_env",
                host=args.host,
                port=port,
                log_level=args.uvicorn_log_level,
                reload=True,
                reload_dirs=reload_dirs,
                factory=True,
            )
            return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
