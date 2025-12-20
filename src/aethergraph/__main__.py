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
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(prog="aethergraph")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the AetherGraph sidecar (blocking).")
    serve.add_argument("--workspace", default="./aethergraph_data")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000, help="0 = auto free port")
    serve.add_argument("--log-level", default="warning")
    serve.add_argument("--uvicorn-log-level", default="warning")

    serve.add_argument(
        "--project-root", default=None, help="Added to sys.path while loading user graphs."
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

    args = parser.parse_args(argv)

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
            spec = LoadSpec(
                modules=list(args.load_module or []),
                paths=list(args.load_path or []),
                project_root=args.project_root,
                strict=bool(args.strict_load),
            )
            if spec.modules or spec.paths:
                report = loader.load(spec)
                # Optional: print load errors but still continue if not strict
                if report.errors and not args.strict_load:
                    for e in report.errors:
                        print(f"[load error] {e.source}: {e.error}")

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

        # Run blocking server (lock released so others can read server.json)
        print(url)
        uvicorn.run(app, host=args.host, port=port, log_level=args.uvicorn_log_level)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
