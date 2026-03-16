# aethergraph/__main__.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
       python -m aethergraph serve --workspace ./aethergraph_workspace --port 0 \
         --project-root . \
         --load-path ./graphs.py

     Notes:
       - --port 0 auto-picks a free port and prints the resulting URL.
       - --load-path / --load-module imports user code BEFORE the server starts,
         so decorated graphs/apps/agents appear immediately in the UI.
       - --project-root is temporarily added to sys.path during loading (for local imports).
       - server.json is written under the workspace for discovery.

  2) Reuse detection (avoid starting multiple servers for the same workspace)
       python -m aethergraph serve --workspace ./aethergraph_workspace --reuse

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
        python -m aethergraph serve # only default agents/apps show up
        ```

        load user graphs from a file and autoreload on changes:
        ```bash
        python -m aethergraph serve --load-path ./graphs.py --reload --reload-include 'src/**/*.py'
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
        - `workspace`: Path to the workspace folder (default: ./aethergraph_workspace).
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
        - `reload-dir`: Additional directory to watch for auto-reload (repeatable). If not provided, defaults to project-root plus parents of --load-path.
        - `reload-include`: Glob pattern to include for auto-reload (repeatable).
        - `reload-exclude`: Glob pattern to exclude from auto-reload (repeatable).

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
    serve.add_argument("--workspace", default="./aethergraph_workspace")
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

    serve.add_argument(
        "--reload-dir",
        action="append",
        default=[],
        help=(
            "Additional directory to watch for auto-reload (repeatable). "
            "If not provided, defaults to project-root plus parents of --load-path."
        ),
    )
    serve.add_argument(
        "--reload-include",
        action="append",
        default=[],
        help=(
            "Glob pattern of files/dirs to include for auto-reload (repeatable). "
            "Example: --reload-include 'src/**/*.py'"
        ),
    )
    serve.add_argument(
        "--reload-exclude",
        action="append",
        default=[],
        help=(
            "Glob pattern of files/dirs to exclude from auto-reload (repeatable). "
            "Example: --reload-exclude 'aethergraph_workspace/**/*' --reload-exclude '*/__pycache__/*'"
        ),
    )

    # ---- run subcommand ----
    run_cmd = sub.add_parser(
        "run",
        help="Run a graph and print results.",
        description=(
            "Run a graph by name or by file path.\n\n"
            "Simple usage (registers file, runs on server, polls):\n"
            "  aethergraph run ./scripts/workflow.py\n"
            "  aethergraph run ./scripts/workflow.py --inputs '{...}'\n"
            "  aethergraph run ./scripts/workflow.py --graph my_graph\n\n"
            "Advanced usage (explicit graph name):\n"
            "  aethergraph run my_graph --via-api --poll\n"
            "  aethergraph run my_graph --load-path ./scripts/workflow.py --via-api\n"
            "  aethergraph run my_graph  # in-process, no server needed"
        ),
    )
    run_cmd.add_argument(
        "target",
        help=(
            "Graph name (e.g. 'my_graph') or path to a .py file "
            "(e.g. './scripts/workflow.py'). When a .py file is given, "
            "it is auto-registered with the server and the graph name "
            "is detected from the registration response."
        ),
    )
    run_cmd.add_argument(
        "--graph",
        default=None,
        help="Explicit graph name (useful when a .py file defines multiple graphs).",
    )
    run_cmd.add_argument(
        "--inputs", default="{}", help="JSON object of inputs to pass to the graph."
    )
    run_cmd.add_argument("--workspace", default="./aethergraph_workspace")
    run_cmd.add_argument("--project-root", default=".")
    run_cmd.add_argument(
        "--load-module", action="append", default=[], help="Module to import (repeatable)."
    )
    run_cmd.add_argument(
        "--load-path", action="append", default=[], help="Python file path to load (repeatable)."
    )
    run_cmd.add_argument("--strict-load", action="store_true")
    run_cmd.add_argument("--log-level", default="warning")
    run_cmd.add_argument(
        "--via-api",
        action="store_true",
        help="Submit via running server API instead of in-process.",
    )
    run_cmd.add_argument(
        "--poll",
        action="store_true",
        help="Poll run status until completion (only with --via-api).",
    )
    run_cmd.add_argument(
        "--no-poll",
        action="store_true",
        help="Disable auto-poll (only relevant for .py file targets).",
    )

    # ---- register subcommand ----
    register = sub.add_parser("register", help="Register a local graph source into registry.")
    register.add_argument("--workspace", default="./aethergraph_workspace")
    register.add_argument("--server-url", default=None)
    register.add_argument("--mode", choices=["auto", "api", "local"], default="auto")
    register.add_argument("--source", choices=["file", "artifact"], default="file")
    register.add_argument("--path", default=None, help="Path to Python file when --source=file.")
    register.add_argument("--artifact-id", default=None, help="Artifact id when --source=artifact.")
    register.add_argument("--uri", default=None, help="Artifact URI when --source=artifact.")
    register.add_argument("--app-config-json", default=None, help="JSON object for app config.")
    register.add_argument("--agent-config-json", default=None, help="JSON object for agent config.")
    register.add_argument("--org-id", default=None)
    register.add_argument("--user-id", default=None)
    register.add_argument("--client-id", default=None)
    register.add_argument("--no-persist", action="store_true")
    register.add_argument("--no-strict", action="store_true")

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

            # Permanently add project_root to sys.path so that hot-loaded
            # files (e.g. via /api/v1/registry/register) can resolve imports
            # like ``from scripts.foo import bar``.
            # TODO(cloud): This is safe for local/OSS (single user, single project)
            # but must NOT be used in cloud mode — see the TODO in
            # RegistrationService._register_source for the per-request approach.
            pr_str = str(Path(project_root).resolve())
            if pr_str not in sys.path:
                sys.path.insert(0, pr_str)

            spec = LoadSpec(
                modules=list(args.load_module or []),
                paths=list(args.load_path or []),
                project_root=args.project_root,
                strict=bool(args.strict_load),
            )

            # Export them to environment so the worker factory can read them
            os.environ["AETHERGRAPH_WORKSPACE"] = args.workspace
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

        log_path = Path(args.workspace) / "logs" / "aethergraph.log"
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
            print(f"[AetherGraph] 🧩 {'Log Path:':<18} {log_path}")
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
            print(f"[AetherGraph] 🧩 {'Log Path:':<18} {log_path}")
            print(f"[AetherGraph] ♻️  {'Auto-reload:':<18} enabled (uvicorn)")

            # --- reload dirs ---
            reload_dirs: list[str] = []

            if args.reload_dir:
                # User explicitly requested reload dirs -> trust them
                reload_dirs.extend(args.reload_dir)
            else:
                # Default behavior: project_root + parents of load-paths
                reload_dirs.append(str(project_root))
                for p in paths:
                    reload_dirs.append(str(Path(p).parent))

            # De-duplicate while preserving order
            seen = set()
            reload_dirs = [d for d in reload_dirs if not (d in seen or seen.add(d))]

            # --- include/exclude globs (None = use uvicorn defaults) ---
            reload_includes = args.reload_include or None
            reload_excludes = args.reload_exclude or None

            print(f"👀 Watching for changes in dirs: {reload_dirs}")
            print(f"👀 Auto-reload include patterns: {reload_includes or 'uvicorn defaults'}")
            print(f"👀 Auto-reload exclude patterns: {reload_excludes or 'uvicorn defaults'}")
            print("=" * 50 + "\n")

            uvicorn.run(
                "aethergraph.server.app_factory:create_app_from_env",
                host=args.host,
                port=port,
                log_level=args.uvicorn_log_level,
                reload=True,
                reload_dirs=reload_dirs,
                reload_includes=reload_includes,
                reload_excludes=reload_excludes,
                factory=True,
            )
            return 0

    if args.cmd == "run":
        inputs = json.loads(args.inputs)

        # --- Resolve target: .py file path vs graph name ---
        target = args.target
        is_file_target = target.endswith(".py") or Path(target).suffix == ".py"

        graph_id: str | None = args.graph  # explicit --graph overrides auto-detect
        via_api = args.via_api
        poll = args.poll

        if is_file_target:
            # Simple mode: file target implies --via-api and --poll by default
            via_api = True
            if not args.no_poll:
                poll = True
            # Add the file to load_path if not already there
            if target not in (args.load_path or []):
                args.load_path = list(args.load_path or []) + [target]

        if via_api:
            # --- Run via API (requires running server) ---
            base = get_running_url_if_any(args.workspace) or "http://127.0.0.1:8745"

            # Auto-register load-paths with the server before running
            for lp in args.load_path or []:
                reg_payload = json.dumps(
                    {
                        "source": "file",
                        "path": str(Path(lp).resolve()),
                        "persist": True,
                        "strict": False,
                    }
                ).encode("utf-8")
                reg_req = Request(
                    url=f"{base.rstrip('/')}/api/v1/registry/register",
                    data=reg_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(reg_req, timeout=20) as resp:  # noqa: S310
                        reg_result = json.loads(resp.read().decode("utf-8"))
                    if reg_result.get("success"):
                        detected_name = reg_result.get("graph_name")
                        print(f"Registered: {lp} -> graph={detected_name}")
                        # Auto-detect graph_id from file registration
                        if graph_id is None and detected_name:
                            graph_id = detected_name
                    else:
                        errs = reg_result.get("errors", [])
                        print(f"Registration warning for {lp}: {errs}", file=sys.stderr)
                except (HTTPError, URLError) as e:
                    detail = e.read().decode("utf-8") if isinstance(e, HTTPError) else str(e)
                    print(f"Failed to register {lp}: {detail}", file=sys.stderr)

            if not graph_id:
                if is_file_target:
                    print(
                        f"Error: could not detect graph name from {target}. "
                        "Use --graph <name> to specify explicitly.",
                        file=sys.stderr,
                    )
                else:
                    graph_id = target  # target is the graph name itself
            if not graph_id:
                return 1

            payload = json.dumps({"inputs": inputs, "origin": "cli"}).encode("utf-8")
            req = Request(
                url=f"{base.rstrip('/')}/api/v1/graphs/{graph_id}/runs",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=30) as resp:  # noqa: S310
                    result = json.loads(resp.read().decode("utf-8"))
            except (HTTPError, URLError) as e:
                detail = e.read().decode("utf-8") if isinstance(e, HTTPError) else str(e)
                print(f"Error submitting run: {detail}", file=sys.stderr)
                return 1

            run_id = result.get("run_id")
            print(f"Run submitted: {run_id}  status={result.get('status')}")

            if poll and run_id:
                print("Polling for completion...")
                while True:
                    time.sleep(1)
                    poll_req = Request(
                        url=f"{base.rstrip('/')}/api/v1/runs/{run_id}",
                        method="GET",
                    )
                    try:
                        with urlopen(poll_req, timeout=10) as resp:  # noqa: S310
                            summary = json.loads(resp.read().decode("utf-8"))
                    except (HTTPError, URLError):
                        continue
                    st = summary.get("status")
                    print(f"  status={st}")
                    if st in ("succeeded", "failed", "canceled"):
                        print(json.dumps(summary, indent=2, default=str))
                        break
            return 0

        # --- Run in-process (no server needed) ---
        # When target is a graph name (not a .py file)
        if graph_id is None:
            graph_id = target

        loader = GraphLoader()
        spec = LoadSpec(
            modules=list(args.load_module or []),
            paths=list(args.load_path or []),
            project_root=args.project_root,
            strict=bool(args.strict_load),
        )

        os.environ.setdefault("AETHERGRAPH_WORKSPACE", args.workspace)
        os.environ.setdefault("AETHERGRAPH_LOG_LEVEL", args.log_level)

        if spec.modules or spec.paths:
            report = loader.load(spec)
            if report.errors:
                for e in report.errors:
                    print(f"[load error] {e.source}: {e.error}", file=sys.stderr)
                    if e.traceback:
                        print(e.traceback, file=sys.stderr)
                if args.strict_load:
                    return 1

        async def _run_local() -> dict:
            from aethergraph.api.v1.deps import RequestIdentity
            from aethergraph.config.loader import load_settings
            from aethergraph.core.runtime.run_types import RunOrigin
            from aethergraph.core.runtime.runtime_services import install_services
            from aethergraph.services.container.default_container import build_default_container

            cfg = load_settings()
            container = build_default_container(root=args.workspace, cfg=cfg)
            install_services(container)
            rm = container.run_manager

            identity = RequestIdentity(user_id="local", org_id="local", mode="local")
            assert graph_id is not None  # guaranteed by target resolution above
            record, outputs, has_waits, continuations = await rm.run_and_wait(
                graph_id,
                inputs=inputs,
                identity=identity,
                origin=RunOrigin.cli,
            )
            return {
                "run_id": record.run_id,
                "graph_id": record.graph_id,
                "status": record.status.value
                if hasattr(record.status, "value")
                else str(record.status),
                "outputs": outputs,
                "has_waits": has_waits,
                "error": record.error,
                "started_at": str(record.started_at),
                "finished_at": str(record.finished_at),
            }

        try:
            result = asyncio.run(_run_local())
            print(json.dumps(result, indent=2, default=str))
            return 0 if result.get("status") == "succeeded" else 1
        except Exception as e:  # noqa: BLE001
            print(f"Run failed: {e}", file=sys.stderr)
            return 1

    if args.cmd == "register":
        app_config = json.loads(args.app_config_json) if args.app_config_json else None
        agent_config = json.loads(args.agent_config_json) if args.agent_config_json else None
        payload = {
            "source": args.source,
            "path": args.path,
            "artifact_id": args.artifact_id,
            "uri": args.uri,
            "app_config": app_config,
            "agent_config": agent_config,
            "persist": not bool(args.no_persist),
            "strict": not bool(args.no_strict),
        }
        headers = {"Content-Type": "application/json"}
        if args.user_id:
            headers["X-User-ID"] = args.user_id
        if args.org_id:
            headers["X-Org-ID"] = args.org_id
        if args.client_id:
            headers["X-Client-ID"] = args.client_id

        def _register_via_api() -> dict:
            base = (
                args.server_url or get_running_url_if_any(args.workspace) or "http://127.0.0.1:8745"
            )
            req = Request(
                url=f"{base.rstrip('/')}/api/v1/registry/register",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(req, timeout=20) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))

        async def _register_via_local() -> dict:
            from aethergraph.core.runtime.runtime_registry import current_registry
            from aethergraph.services.registry.registration_service import RegistrationService
            from aethergraph.storage.docstore.fs_doc import FSDocStore
            from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore

            docs = FSDocStore(root=str(Path(args.workspace) / "docs"))
            manifests = RegistrationManifestStore(doc_store=docs)
            service = RegistrationService(
                registry=current_registry(),
                manifest_store=manifests,
            )
            tenant = {"org_id": args.org_id, "user_id": args.user_id}
            if args.source == "file":
                if not args.path:
                    raise ValueError("--path is required for --source=file")
                result = await service.register_by_file(
                    args.path,
                    app_config=app_config,
                    agent_config=agent_config,
                    tenant=tenant,
                    persist=not bool(args.no_persist),
                    strict=not bool(args.no_strict),
                )
            else:
                result = await service.register_by_artifact(
                    artifact_id=args.artifact_id,
                    uri=args.uri,
                    app_config=app_config,
                    agent_config=agent_config,
                    tenant=tenant,
                    persist=not bool(args.no_persist),
                    strict=not bool(args.no_strict),
                )
            return RegistrationService.to_dict(result)

        if args.mode in {"api", "auto"}:
            try:
                out = _register_via_api()
                print(json.dumps(out, indent=2))
                return 0
            except HTTPError as e:
                detail = e.read().decode("utf-8")
                if args.mode == "api":
                    print(detail or str(e), file=sys.stderr)
                    return 1
            except URLError as e:
                if args.mode == "api":
                    print(str(e), file=sys.stderr)
                    return 1

        try:
            out = asyncio.run(_register_via_local())
            print(json.dumps(out, indent=2))
            return 0
        except Exception as e:  # noqa: BLE001
            print(str(e), file=sys.stderr)
            return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
