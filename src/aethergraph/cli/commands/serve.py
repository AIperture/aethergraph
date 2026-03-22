from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import uvicorn

from aethergraph.cli import common, load, output
from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.server.app_factory import create_app
from aethergraph.server.loading import GraphLoader
from aethergraph.server.server_state import (
    get_running_url_if_any,
    pick_free_port,
    workspace_lock,
    write_server_state,
)


def register_parser(subparsers) -> None:
    serve = subparsers.add_parser("serve", help="Run the AetherGraph sidecar (blocking).")
    common.add_workspace_argument(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8745, help="0 = auto free port")
    common.add_log_level_argument(serve)
    serve.add_argument("--uvicorn-log-level", default="info")
    serve.add_argument(
        "--project-root",
        default=".",
        help="Root directory for the project. Added to sys.path while loading user graphs.",
    )
    common.add_load_module_argument(serve)
    common.add_load_path_argument(serve)
    common.add_strict_load_argument(serve)
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
    serve.set_defaults(handler=handle)


def _compute_reload_dirs(args, *, project_root: str, paths: list[str]) -> list[str]:
    if args.reload_dir:
        reload_dirs = list(args.reload_dir)
    else:
        reload_dirs = [str(project_root), *[Path(p).parent.as_posix() for p in paths]]
    seen: set[str] = set()
    return [d for d in reload_dirs if not (d in seen or seen.add(d))]


def handle(args: argparse.Namespace) -> int:
    loader = GraphLoader()

    with workspace_lock(args.workspace):
        running = get_running_url_if_any(args.workspace)
        if running:
            if args.reuse:
                print(running)
                return 0
            print(f"Already running for workspace: {running}")
            return 0

        project_root = args.project_root
        modules = list(args.load_module or [])
        paths = list(args.load_path or [])
        load.prepare_project_root(project_root)

        spec = load.make_load_spec(args)
        load.export_load_environment(
            workspace=args.workspace,
            project_root=project_root,
            modules=modules,
            paths=paths,
            strict_load=bool(args.strict_load),
            log_level=args.log_level,
        )

        output.print_load_intro(spec)
        if spec.modules or spec.paths:
            report = load.load_graphs(loader, spec)
            if report.errors:
                output.print_load_errors(report.errors, strict_load=bool(args.strict_load))
                if args.strict_load:
                    return 1
        output.print_load_complete()

        cfg = load_settings()
        set_current_settings(cfg)

        app = create_app(workspace=args.workspace, cfg=cfg, log_level=args.log_level)
        app.state.last_load_report = getattr(loader, "last_report", None)

        port = pick_free_port(int(args.port))
        url = f"http://{args.host}:{port}"
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
    output.print_server_banner(
        url=url,
        workspace=args.workspace,
        log_path=log_path,
        reload_enabled=bool(args.reload),
    )

    if not args.reload:
        print("=" * 50 + "\n")
        uvicorn.run(
            app,
            host=args.host,
            port=port,
            log_level=args.uvicorn_log_level,
        )
        return 0

    reload_dirs = _compute_reload_dirs(args, project_root=project_root, paths=paths)
    reload_includes = args.reload_include or None
    reload_excludes = args.reload_exclude or None
    output.print_reload_watch_config(reload_dirs, reload_includes, reload_excludes)
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
