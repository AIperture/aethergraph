from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError

from aethergraph.cli import common, http, load, output
from aethergraph.server.loading import GraphLoader


def register_parser(subparsers) -> None:
    run_cmd = subparsers.add_parser(
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
    common.add_common_load_arguments(run_cmd)
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
    run_cmd.set_defaults(handler=handle)


def _resolve_run_mode(args: argparse.Namespace) -> tuple[bool, str | None, bool, bool]:
    target = args.target
    is_file_target = target.endswith(".py") or Path(target).suffix == ".py"
    graph_id = args.graph
    via_api = bool(args.via_api)
    poll = bool(args.poll)

    if is_file_target:
        via_api = True
        if not args.no_poll:
            poll = True
        if target not in (args.load_path or []):
            args.load_path = list(args.load_path or []) + [target]

    return is_file_target, graph_id, via_api, poll


def _register_load_paths(base: str, load_paths: list[str], graph_id: str | None) -> str | None:
    for load_path in load_paths:
        try:
            reg_result = http.post_json(
                f"{base.rstrip('/')}/api/v1/registry/register",
                {
                    "source": "file",
                    "path": str(Path(load_path).resolve()),
                    "persist": True,
                    "strict": False,
                },
            )
        except (HTTPError, URLError) as exc:
            print(f"Failed to register {load_path}: {http.format_http_error(exc)}", file=sys.stderr)
            continue

        if reg_result.get("success"):
            detected_name = reg_result.get("graph_name")
            print(f"Registered: {load_path} -> graph={detected_name}")
            if graph_id is None and detected_name:
                graph_id = detected_name
        else:
            print(
                f"Registration warning for {load_path}: {reg_result.get('errors', [])}",
                file=sys.stderr,
            )

    return graph_id


def _submit_api_run(base: str, graph_id: str, inputs: dict) -> dict:
    return http.post_json(
        f"{base.rstrip('/')}/api/v1/graphs/{graph_id}/runs",
        {"inputs": inputs, "origin": "cli"},
        timeout=30,
    )


def _poll_run(base: str, run_id: str) -> None:
    print("Polling for completion...")
    while True:
        time.sleep(1)
        try:
            summary = http.get_json(f"{base.rstrip('/')}/api/v1/runs/{run_id}", timeout=10)
        except (HTTPError, URLError):
            continue
        status = summary.get("status")
        print(f"  status={status}")
        if status in ("succeeded", "failed", "canceled"):
            output.print_json(summary)
            break


async def _run_local_async(*, graph_id: str, inputs: dict, workspace: str) -> dict:
    from aethergraph.api.v1.deps import RequestIdentity
    from aethergraph.config.loader import load_settings
    from aethergraph.core.runtime.run_types import RunOrigin
    from aethergraph.core.runtime.runtime_services import install_services
    from aethergraph.services.container.default_container import build_default_container

    cfg = load_settings()
    container = build_default_container(root=workspace, cfg=cfg)
    install_services(container)
    rm = container.run_manager

    identity = RequestIdentity(user_id="local", org_id="local", mode="local")
    record, outputs, has_waits, continuations = await rm.run_and_wait(
        graph_id,
        inputs=inputs,
        identity=identity,
        origin=RunOrigin.cli,
    )
    return {
        "run_id": record.run_id,
        "graph_id": record.graph_id,
        "status": record.status.value if hasattr(record.status, "value") else str(record.status),
        "outputs": outputs,
        "has_waits": has_waits,
        "error": record.error,
        "started_at": str(record.started_at),
        "finished_at": str(record.finished_at),
    }


def _run_local(args: argparse.Namespace, *, graph_id: str, inputs: dict) -> int:
    loader = GraphLoader()
    spec = load.make_load_spec(args)

    os.environ.setdefault("AETHERGRAPH_WORKSPACE", args.workspace)
    os.environ.setdefault("AETHERGRAPH_LOG_LEVEL", args.log_level)

    if spec.modules or spec.paths:
        report = load.load_graphs(loader, spec)
        if report.errors:
            output.print_load_errors(report.errors, strict_load=bool(args.strict_load))
            if args.strict_load:
                return 1

    try:
        result = asyncio.run(
            _run_local_async(graph_id=graph_id, inputs=inputs, workspace=args.workspace)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    output.print_json(result)
    return 0 if result.get("status") == "succeeded" else 1


def handle(args: argparse.Namespace) -> int:
    inputs = json.loads(args.inputs)
    is_file_target, graph_id, via_api, poll = _resolve_run_mode(args)

    if via_api:
        base = http.resolve_server_base_url(workspace=args.workspace)
        graph_id = _register_load_paths(base, list(args.load_path or []), graph_id)

        if not graph_id:
            if is_file_target:
                print(
                    f"Error: could not detect graph name from {args.target}. Use --graph <name> to specify explicitly.",
                    file=sys.stderr,
                )
            else:
                graph_id = args.target
        if not graph_id:
            return 1

        try:
            result = _submit_api_run(base, graph_id, inputs)
        except (HTTPError, URLError) as exc:
            print(f"Error submitting run: {http.format_http_error(exc)}", file=sys.stderr)
            return 1

        run_id = result.get("run_id")
        print(f"Run submitted: {run_id}  status={result.get('status')}")
        if poll and run_id:
            _poll_run(base, run_id)
        return 0

    if graph_id is None:
        graph_id = args.target
    return _run_local(args, graph_id=graph_id, inputs=inputs)
