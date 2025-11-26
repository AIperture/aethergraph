# aethergraph/server.py
from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.server.app_factory import create_app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aethergraph-server",
        description="Run the AetherGraph HTTP/WS server.",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--workspace",
        default="./aethergraph_data",
        help="Workspace directory for AG data (default: ./aethergraph_data).",
    )
    parser.add_argument(
        "--log-level",
        dest="app_log_level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Application log level (default: info).",
    )
    parser.add_argument(
        "--uvicorn-log-level",
        dest="uvicorn_log_level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point for running AetherGraph as a long-lived server.

    Example:
        python -m aethergraph.server --host 0.0.0.0 --port 8000
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1) Load and install settings (same as sidecar)
    cfg = load_settings()
    set_current_settings(cfg)

    # 2) Build the FastAPI app with your existing factory
    app = create_app(
        workspace=args.workspace,
        cfg=cfg,
        log_level=args.app_log_level,
    )

    # 3) Run uvicorn in this process (no threads, daemon-style)
    #    This blocks until the server is stopped.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
