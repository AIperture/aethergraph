from __future__ import annotations

import argparse


def add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default="./aethergraph_workspace")


def add_project_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")


def add_load_module_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--load-module",
        action="append",
        default=[],
        help="Module to import (repeatable).",
    )


def add_load_path_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--load-path",
        action="append",
        default=[],
        help="Python file path to load (repeatable).",
    )


def add_strict_load_argument(
    parser: argparse.ArgumentParser, *, help_text: str | None = None
) -> None:
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help=help_text or "Raise if graph loading fails.",
    )


def add_log_level_argument(parser: argparse.ArgumentParser, *, default: str = "warning") -> None:
    parser.add_argument("--log-level", default=default)


def add_common_load_arguments(parser: argparse.ArgumentParser) -> None:
    add_workspace_argument(parser)
    add_project_root_argument(parser)
    add_load_module_argument(parser)
    add_load_path_argument(parser)
    add_strict_load_argument(parser)
    add_log_level_argument(parser)
