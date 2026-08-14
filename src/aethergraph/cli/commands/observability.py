from __future__ import annotations

import argparse
import sys

from aethergraph.cli import output
from aethergraph.observability.legacy_cleanup import cleanup_legacy_observability


def register_parser(subparsers) -> None:
    observability = subparsers.add_parser(
        "observability",
        help="Run explicit observability administration.",
    )
    actions = observability.add_subparsers(dest="observability_cmd", required=True)
    legacy = actions.add_parser(
        "legacy",
        help="Report or remove unsupported pre-v2 observability data.",
    )
    legacy.add_argument("--workspace", default="./aethergraph_workspace")
    legacy.add_argument(
        "--apply",
        action="store_true",
        help="Execute the reported cleanup. Without this flag the command is read-only.",
    )
    legacy.add_argument(
        "--archive-dir",
        default=None,
        help="Empty directory outside the workspace to archive candidates before cleanup.",
    )
    legacy.set_defaults(handler=handle_legacy)


def handle_legacy(args: argparse.Namespace) -> int:
    if args.archive_dir and not args.apply:
        print("--archive-dir requires --apply", file=sys.stderr)
        return 2
    try:
        result = cleanup_legacy_observability(
            args.workspace,
            apply=bool(args.apply),
            archive_dir=args.archive_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    output.print_json(result.to_dict())
    return 0
