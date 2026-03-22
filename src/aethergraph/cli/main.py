from __future__ import annotations

import argparse
import sys

from aethergraph.cli.commands import register, run, serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aethergraph")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    serve.register_parser(subparsers)
    run.register_parser(subparsers)
    register.register_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
