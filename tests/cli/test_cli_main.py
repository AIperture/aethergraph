from __future__ import annotations

import argparse

import pytest

from aethergraph import __main__ as package_main
from aethergraph.cli import main as cli_main
from aethergraph.cli.commands import register, run, serve


def test_build_parser_exposes_current_top_level_commands() -> None:
    parser = cli_main.build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert {"serve", "run", "register"} <= set(subparsers_action.choices)


@pytest.mark.parametrize(
    ("argv", "expected_handler"),
    [
        (["serve"], serve.handle),
        (["run", "my_graph"], run.handle),
        (["register"], register.handle),
    ],
)
def test_subcommands_bind_handlers(argv: list[str], expected_handler) -> None:
    parser = cli_main.build_parser()
    args = parser.parse_args(argv)
    assert args.handler is expected_handler


def test_cli_main_dispatches_to_bound_handler(monkeypatch) -> None:
    parser = cli_main.build_parser()
    args = parser.parse_args(["serve"])
    assert args.handler is serve.handle

    monkeypatch.setattr(serve, "handle", lambda parsed_args: 12)
    assert cli_main.main(["serve"]) == 12


def test_package_main_delegates_to_cli_main(monkeypatch) -> None:
    assert package_main.main is cli_main.main
