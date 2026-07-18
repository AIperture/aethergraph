from __future__ import annotations

import json
from pathlib import Path

from aethergraph.cli import main as cli_main
from aethergraph.cli.commands import observability


def test_observability_legacy_defaults_to_read_only_report(
    tmp_path: Path,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    exit_code = cli_main.main(["observability", "legacy", "--workspace", str(workspace)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["report"]["candidate_bytes"] == 0


def test_observability_archive_requires_explicit_apply(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    exit_code = cli_main.main(
        [
            "observability",
            "legacy",
            "--workspace",
            str(workspace),
            "--archive-dir",
            str(tmp_path / "archive"),
        ]
    )

    assert exit_code == 2
    assert "--archive-dir requires --apply" in capsys.readouterr().err


def test_observability_legacy_parser_binds_one_cleanup_handler() -> None:
    args = cli_main.build_parser().parse_args(["observability", "legacy"])
    assert args.handler is observability.handle_legacy
