from __future__ import annotations

import argparse

from aethergraph.cli.commands import run


def _args(**overrides) -> argparse.Namespace:
    base = {
        "target": "my_graph",
        "graph": None,
        "inputs": "{}",
        "workspace": "./aethergraph_workspace",
        "project_root": ".",
        "load_module": [],
        "load_path": [],
        "strict_load": False,
        "log_level": "warning",
        "via_api": False,
        "poll": False,
        "no_poll": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_file_target_forces_api_and_auto_poll() -> None:
    args = _args(target="./scripts/workflow.py")
    is_file_target, graph_id, via_api, poll = run._resolve_run_mode(args)
    assert is_file_target is True
    assert graph_id is None
    assert via_api is True
    assert poll is True
    assert args.load_path == ["./scripts/workflow.py"]


def test_graph_name_target_stays_local_by_default() -> None:
    args = _args(target="my_graph")
    is_file_target, graph_id, via_api, poll = run._resolve_run_mode(args)
    assert is_file_target is False
    assert graph_id is None
    assert via_api is False
    assert poll is False


def test_missing_detected_graph_name_returns_failure(capsys, monkeypatch) -> None:
    args = _args(target="./scripts/workflow.py")
    monkeypatch.setattr(
        "aethergraph.cli.commands.run._register_load_paths",
        lambda base, load_paths, graph_id: None,
    )
    monkeypatch.setattr(
        "aethergraph.cli.commands.run.http.resolve_server_base_url",
        lambda workspace: "http://127.0.0.1:8745",
    )
    assert run.handle(args) == 1
    assert "could not detect graph name" in capsys.readouterr().err
