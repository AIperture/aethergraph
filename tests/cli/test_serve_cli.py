from __future__ import annotations

import argparse

from aethergraph.cli.commands import serve


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_reuse_returns_running_url(monkeypatch, capsys) -> None:
    args = argparse.Namespace(
        workspace="./aethergraph_workspace",
        host="127.0.0.1",
        port=8745,
        log_level="warning",
        uvicorn_log_level="info",
        project_root=".",
        load_module=[],
        load_path=[],
        strict_load=False,
        reuse=True,
        reload=False,
        reload_dir=[],
        reload_include=[],
        reload_exclude=[],
    )
    monkeypatch.setattr(
        "aethergraph.cli.commands.serve.workspace_lock", lambda workspace: _NullContext()
    )
    monkeypatch.setattr(
        "aethergraph.cli.commands.serve.get_running_url_if_any",
        lambda workspace: "http://127.0.0.1:8745",
    )
    assert serve.handle(args) == 0
    assert capsys.readouterr().out.strip() == "http://127.0.0.1:8745"


def test_default_reload_dirs_use_project_root_and_load_path_parents() -> None:
    args = argparse.Namespace(reload_dir=[])
    reload_dirs = serve._compute_reload_dirs(
        args,
        project_root=".",
        paths=["graphs/app.py", "graphs/nested/flow.py"],
    )
    assert reload_dirs == [".", "graphs", "graphs/nested"]
