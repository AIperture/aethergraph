from __future__ import annotations

import argparse
from inspect import getdoc
from pathlib import Path
from urllib.error import URLError

from aethergraph.cli.commands import register


def _args(**overrides) -> argparse.Namespace:
    base = {
        "workspace": "./aethergraph_workspace",
        "server_url": None,
        "source": "file",
        "path": "./workflow.py",
        "artifact_id": None,
        "uri": None,
        "app_config_json": None,
        "agent_config_json": None,
        "org_id": None,
        "user_id": None,
        "client_id": None,
        "no_persist": False,
        "no_strict": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_payload_includes_identity_headers() -> None:
    payload, headers = register._build_payload(
        _args(org_id="org-1", user_id="user-1", client_id="client-1")
    )
    assert payload["source"] == "file"
    assert headers == {
        "X-User-ID": "user-1",
        "X-Org-ID": "org-1",
        "X-Client-ID": "client-1",
    }


def test_transport_failure_fails_directly_without_local_fallback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "aethergraph.cli.commands.register._register_via_api",
        lambda args, payload, headers: (_ for _ in ()).throw(URLError("boom")),
    )

    assert register.handle(_args()) == 1
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert captured.out == ""


def test_register_parser_has_no_storage_fallback_mode() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register.register_parser(subparsers)

    args = parser.parse_args(["register", "--path", "workflow.py"])

    assert not hasattr(args, "mode")


def test_register_command_has_no_local_store_path_and_strict_public_docstrings() -> None:
    source = Path(register.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "_register_via_local",
        "FSDocStore",
        "RegistrationManifestStore",
        'choices=["auto", "api", "local"]',
        "asyncio.run",
    ):
        assert forbidden not in source
    for function in (register.register_parser, register.handle):
        doc = getdoc(function) or ""
        assert doc.splitlines()[0]
        assert "Intro:" in doc
        assert doc.count("```python") >= 2
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Notes:" in doc
