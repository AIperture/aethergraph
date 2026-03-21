from __future__ import annotations

import argparse
from urllib.error import URLError

from aethergraph.cli.commands import register


def _args(**overrides) -> argparse.Namespace:
    base = {
        "workspace": "./aethergraph_workspace",
        "server_url": None,
        "mode": "auto",
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


def test_mode_auto_falls_back_to_local(monkeypatch, capsys) -> None:
    async def fake_register_via_local(args, *, payload):
        return {"success": True, "graph_name": "demo"}

    monkeypatch.setattr(
        "aethergraph.cli.commands.register._register_via_api",
        lambda args, payload, headers: (_ for _ in ()).throw(URLError("boom")),
    )
    monkeypatch.setattr(
        "aethergraph.cli.commands.register._register_via_local", fake_register_via_local
    )
    assert register.handle(_args(mode="auto")) == 0
    assert '"success": true' in capsys.readouterr().out.lower()
