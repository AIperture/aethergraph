from __future__ import annotations

import json
from types import SimpleNamespace

from aethergraph.cli.commands import host as host_command
from aethergraph.cli.main import build_parser


def test_host_command_requires_exact_launch_handles() -> None:
    args = build_parser().parse_args(
        [
            "host",
            "--manifest",
            "host-manifest.json",
            "--runtime-identity",
            "runtime-identity.json",
            "--settings",
            "application.env",
            "--workspace",
            "runtime",
            "--control-token",
            "control-token.handle",
        ]
    )

    assert args.manifest == "host-manifest.json"
    assert args.runtime_identity == "runtime-identity.json"
    assert args.provider_secrets is None


def test_host_command_classifies_and_reports_missing_dependency(
    monkeypatch,
    capsys,
) -> None:
    async def fail_with_missing_dependency(_args) -> int:
        raise ModuleNotFoundError("Compiled entrypoint requires missing Python module 'rapidfuzz'.")

    monkeypatch.setattr(host_command, "_run_host", fail_with_missing_dependency)

    assert host_command.handle(SimpleNamespace()) == 3
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["code"] == "host.missing_dependency"
    assert "rapidfuzz" in diagnostic["detail"]
