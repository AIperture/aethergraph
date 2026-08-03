from __future__ import annotations

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
