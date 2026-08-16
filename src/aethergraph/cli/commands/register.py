from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError

from aethergraph.cli import http, output


def register_parser(subparsers) -> None:
    """Register the clean-cut API-only graph-registration command.

    Intro:
        Adds one parser whose persistence boundary is always the running AG registry
        API; the command has no local store opener or transport-failure fallback.

    Examples:
        Attach registration to a CLI parser:
        ```python
        subparsers = parser.add_subparsers(dest="cmd")
        register_parser(subparsers)
        ```

        Parse an explicit server request:
        ```python
        args = parser.parse_args(["register", "--server-url", "http://127.0.0.1:8000"])
        ```

    Args:
        subparsers: Parent argparse subparser collection.

    Returns:
        None: The API-only `register` command was attached.

    Notes:
        Local registration requires a local AG server; filesystem fallback is
        intentionally absent under the storage-provider clean cut.
    """
    register = subparsers.add_parser(
        "register", help="Register a local graph source into registry."
    )
    register.add_argument("--workspace", default="./aethergraph_workspace")
    register.add_argument("--server-url", default=None)
    register.add_argument("--source", choices=["file", "artifact"], default="file")
    register.add_argument("--path", default=None, help="Path to Python file when --source=file.")
    register.add_argument("--artifact-id", default=None, help="Artifact id when --source=artifact.")
    register.add_argument("--uri", default=None, help="Artifact URI when --source=artifact.")
    register.add_argument("--app-config-json", default=None, help="JSON object for app config.")
    register.add_argument("--agent-config-json", default=None, help="JSON object for agent config.")
    register.add_argument("--org-id", default=None)
    register.add_argument("--user-id", default=None)
    register.add_argument("--client-id", default=None)
    register.add_argument("--no-persist", action="store_true")
    register.add_argument("--no-strict", action="store_true")
    register.set_defaults(handler=handle)


def _build_payload(args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    app_config = json.loads(args.app_config_json) if args.app_config_json else None
    agent_config = json.loads(args.agent_config_json) if args.agent_config_json else None
    payload = {
        "source": args.source,
        "path": args.path,
        "artifact_id": args.artifact_id,
        "uri": args.uri,
        "app_config": app_config,
        "agent_config": agent_config,
        "persist": not bool(args.no_persist),
        "strict": not bool(args.no_strict),
    }
    headers: dict[str, str] = {}
    if args.user_id:
        headers["X-User-ID"] = args.user_id
    if args.org_id:
        headers["X-Org-ID"] = args.org_id
    if args.client_id:
        headers["X-Client-ID"] = args.client_id
    return payload, headers


def _register_via_api(args: argparse.Namespace, payload: dict, headers: dict[str, str]) -> dict:
    base = http.resolve_server_base_url(workspace=args.workspace, server_url=args.server_url)
    return http.post_json(
        f"{base.rstrip('/')}/api/v1/registry/register",
        payload,
        headers=headers,
    )


def handle(args: argparse.Namespace) -> int:
    """Submit one graph-registration request through the exact AG API.

    Intro:
        Builds the frozen request payload and sends it to the explicitly resolved AG
        server, returning failure directly when transport or HTTP handling fails.

    Examples:
        Register a file through a local server:
        ```python
        exit_code = handle(args)
        ```

        Propagate an unavailable server as a nonzero result:
        ```python
        assert handle(unreachable_args) == 1
        ```

    Args:
        args: Parsed registration source, identity, workspace, and server options.

    Returns:
        int: Zero on a successful API response; one on HTTP or transport failure.

    Notes:
        The command never opens workspace storage and never falls back to a local
        registry implementation.
    """
    payload, headers = _build_payload(args)

    try:
        out = _register_via_api(args, payload, headers)
    except HTTPError as exc:
        print(http.format_http_error(exc) or str(exc), file=sys.stderr)
        return 1
    except URLError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    output.print_json(out)
    return 0
