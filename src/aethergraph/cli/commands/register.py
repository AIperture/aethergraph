from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError

from aethergraph.cli import http, output


def register_parser(subparsers) -> None:
    register = subparsers.add_parser(
        "register", help="Register a local graph source into registry."
    )
    register.add_argument("--workspace", default="./aethergraph_workspace")
    register.add_argument("--server-url", default=None)
    register.add_argument("--mode", choices=["auto", "api", "local"], default="auto")
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


async def _register_via_local(args: argparse.Namespace, *, payload: dict) -> dict:
    from aethergraph.core.runtime.runtime_registry import current_registry
    from aethergraph.services.registry.registration_service import RegistrationService
    from aethergraph.storage.docstore.fs_doc import FSDocStore
    from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore

    docs = FSDocStore(root=str(Path(args.workspace) / "docs"))
    manifests = RegistrationManifestStore(doc_store=docs)
    service = RegistrationService(
        registry=current_registry(),
        manifest_store=manifests,
    )
    tenant = {"org_id": args.org_id, "user_id": args.user_id}
    if args.source == "file":
        if not args.path:
            raise ValueError("--path is required for --source=file")
        result = await service.register_by_file(
            args.path,
            app_config=payload["app_config"],
            agent_config=payload["agent_config"],
            tenant=tenant,
            persist=payload["persist"],
            strict=payload["strict"],
        )
    else:
        result = await service.register_by_artifact(
            artifact_id=args.artifact_id,
            uri=args.uri,
            app_config=payload["app_config"],
            agent_config=payload["agent_config"],
            tenant=tenant,
            persist=payload["persist"],
            strict=payload["strict"],
        )
    return RegistrationService.to_dict(result)


def handle(args: argparse.Namespace) -> int:
    payload, headers = _build_payload(args)

    if args.mode in {"api", "auto"}:
        try:
            out = _register_via_api(args, payload, headers)
            output.print_json(out)
            return 0
        except HTTPError as exc:
            if args.mode == "api":
                print(http.format_http_error(exc) or str(exc), file=sys.stderr)
                return 1
        except URLError as exc:
            if args.mode == "api":
                print(str(exc), file=sys.stderr)
                return 1

    try:
        out = asyncio.run(_register_via_local(args, payload=payload))
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    output.print_json(out)
    return 0
