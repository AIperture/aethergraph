"""Blocking immutable AG Host command for local supervisor ownership."""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
import uvicorn

from aethergraph.config.config import SlackSettings, TelegramSettings
from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.contracts.integration import IntegrationKind
from aethergraph.services.host import (
    HostManifestError,
    HostProviderConnection,
    HostRuntimeIdentity,
    build_host,
    load_host_manifest,
)


class _LaunchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _RuntimeIdentityRecord(_LaunchModel):
    environment_snapshot_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_settings_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _SlackProviderSecret(_LaunchModel):
    kind: Literal["slack"]
    integration_id: str = Field(min_length=1, max_length=255)
    mode: Literal["socket_mode"]
    bot_token: SecretStr
    app_token: SecretStr


class _TelegramProviderSecret(_LaunchModel):
    kind: Literal["telegram"]
    integration_id: str = Field(min_length=1, max_length=255)
    mode: Literal["polling"]
    bot_token: SecretStr


_ProviderSecret = Annotated[
    _SlackProviderSecret | _TelegramProviderSecret,
    Field(discriminator="kind"),
]


class _ProviderSecretBundle(_LaunchModel):
    connections: tuple[_ProviderSecret, ...] = ()


def register_parser(subparsers) -> None:
    command = subparsers.add_parser(
        "host",
        help="Run one immutable compiled deployment on loopback (blocking).",
    )
    command.add_argument("--manifest", required=True, help="Exact sealed HostManifest JSON file.")
    command.add_argument(
        "--runtime-identity",
        required=True,
        help="Exact supervisor-verified runtime identity JSON file.",
    )
    command.add_argument(
        "--settings",
        required=True,
        help="Exact pinned application settings dotenv file.",
    )
    command.add_argument(
        "--workspace",
        required=True,
        help="Deployment-owned operational workspace.",
    )
    command.add_argument(
        "--control-token",
        required=True,
        help="File handle containing the per-launch supervisor token.",
    )
    command.add_argument(
        "--provider-secrets",
        help="Optional secret bundle handle for exact Slack/Telegram connections.",
    )
    command.add_argument("--log-level", default="warning")
    command.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_host(args))
    except KeyboardInterrupt:
        return 0
    except (HostManifestError, FileNotFoundError, ValidationError, ValueError):
        _emit_failure("host.invalid_launch")
        return 2
    except ImportError:
        _emit_failure("host.missing_dependency")
        return 3
    except Exception:
        _emit_failure("host.start_failed")
        return 1


async def _run_host(args: argparse.Namespace) -> int:
    manifest = load_host_manifest(args.manifest)
    runtime_identity = _load_runtime_identity(args.runtime_identity)
    settings_path = _require_file(args.settings, max_bytes=1_000_000)
    settings_digest = sha256(settings_path.read_bytes()).hexdigest()
    if settings_digest != manifest.application_settings_digest:
        raise HostManifestError("Pinned application settings digest does not match manifest.")
    if runtime_identity.application_settings_digest != settings_digest:
        raise HostManifestError("Runtime identity does not match pinned application settings.")
    settings = load_settings(env_file=settings_path)
    set_current_settings(settings)
    control_token = _read_control_token(args.control_token)
    providers = _load_provider_connections(args.provider_secrets, settings)
    host = build_host(
        manifest=manifest,
        runtime_identity=runtime_identity,
        workspace=args.workspace,
        settings=settings,
        provider_connections=providers,
    )
    app = host.create_app(control_token=control_token)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2048)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level=args.log_level,
            access_log=False,
        )
    )
    app.state.host_shutdown = lambda: setattr(server, "should_exit", True)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await _wait_for_server_start(server, task)
        print(
            json.dumps(
                {
                    "schema_version": "aethergraph.host-ready/v1",
                    "deployment_id": manifest.deployment_id,
                    "build_id": manifest.build_id,
                    "manifest_digest": manifest.manifest_digest,
                    "base_url": f"http://127.0.0.1:{port}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        await task
    finally:
        if not task.done():
            server.should_exit = True
            await task
        listener.close()
    return 0


def _load_runtime_identity(path: str | Path) -> HostRuntimeIdentity:
    identity_path = _require_file(path, max_bytes=64_000)
    record = _RuntimeIdentityRecord.model_validate_json(identity_path.read_text(encoding="utf-8"))
    return HostRuntimeIdentity(**record.model_dump())


def _load_provider_connections(
    path: str | Path | None,
    settings,
) -> tuple[HostProviderConnection, ...]:
    if path is None:
        return ()
    secret_path = _require_file(path, max_bytes=64_000)
    bundle = _ProviderSecretBundle.model_validate_json(secret_path.read_text(encoding="utf-8"))
    connections: list[HostProviderConnection] = []
    for secret in bundle.connections:
        if isinstance(secret, _SlackProviderSecret):
            from aethergraph.plugins.channel.adapters.slack import SlackChannelAdapter

            provider_settings = settings.model_copy(deep=True)
            provider_settings.slack = SlackSettings(
                integration_id=secret.integration_id,
                enabled=True,
                bot_token=secret.bot_token,
                app_token=secret.app_token,
                socket_mode_enabled=True,
                webhook_enabled=False,
            )
            delivery_adapter = SlackChannelAdapter(bot_token=secret.bot_token.get_secret_value())

            async def close_slack_delivery():
                return None

            def slack_transport(container, selected=provider_settings):
                from aethergraph.plugins.channel.websockets.slack_ws import (
                    SlackSocketModeRunner,
                )

                return SlackSocketModeRunner(container=container, settings=selected)

            connections.append(
                HostProviderConnection(
                    integration_id=secret.integration_id,
                    integration_kind=IntegrationKind.SLACK,
                    delivery_adapter=delivery_adapter,
                    transport_factory=slack_transport,
                    close_delivery=close_slack_delivery,
                )
            )
            continue

        from aethergraph.plugins.channel.adapters.telegram import TelegramChannelAdapter

        provider_settings = settings.model_copy(deep=True)
        provider_settings.telegram = TelegramSettings(
            integration_id=secret.integration_id,
            enabled=True,
            bot_token=secret.bot_token,
            webhook_enabled=False,
            polling_enabled=True,
        )
        delivery_adapter = TelegramChannelAdapter(bot_token=secret.bot_token.get_secret_value())

        async def close_telegram_delivery(selected=delivery_adapter):
            await selected.aclose()

        def telegram_transport(container, selected=provider_settings):
            from aethergraph.plugins.channel.websockets.telegram_polling import (
                TelegramPollingRunner,
            )

            return TelegramPollingRunner(container=container, settings=selected)

        connections.append(
            HostProviderConnection(
                integration_id=secret.integration_id,
                integration_kind=IntegrationKind.TELEGRAM,
                delivery_adapter=delivery_adapter,
                transport_factory=telegram_transport,
                close_delivery=close_telegram_delivery,
            )
        )
    return tuple(connections)


def _read_control_token(path: str | Path) -> str:
    token_path = _require_file(path, max_bytes=4_096)
    token = token_path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("Control token must contain at least 32 characters.")
    return token


def _require_file(path: str | Path, *, max_bytes: int) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.stat().st_size > max_bytes:
        raise ValueError("AG Host launch input exceeds its bounded size.")
    return resolved


async def _wait_for_server_start(
    server: uvicorn.Server,
    task: asyncio.Task[None],
) -> None:
    async with asyncio.timeout(60):
        while not server.started:
            if task.done():
                error = task.exception()
                if error is not None:
                    raise error
                raise RuntimeError("AG Host server stopped before readiness.")
            await asyncio.sleep(0.02)


def _emit_failure(code: str) -> None:
    print(
        json.dumps(
            {
                "schema_version": "aethergraph.host-diagnostic/v1",
                "kind": "host.failed",
                "code": code,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


__all__ = ["handle", "register_parser"]
