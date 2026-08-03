from __future__ import annotations

import asyncio

import pytest

from aethergraph.contracts.integration import (
    HostManifest,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    SemanticEventKind,
)
from aethergraph.services.integration import (
    IntegrationConnection,
    IntegrationConnectionState,
    IntegrationManager,
    IntegrationManagerError,
)

_DIGEST = "a" * 64


class _Transport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.ready = asyncio.Event()
        self.stopped = False

    async def start(self) -> None:
        if self.fail:
            raise RuntimeError("provider failed")
        self.ready.set()
        await asyncio.Event().wait()

    async def wait_ready(self) -> None:
        if self.fail:
            await asyncio.sleep(0)
            raise RuntimeError("provider failed")
        await self.ready.wait()

    async def stop(self) -> None:
        self.stopped = True
        self.ready.clear()


def _manifest(*, kind: IntegrationKind | None = None) -> HostManifest:
    routes = ()
    if kind is not None:
        routes = (
            IntegrationRoute(
                route_id=f"route-{kind.value}",
                integration_id=f"integration-{kind.value}",
                integration_kind=kind,
                entry_agent_id="demo",
                enabled=True,
                match_policy=IntegrationMatchPolicy(),
                session_policy=IntegrationSessionPolicy(scope="conversation"),
                required_capabilities=IntegrationCapabilities(
                    event_kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
                    streaming=False,
                    interactions=False,
                    attachments=False,
                    cancellation=False,
                ),
            ),
        )
    return HostManifest(
        deployment_id="deployment-1",
        build_id="0123456789abcdef01234567",
        source_digest=_DIGEST,
        build_root="C:/build/0123456789abcdef01234567",
        entrypoint_module="demo_compiled.entry",
        entrypoint_symbol="demo_entry",
        graph_id="demo.graph",
        entry_agent_id="demo",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        integration_routes=routes,
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


@pytest.mark.asyncio
async def test_integration_manager_starts_and_stops_exact_connection() -> None:
    transport = _Transport()
    manager = IntegrationManager(
        manifest=_manifest(kind=IntegrationKind.TELEGRAM),
        connections=(
            IntegrationConnection(
                integration_id="integration-telegram",
                integration_kind=IntegrationKind.TELEGRAM,
                transport=transport,
                delivery_adapter=object(),
            ),
        ),
        readiness_timeout_seconds=1,
    )

    await manager.start()

    assert manager.ready
    assert manager.statuses()[0].state == IntegrationConnectionState.READY
    assert set(manager.channel_adapters()) == {"tg"}

    await manager.stop()

    assert transport.stopped
    assert manager.statuses()[0].state == IntegrationConnectionState.STOPPED


def test_integration_manager_requires_exact_provider_routes() -> None:
    with pytest.raises(IntegrationManagerError, match="missing=slack"):
        IntegrationManager(
            manifest=_manifest(kind=IntegrationKind.SLACK),
            connections=(),
        )


@pytest.mark.asyncio
async def test_integration_manager_fails_host_startup() -> None:
    transport = _Transport(fail=True)
    manager = IntegrationManager(
        manifest=_manifest(kind=IntegrationKind.SLACK),
        connections=(
            IntegrationConnection(
                integration_id="integration-slack",
                integration_kind=IntegrationKind.SLACK,
                transport=transport,
                delivery_adapter=object(),
            ),
        ),
        readiness_timeout_seconds=1,
    )

    with pytest.raises(IntegrationManagerError, match="failed to start"):
        await manager.start()

    assert transport.stopped
