"""Explicit provider transport lifecycle for one immutable AG Host."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from aethergraph.contracts.integration import HostManifest, IntegrationKind


class IntegrationTransport(Protocol):
    """Lifecycle required from an explicitly configured provider connection."""

    async def start(self) -> None: ...

    async def wait_ready(self) -> None: ...

    async def stop(self) -> None: ...


class IntegrationConnectionState(StrEnum):
    """Closed lifecycle states reported by the Integration Manager."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class IntegrationConnection:
    """One explicit provider transport selected by the control plane."""

    integration_id: str
    integration_kind: IntegrationKind
    transport: IntegrationTransport
    delivery_adapter: object
    close_delivery: Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class IntegrationConnectionStatus:
    """Redacted connection state safe for readiness and diagnostics."""

    integration_id: str
    integration_kind: IntegrationKind
    state: IntegrationConnectionState
    error_code: str | None = None


class IntegrationManagerError(RuntimeError):
    """Report deterministic provider configuration or lifecycle failure."""


class IntegrationManager:
    """Own all explicitly configured provider transports for one AG Host."""

    def __init__(
        self,
        *,
        manifest: HostManifest,
        connections: Sequence[IntegrationConnection],
        readiness_timeout_seconds: float = 30.0,
    ) -> None:
        """Create an immutable provider lifecycle manager.

        Construction validates that enabled provider routes and configured
        connections match exactly. It performs no dependency import, network
        access, credential lookup, or background startup.

        Examples:
            Create an endpoint-only manager:
                ```python
                manager = IntegrationManager(manifest=manifest, connections=())
                ```

            Create a manager with one Slack connection:
                ```python
                manager = IntegrationManager(
                    manifest=manifest,
                    connections=(slack_connection,),
                    readiness_timeout_seconds=10.0,
                )
                ```

        Args:
            manifest: Immutable Host manifest containing route authority.
            connections: Exact configured provider transport instances.
            readiness_timeout_seconds: Maximum startup wait for each connection.

        Returns:
            None: The initialized manager remains stopped until `start`.

        Notes:
            AG UI and Agent Endpoint routes are HTTP surfaces and require no
            provider connection object.
        """

        if readiness_timeout_seconds <= 0:
            raise ValueError("readiness_timeout_seconds must be positive")
        self.manifest = manifest
        self.readiness_timeout_seconds = readiness_timeout_seconds
        by_id = {connection.integration_id: connection for connection in connections}
        if len(by_id) != len(connections):
            raise IntegrationManagerError("Provider integration_id values must be unique.")
        expected = {
            (route.integration_id, route.integration_kind)
            for route in manifest.integration_routes
            if route.enabled
            and route.integration_kind in {IntegrationKind.SLACK, IntegrationKind.TELEGRAM}
        }
        actual = {
            (connection.integration_id, connection.integration_kind) for connection in connections
        }
        if expected != actual:
            missing = sorted(f"{kind.value}:{identifier}" for identifier, kind in expected - actual)
            unexpected = sorted(
                f"{kind.value}:{identifier}" for identifier, kind in actual - expected
            )
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            raise IntegrationManagerError(
                "Provider connections do not match enabled routes: " + "; ".join(details)
            )
        kinds = [connection.integration_kind for connection in connections]
        if len(kinds) != len(set(kinds)):
            raise IntegrationManagerError(
                "The initial local Host supports one connection per provider kind."
            )
        self._connections = by_id
        self._states = {identifier: IntegrationConnectionState.STOPPED for identifier in by_id}
        self._error_codes: dict[str, str | None] = {identifier: None for identifier in by_id}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False

    async def start(self) -> None:
        """Start every configured provider and wait for exact readiness.

        All transports start concurrently. Any timeout or startup error marks the
        connection failed, stops every connection, and fails Host startup.

        Examples:
            Start an endpoint-only Host:
                ```python
                await manager.start()
                assert manager.ready
                ```

            Start configured providers:
                ```python
                try:
                    await manager.start()
                except IntegrationManagerError:
                    await manager.stop()
                ```

        Args:
            None.

        Returns:
            None: Returns only after every configured provider reports ready.

        Notes:
            Repeated startup is rejected; deployment changes require a new Host.
        """

        if self._started:
            raise IntegrationManagerError("Integration Manager is already started.")
        self._started = True
        if not self._connections:
            return
        try:
            await asyncio.gather(
                *(self._start_one(connection) for connection in self._connections.values())
            )
        except Exception as exc:
            await self.stop()
            if isinstance(exc, IntegrationManagerError):
                raise
            raise IntegrationManagerError("Provider transport startup failed.") from exc

    async def stop(self) -> None:
        """Stop every configured provider and clear all background tasks.

        Stop requests are issued concurrently, remaining startup tasks are
        canceled, and all connection states become stopped.

        Examples:
            Stop a ready manager:
                ```python
                await manager.stop()
                assert not manager.ready
                ```

            Stop after partial startup:
                ```python
                try:
                    await manager.start()
                finally:
                    await manager.stop()
                ```

        Args:
            None.

        Returns:
            None: Returns after transport stop methods and task cleanup finish.

        Notes:
            Stop is idempotent so Host shutdown may call it after startup failure.
        """

        if self._connections:
            await asyncio.gather(
                *(connection.transport.stop() for connection in self._connections.values()),
                return_exceptions=True,
            )
        current = asyncio.current_task()
        for task in self._tasks.values():
            if task is not current and not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        if self._connections:
            await asyncio.gather(
                *(connection.close_delivery() for connection in self._connections.values()),
                return_exceptions=True,
            )
        for identifier in self._states:
            self._states[identifier] = IntegrationConnectionState.STOPPED
        self._started = False

    @property
    def ready(self) -> bool:
        """Report whether all configured providers are ready.

        Endpoint-only Hosts are ready after manager startup because their HTTP
        readiness is owned by the Host server rather than a provider transport.

        Examples:
            Check endpoint-only readiness:
                ```python
                await manager.start()
                assert manager.ready
                ```

            Check readiness before startup:
                ```python
                assert manager.ready is False
                ```

        Args:
            None.

        Returns:
            bool: True only after startup and when every connection is ready.

        Notes:
            Host HTTP readiness additionally checks build and application state.
        """

        return self._started and all(
            state == IntegrationConnectionState.READY for state in self._states.values()
        )

    def statuses(self) -> tuple[IntegrationConnectionStatus, ...]:
        """Return stable redacted diagnostics for all provider connections.

        Status records expose only configured identities, provider kinds, closed
        lifecycle states, and bounded error codes.

        Examples:
            Read provider readiness:
                ```python
                statuses = manager.statuses()
                assert all(item.state == "ready" for item in statuses)
                ```

            Read an endpoint-only manager:
                ```python
                assert manager.statuses() == ()
                ```

        Args:
            None.

        Returns:
            tuple[IntegrationConnectionStatus, ...]: Sorted redacted statuses.

        Notes:
            Exceptions and credentials are never included in these records.
        """

        return tuple(
            IntegrationConnectionStatus(
                integration_id=identifier,
                integration_kind=self._connections[identifier].integration_kind,
                state=self._states[identifier],
                error_code=self._error_codes[identifier],
            )
            for identifier in sorted(self._connections)
        )

    def channel_adapters(self) -> dict[str, object]:
        """Return the exact provider delivery adapters for Host composition.

        The mapping uses canonical Channel prefixes and is constructed only from
        explicitly supplied provider connections.

        Examples:
            Install configured delivery adapters:
                ```python
                container = build_default_container(
                    channel_adapters=manager.channel_adapters(),
                )
                ```

            Inspect an endpoint-only manager:
                ```python
                assert manager.channel_adapters() == {}
                ```

        Args:
            None.

        Returns:
            dict[str, object]: New prefix-to-adapter mapping for this Host.

        Notes:
            The initial local Host accepts at most one connection per provider
            kind because the current Channel address has one canonical prefix.
        """

        prefixes = {
            IntegrationKind.SLACK: "slack",
            IntegrationKind.TELEGRAM: "tg",
        }
        return {
            prefixes[connection.integration_kind]: connection.delivery_adapter
            for connection in self._connections.values()
        }

    async def _start_one(self, connection: IntegrationConnection) -> None:
        identifier = connection.integration_id
        self._states[identifier] = IntegrationConnectionState.STARTING
        task = asyncio.create_task(connection.transport.start())
        self._tasks[identifier] = task
        try:
            await asyncio.wait_for(
                connection.transport.wait_ready(),
                timeout=self.readiness_timeout_seconds,
            )
        except TimeoutError as exc:
            self._states[identifier] = IntegrationConnectionState.FAILED
            self._error_codes[identifier] = "integration.readiness_timeout"
            raise IntegrationManagerError(
                f"Provider connection did not become ready: {identifier}"
            ) from exc
        except Exception as exc:
            self._states[identifier] = IntegrationConnectionState.FAILED
            self._error_codes[identifier] = "integration.start_failed"
            raise IntegrationManagerError(
                f"Provider connection failed to start: {identifier}"
            ) from exc
        if task.done() and (error := task.exception()) is not None:
            self._states[identifier] = IntegrationConnectionState.FAILED
            self._error_codes[identifier] = "integration.start_failed"
            raise IntegrationManagerError(
                f"Provider connection failed to start: {identifier}"
            ) from error
        self._states[identifier] = IntegrationConnectionState.READY
        task.add_done_callback(self._task_done_callback(identifier))

    def _task_done_callback(self, identifier: str) -> Callable[[asyncio.Task[None]], None]:
        def completed(task: asyncio.Task[None]) -> None:
            if task.cancelled() or not self._started:
                return
            try:
                error = task.exception()
            except asyncio.CancelledError:
                return
            if error is not None:
                self._states[identifier] = IntegrationConnectionState.FAILED
                self._error_codes[identifier] = "integration.transport_failed"

        return completed


__all__ = [
    "IntegrationConnection",
    "IntegrationConnectionState",
    "IntegrationConnectionStatus",
    "IntegrationManager",
    "IntegrationManagerError",
    "IntegrationTransport",
]
