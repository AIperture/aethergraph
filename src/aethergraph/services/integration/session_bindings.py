"""Durable provider-neutral external-session binding storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Literal, Protocol

from aethergraph.contracts.integration import (
    ExternalIdentity,
    ExternalSessionBinding,
    IntegrationRoute,
)


class SessionBindingError(RuntimeError):
    """Structured failure raised for invalid or incompatible session bindings."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.binding_thread_required",
            "integration.binding_build_mismatch",
        ],
        message: str,
    ) -> None:
        """Create one stable external-session binding failure.

        Examples:
            Reject a missing required thread:
            ```python
            SessionBindingError(
                code="integration.binding_thread_required",
                message="This route requires a thread identity.",
            )
            ```

            Reject an attempt to move a session to another build:
            ```python
            SessionBindingError(
                code="integration.binding_build_mismatch",
                message="Existing session is pinned to another build.",
            )
            ```

        Args:
            code: Stable machine-readable binding failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Binding failures never create a replacement session implicitly.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class BindingResolution:
    """Result of atomically resolving or creating one external binding."""

    binding: ExternalSessionBinding
    created: bool


class ExternalSessionBindingStore(Protocol):
    """Provider-neutral persistence contract for durable session bindings."""

    async def get_or_create(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
    ) -> BindingResolution:
        """Resolve or create one exact route-scoped binding.

        Examples:
            Resolve a conversation:
            ```python
            result = await store.get_or_create(
                route=route,
                external_identity=identity,
                build_id="build-1",
                binding_id="binding-1",
                ag_session_id="session-1",
                now=now,
            )
            ```

            Inspect creation ownership:
            ```python
            if result.created:
                await create_ag_session(result.binding.ag_session_id)
            ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.
            build_id: Immutable build identity for the bound AG session.
            binding_id: Candidate binding identifier.
            ag_session_id: Candidate AG session identifier.
            now: Authoritative acceptance timestamp.

        Returns:
            BindingResolution: Persisted binding and creation ownership.

        Notes:
            Implementations must make scope creation atomic.
        """
        ...

    async def get(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read one exact route-scoped binding.

        Examples:
            Read an existing binding:
            ```python
            binding = await store.get(route=route, external_identity=identity)
            ```

            Detect an unbound identity:
            ```python
            assert await store.get(route=route, external_identity=identity) is None
            ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.

        Returns:
            ExternalSessionBinding | None: Persisted binding when present.

        Notes:
            Implementations must use the same scope calculation as creation.
        """
        ...


class SQLiteExternalSessionBindingStore:
    """Persist unique external-session bindings in a local SQLite database."""

    def __init__(self, path: str | Path) -> None:
        """Create or open the integration operational database.

        Examples:
            Create a store:
            ```python
            store = SQLiteExternalSessionBindingStore("host/integration.db")
            ```

            Reopen bindings after a host restart:
            ```python
            restored = SQLiteExternalSessionBindingStore("host/integration.db")
            ```

        Args:
            path: SQLite database path owned by the local AG Host workspace.

        Returns:
            None.

        Notes:
            A unique `(route_id, scope_key)` index makes concurrent creation
            deterministic across tasks and processes.
        """
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_session_bindings (
                    binding_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    build_id TEXT NOT NULL,
                    binding_json TEXT NOT NULL,
                    UNIQUE(route_id, scope_key)
                )
                """
            )

    async def get_or_create(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
    ) -> BindingResolution:
        """Atomically resolve or create the route-defined external session.

        Examples:
            Create a first binding:
            ```python
            result = await store.get_or_create(
                route=route,
                external_identity=identity,
                build_id="build-1",
                binding_id="binding-1",
                ag_session_id="session-1",
                now=now,
            )
            ```

            Observe an existing concurrent winner:
            ```python
            assert (await store.get_or_create(**same_scope)).created is False
            ```

        Args:
            route: Exact resolved integration route and its session policy.
            external_identity: Authenticated external conversation identity.
            build_id: Immutable build identity for the AG session.
            binding_id: Candidate identifier used only if this call creates the row.
            ag_session_id: Candidate AG session used only if this call creates the row.
            now: Authoritative acceptance timestamp.

        Returns:
            BindingResolution: Persisted binding and whether this call created it.

        Notes:
            Existing bindings remain pinned to their original build. A mismatch
            fails instead of creating a second session path.
        """
        return await asyncio.to_thread(
            self._get_or_create,
            route,
            external_identity,
            build_id,
            binding_id,
            ag_session_id,
            now,
        )

    def _get_or_create(
        self,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
    ) -> BindingResolution:
        scope_key = _scope_key(route=route, external_identity=external_identity)
        candidate = ExternalSessionBinding(
            binding_id=binding_id,
            route_id=route.route_id,
            external_identity=external_identity,
            ag_session_id=ag_session_id,
            build_id=build_id,
            created_at=now,
            last_seen_at=now,
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT build_id, binding_json
                FROM external_session_bindings
                WHERE route_id = ? AND scope_key = ?
                """,
                (route.route_id, scope_key),
            ).fetchone()
            if row is not None:
                binding = ExternalSessionBinding.model_validate_json(row["binding_json"])
                if binding.build_id != build_id:
                    raise SessionBindingError(
                        code="integration.binding_build_mismatch",
                        message=(
                            f"External session is pinned to build {binding.build_id!r}, "
                            f"not {build_id!r}."
                        ),
                    )
                refreshed = binding.model_copy(
                    update={"last_seen_at": max(binding.last_seen_at, now)}
                )
                conn.execute(
                    """
                    UPDATE external_session_bindings
                    SET binding_json = ?
                    WHERE binding_id = ?
                    """,
                    (refreshed.model_dump_json(), refreshed.binding_id),
                )
                return BindingResolution(binding=refreshed, created=False)

            conn.execute(
                """
                INSERT INTO external_session_bindings(
                    binding_id, route_id, scope_key, build_id, binding_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    candidate.binding_id,
                    candidate.route_id,
                    scope_key,
                    candidate.build_id,
                    candidate.model_dump_json(),
                ),
            )
            return BindingResolution(binding=candidate, created=True)

    async def get(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read the exact binding selected by one route session policy.

        Examples:
            Read an existing binding:
            ```python
            binding = await store.get(route=route, external_identity=identity)
            ```

            Detect an unbound conversation:
            ```python
            assert await store.get(route=route, external_identity=new_identity) is None
            ```

        Args:
            route: Exact resolved integration route.
            external_identity: Authenticated external identity to scope.

        Returns:
            ExternalSessionBinding | None: Persisted binding when present.

        Notes:
            Reads apply the same route-authored scope calculation as creation.
        """
        return await asyncio.to_thread(self._get, route, external_identity)

    def _get(
        self,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        scope_key = _scope_key(route=route, external_identity=external_identity)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT binding_json
                FROM external_session_bindings
                WHERE route_id = ? AND scope_key = ?
                """,
                (route.route_id, scope_key),
            ).fetchone()
        if row is None:
            return None
        return ExternalSessionBinding.model_validate_json(row["binding_json"])


def _scope_key(*, route: IntegrationRoute, external_identity: ExternalIdentity) -> str:
    scope = route.session_policy.scope
    include_thread = scope in {"conversation_thread", "conversation_thread_user"}
    include_user = scope in {"conversation_user", "conversation_thread_user"}
    if include_thread and external_identity.thread_id is None:
        raise SessionBindingError(
            code="integration.binding_thread_required",
            message=f"Route {route.route_id!r} requires an external thread identity.",
        )
    fields = {
        "tenant_id": external_identity.tenant_id,
        "conversation_id": external_identity.conversation_id,
        "thread_id": external_identity.thread_id if include_thread else None,
        "user_id": external_identity.user_id if include_user else None,
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))
