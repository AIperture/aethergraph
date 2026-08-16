"""Canonical registry-manifest persistence over provider storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import json
from typing import Any
import uuid

from aethergraph.services.canonical_storage_scope import validate_storage_owner_scope
from aethergraph.services.scope.tenant import normalize_registry_tenant
from aethergraph.storage.contracts import (
    DocumentQuery,
    DocumentRecord,
    DocumentStore,
    PageRequest,
    StorageBundle,
    StorageCapacityError,
    StorageScope,
)

_NAMESPACE = "registry.manifests"
_SCHEMA_VERSION = 1
_PAGE_SIZE = 100
_MAX_ENTRIES = 1_000
_COMPATIBILITY_KEY = "compatibility_metadata"
_TENANT_KEY = "registry_tenant_key"
_DEPRECATED_APP_ID = "app_id"


class CanonicalRegistrationManifestStore:
    """Project the frozen registry-manifest API onto canonical documents."""

    def __init__(
        self,
        *,
        repository: DocumentStore,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind registry manifests to one canonical owner and document repository.

        Construction performs no provider lookup or I/O. Registry tenant identity is
        normalized separately from the provider-authoritative owner scope.

        Examples:
            Bind a runtime repository:
                ```python
                store = CanonicalRegistrationManifestStore(
                    repository=bundle.registry_manifests,
                    owner_scope=owner_scope,
                    clock=clock,
                )
                ```

            Bind a deterministic test clock:
                ```python
                store = CanonicalRegistrationManifestStore(
                    repository=fake_documents,
                    owner_scope=StorageScope(project_id="project-1"),
                    clock=lambda: fixed_now,
                )
                ```

        Args:
            repository: Exact canonical registry-manifest document repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC timestamp source.

        Returns:
            None: The provider-backed service projection is ready.

        Notes:
            The service owns neither repository lifecycle nor provider selection.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def upsert_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Create or revise one registry manifest with exact revision CAS.

        App identity is removed from the indexed document surface and retained only
        inside explicitly deprecated optional compatibility metadata.

        Examples:
            Create a global manifest:
                ```python
                row = await store.upsert_entry({"source_kind": "file", "source_ref": path})
                ```

            Persist deprecated App compatibility metadata:
                ```python
                row = await store.upsert_entry({"entry_id": "m1", "app_id": "legacy-app"})
                ```

        Args:
            entry: Frozen public registry-manifest-shaped mapping to persist.

        Returns:
            dict[str, Any]: Public manifest projection at the committed revision.

        Notes:
            Concurrent stale writes fail directly; no retry or alternate store is used.
        """
        row = dict(entry)
        entry_id = str(row.get("entry_id") or uuid.uuid4().hex)
        now = _utc_iso(self._clock())
        row["entry_id"] = entry_id
        row.setdefault("active", True)
        row.setdefault("created_at", now)
        row["updated_at"] = now
        current = await self._repository.get(self._owner_scope, _NAMESPACE, entry_id)
        committed = await self._repository.compare_and_set(
            self._owner_scope,
            _NAMESPACE,
            entry_id,
            current.revision if current is not None else 0,
            _encode(row),
            _SCHEMA_VERSION,
        )
        return _decode(committed)

    async def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Read one manifest by exact canonical document identity.

        The public projection restores only the explicitly deprecated App metadata;
        provider revision and physical representation remain private.

        Examples:
            Read an existing entry:
                ```python
                row = await store.get_entry("manifest-1")
                ```

            Detect an absent entry:
                ```python
                assert await store.get_entry("missing") is None
                ```

        Args:
            entry_id: Exact stable registry-manifest identity.

        Returns:
            dict[str, Any] | None: Public manifest projection or `None`.

        Notes:
            The lookup is scoped and indexed; it never scans provider documents.
        """
        record = await self._repository.get(self._owner_scope, _NAMESPACE, entry_id)
        return _decode(record) if record is not None else None

    async def set_last_error(self, *, entry_id: str, last_error: str | None) -> None:
        """Update one manifest replay error at its exact current revision.

        Missing entries preserve the frozen no-op behavior. Concurrent replacement is
        reported as a canonical revision conflict.

        Examples:
            Record a replay failure:
                ```python
                await store.set_last_error(entry_id="manifest-1", last_error="invalid source")
                ```

            Clear a prior replay failure:
                ```python
                await store.set_last_error(entry_id="manifest-1", last_error=None)
                ```

        Args:
            entry_id: Exact stable registry-manifest identity.
            last_error: Bounded caller-authored diagnostic or `None`.

        Returns:
            None: The error was updated or the manifest was absent.

        Notes:
            No unconditional overwrite, retry, or fallback path is provided.
        """
        current = await self._repository.get(self._owner_scope, _NAMESPACE, entry_id)
        if current is None:
            return
        row = _decode(current)
        row["last_error"] = last_error
        row["updated_at"] = _utc_iso(self._clock())
        await self._repository.compare_and_set(
            self._owner_scope,
            _NAMESPACE,
            entry_id,
            current.revision,
            _encode(row),
            _SCHEMA_VERSION,
        )

    async def list_entries(
        self,
        *,
        tenant: Mapping[str, str | None] | None = None,
        include_global: bool = True,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """List a bounded tenant-visible registry-manifest projection.

        Canonical queries filter active state and normalized tenant keys in the
        provider before service hydration. The frozen list result remains ordered by
        authored update time.

        Examples:
            List all active manifests:
                ```python
                rows = await store.list_entries()
                ```

            List one tenant without global entries:
                ```python
                rows = await store.list_entries(tenant=tenant, include_global=False)
                ```

        Args:
            tenant: Optional registry org/user compatibility identity.
            include_global: Include global entries when a tenant is supplied.
            active_only: Restrict results to active manifests.

        Returns:
            list[dict[str, Any]]: Detached public manifest mappings.

        Notes:
            More than `_MAX_ENTRIES` fails explicitly instead of scanning unboundedly.
        """
        records = await self._query_records(
            tenant=tenant,
            include_global=include_global,
            active_only=active_only,
        )
        rows = [_decode(record) for record in records]
        rows.sort(key=lambda row: str(row.get("updated_at") or ""))
        return rows

    async def delete_entries_for_app(
        self,
        *,
        app_id: str,
        tenant: Mapping[str, str | None] | None = None,
        include_global: bool = False,
    ) -> int:
        """Delete bounded manifests carrying one deprecated App identity.

        App identity is inspected only after bounded hydration of compatibility
        metadata and is never a provider scope, schema column, or query index.

        Examples:
            Delete one tenant's App manifests:
                ```python
                count = await store.delete_entries_for_app(app_id="app-1", tenant=tenant)
                ```

            Include global compatibility entries explicitly:
                ```python
                count = await store.delete_entries_for_app(
                    app_id="app-1", tenant=tenant, include_global=True
                )
                ```

        Args:
            app_id: Deprecated optional App compatibility identity.
            tenant: Optional registry org/user compatibility identity.
            include_global: Include global manifests in a tenant-scoped deletion.

        Returns:
            int: Number of exact manifest revisions deleted.

        Notes:
            Revision conflicts propagate; deletion never retries another store.
        """
        return await self._delete_subject(
            field="app_id",
            subject_id=app_id,
            tenant=tenant,
            include_global=include_global,
        )

    async def delete_entries_for_agent(
        self,
        *,
        agent_id: str,
        tenant: Mapping[str, str | None] | None = None,
        include_global: bool = False,
    ) -> int:
        """Delete bounded manifests carrying one Agent identity.

        The service queries the exact owner and normalized registry tenant before
        comparing hydrated Agent compatibility fields.

        Examples:
            Delete one tenant's Agent manifests:
                ```python
                count = await store.delete_entries_for_agent(agent_id="agent-1", tenant=tenant)
                ```

            Delete matching global manifests:
                ```python
                count = await store.delete_entries_for_agent(
                    agent_id="agent-1", tenant=tenant, include_global=True
                )
                ```

        Args:
            agent_id: Exact registry Agent identity.
            tenant: Optional registry org/user compatibility identity.
            include_global: Include global manifests in a tenant-scoped deletion.

        Returns:
            int: Number of exact manifest revisions deleted.

        Notes:
            Provider revisions prevent deletion of concurrently replaced entries.
        """
        return await self._delete_subject(
            field="agent_id",
            subject_id=agent_id,
            tenant=tenant,
            include_global=include_global,
        )

    async def _delete_subject(
        self,
        *,
        field: str,
        subject_id: str,
        tenant: Mapping[str, str | None] | None,
        include_global: bool,
    ) -> int:
        records = await self._query_records(
            tenant=tenant,
            include_global=include_global,
            active_only=False,
        )
        deleted = 0
        for record in records:
            row = _decode(record)
            if str(row.get(field) or "") != subject_id:
                continue
            if await self._repository.delete(
                self._owner_scope,
                _NAMESPACE,
                record.document_id,
                record.revision,
            ):
                deleted += 1
        return deleted

    async def _query_records(
        self,
        *,
        tenant: Mapping[str, str | None] | None,
        include_global: bool,
        active_only: bool,
    ) -> list[DocumentRecord]:
        normalized = normalize_registry_tenant(tenant)
        tenant_keys = (
            (None,)
            if normalized is None
            else (
                (_tenant_key(normalized), "global")
                if include_global
                else (_tenant_key(normalized),)
            )
        )
        records: list[DocumentRecord] = []
        for tenant_key in tenant_keys:
            metadata: dict[str, object] = {}
            if tenant_key is not None:
                metadata[_TENANT_KEY] = tenant_key
            if active_only:
                metadata["active"] = True
            cursor: str | None = None
            while True:
                page = await self._repository.query(
                    DocumentQuery(
                        scope=self._owner_scope,
                        namespace=_NAMESPACE,
                        metadata=metadata,
                        page=PageRequest(limit=_PAGE_SIZE, cursor=cursor),
                    )
                )
                records.extend(page.items)
                if len(records) > _MAX_ENTRIES or (
                    len(records) == _MAX_ENTRIES and page.next_cursor is not None
                ):
                    raise StorageCapacityError(
                        f"Registry manifest query exceeds {_MAX_ENTRIES} entries"
                    )
                cursor = page.next_cursor
                if cursor is None:
                    break
        return records


def bind_canonical_registration_manifest_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalRegistrationManifestStore:
    """Bind registry persistence to the bundle's exact manifest document field.

    The binding is construction-only; `DefaultContainer` owns provider selection,
    readiness, publication, and shutdown.

    Examples:
        Bind production composition inputs:
            ```python
            manifests = bind_canonical_registration_manifest_store(
                bundle=bundle, owner_scope=owner_scope, clock=clock
            )
            ```

        Bind a deterministic fake bundle:
            ```python
            manifests = bind_canonical_registration_manifest_store(
                bundle=fake_bundle,
                owner_scope=StorageScope(project_id="project-1"),
                clock=lambda: fixed_now,
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC timestamp source.

    Returns:
        CanonicalRegistrationManifestStore: Frozen service-facing projection.

    Notes:
        The binding performs no provider selection, I/O, fallback, or lifecycle action.
    """
    return CanonicalRegistrationManifestStore(
        repository=bundle.registry_manifests,
        owner_scope=owner_scope,
        clock=clock,
    )


def _encode(row: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _plain(row)
    if not isinstance(encoded, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("registry manifest must be a mapping")
    for key in (_COMPATIBILITY_KEY, "application_id", "client_id"):
        if key in encoded:
            raise ValueError(f"registry manifest reserves or rejects {key!r}")
    app_id = encoded.pop(_DEPRECATED_APP_ID, None)
    tenant = normalize_registry_tenant(encoded.get("tenant"))
    encoded["tenant"] = tenant
    encoded[_TENANT_KEY] = _tenant_key(tenant)
    if app_id is not None:
        encoded[_COMPATIBILITY_KEY] = {
            _DEPRECATED_APP_ID: {
                "value": str(app_id),
                "deprecated": True,
                "scheduled_removal": "future breaking release",
            }
        }
    return encoded


def _decode(record: DocumentRecord) -> dict[str, Any]:
    row = _plain(record.document)
    if not isinstance(row, dict):  # pragma: no cover - DocumentRecord guarantees this
        raise TypeError("registry manifest document must be a mapping")
    row.pop(_TENANT_KEY, None)
    compatibility = row.pop(_COMPATIBILITY_KEY, None)
    if isinstance(compatibility, dict):
        app = compatibility.get(_DEPRECATED_APP_ID)
        if isinstance(app, dict) and app.get("deprecated") is True and app.get("value"):
            row[_DEPRECATED_APP_ID] = str(app["value"])
    return row


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _tenant_key(tenant: Mapping[str, str | None] | None) -> str:
    if tenant is None:
        return "global"
    return json.dumps(
        {"org_id": tenant.get("org_id"), "user_id": tenant.get("user_id")},
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware UTC datetime")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("clock must return UTC")
    return value.isoformat().replace("+00:00", "Z")
