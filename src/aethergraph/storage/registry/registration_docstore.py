from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
import uuid

from aethergraph.contracts.storage.doc_store import DocStore

REGISTRY_DOC_PREFIX = "registry:entry:"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_tenant(tenant: Mapping[str, str | None] | None) -> dict[str, str | None] | None:
    if tenant is None:
        return None
    norm = {
        "org_id": tenant.get("org_id"),
        "user_id": tenant.get("user_id"),
        "client_id": tenant.get("client_id"),
    }
    if not any(norm.values()):
        return None
    return norm


def _tenant_matches(
    *,
    entry_tenant: Mapping[str, str | None] | None,
    requested_tenant: Mapping[str, str | None] | None,
    include_global: bool,
) -> bool:
    entry_norm = _normalize_tenant(entry_tenant)
    req_norm = _normalize_tenant(requested_tenant)

    if req_norm is None:
        # No tenant filter -> include all entries.
        return True

    if entry_norm is None:
        return bool(include_global)

    return (
        (entry_norm.get("org_id") or "") == (req_norm.get("org_id") or "")
        and (entry_norm.get("user_id") or "") == (req_norm.get("user_id") or "")
        and (entry_norm.get("client_id") or "") == (req_norm.get("client_id") or "")
    )


class RegistrationManifestStore:
    """
    Tenant-aware manifest persistence for source-based registration replay.
    """

    def __init__(self, *, doc_store: DocStore):
        self._docs = doc_store

    @staticmethod
    def doc_id(entry_id: str) -> str:
        return f"{REGISTRY_DOC_PREFIX}{entry_id}"

    async def upsert_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now_iso()
        row = dict(entry)
        row.setdefault("entry_id", uuid.uuid4().hex)
        row.setdefault("active", True)
        row.setdefault("created_at", now)
        row["updated_at"] = now
        await self._docs.put(self.doc_id(str(row["entry_id"])), row)
        return row

    async def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        return await self._docs.get(self.doc_id(entry_id))

    async def set_last_error(self, *, entry_id: str, last_error: str | None) -> None:
        row = await self.get_entry(entry_id)
        if not row:
            return
        row["last_error"] = last_error
        row["updated_at"] = _utc_now_iso()
        await self._docs.put(self.doc_id(entry_id), row)

    async def list_entries(
        self,
        *,
        tenant: Mapping[str, str | None] | None = None,
        include_global: bool = True,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        ids = await self._docs.list()
        out: list[dict[str, Any]] = []
        for doc_id in ids:
            if not doc_id.startswith(REGISTRY_DOC_PREFIX):
                continue
            row = await self._docs.get(doc_id)
            if not row:
                continue
            if active_only and not bool(row.get("active", True)):
                continue
            if not _tenant_matches(
                entry_tenant=row.get("tenant"),
                requested_tenant=tenant,
                include_global=include_global,
            ):
                continue
            out.append(row)
        out.sort(key=lambda r: str(r.get("updated_at") or ""))
        return out

    async def delete_entries_for_app(
        self,
        *,
        app_id: str,
        tenant: Mapping[str, str | None] | None = None,
        include_global: bool = False,
    ) -> int:
        return await self._delete_entries_for_subject(
            subject_field="app_id",
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
        return await self._delete_entries_for_subject(
            subject_field="agent_id",
            subject_id=agent_id,
            tenant=tenant,
            include_global=include_global,
        )

    async def _delete_entries_for_subject(
        self,
        *,
        subject_field: str,
        subject_id: str,
        tenant: Mapping[str, str | None] | None,
        include_global: bool,
    ) -> int:
        ids = await self._docs.list()
        deleted = 0
        for doc_id in ids:
            if not doc_id.startswith(REGISTRY_DOC_PREFIX):
                continue
            row = await self._docs.get(doc_id)
            if not row:
                continue
            if str(row.get(subject_field) or "") != subject_id:
                continue
            if not _tenant_matches(
                entry_tenant=row.get("tenant"),
                requested_tenant=tenant,
                include_global=include_global,
            ):
                continue
            await self._docs.delete(doc_id)
            deleted += 1
        return deleted
