from __future__ import annotations

import pytest

from aethergraph.storage.docstore.fs_doc import FSDocStore
from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore


@pytest.mark.asyncio
async def test_manifest_store_tenant_filtering(tmp_path):
    docs = FSDocStore(root=str(tmp_path / "docs"))
    store = RegistrationManifestStore(doc_store=docs)

    await store.upsert_entry(
        {
            "entry_id": "global",
            "source_kind": "file",
            "source_ref": "/tmp/global.py",
            "tenant": None,
            "active": True,
        }
    )
    await store.upsert_entry(
        {
            "entry_id": "u1",
            "source_kind": "file",
            "source_ref": "/tmp/u1.py",
            "tenant": {"org_id": "o1", "user_id": "u1", "client_id": None},
            "active": True,
        }
    )
    await store.upsert_entry(
        {
            "entry_id": "u2",
            "source_kind": "file",
            "source_ref": "/tmp/u2.py",
            "tenant": {"org_id": "o1", "user_id": "u2", "client_id": None},
            "active": True,
        }
    )

    tenant_u1 = {"org_id": "o1", "user_id": "u1", "client_id": None}
    rows_with_global = await store.list_entries(tenant=tenant_u1, include_global=True)
    rows_no_global = await store.list_entries(tenant=tenant_u1, include_global=False)

    assert {r["entry_id"] for r in rows_with_global} == {"global", "u1"}
    assert {r["entry_id"] for r in rows_no_global} == {"u1"}
