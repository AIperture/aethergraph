from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import getdoc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.registry import (
    CanonicalRegistrationManifestStore,
    RegistrationService,
    UnifiedRegistry,
    bind_canonical_registration_manifest_store,
)
from aethergraph.storage.contracts import StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalDocumentStore,
    LocalSQLiteDatabase,
)

_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")
_GRAPH_SOURCE = """
from aethergraph import graphify, tool

@tool(name="canonical_registry_identity", outputs=["value"])
def canonical_registry_identity(value: int):
    return {"value": value}

@graphify(name="canonical_registry_graph", inputs=["value"], outputs=["value"])
def canonical_registry_graph(value):
    result = canonical_registry_identity(value=value)
    return {"value": result.value}
"""


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 6, tzinfo=UTC)

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(microseconds=1)
        return current


def _store(
    root: Path,
) -> tuple[CanonicalRegistrationManifestStore, LocalSQLiteDatabase, _Clock]:
    clock = _Clock()
    database = LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    documents = LocalDocumentStore(database=database, clock=clock)
    return (
        CanonicalRegistrationManifestStore(
            repository=documents,
            owner_scope=_OWNER,
            clock=clock.now,
        ),
        database,
        clock,
    )


@pytest.mark.asyncio
async def test_canonical_registry_keeps_app_id_only_in_deprecated_compatibility_metadata(
    tmp_path: Path,
) -> None:
    store, database, _clock = _store(tmp_path)

    row = await store.upsert_entry(
        {
            "entry_id": "manifest-1",
            "source_kind": "file",
            "source_ref": "graph.py",
            "app_id": "legacy-app",
            "agent_id": "agent-1",
            "tenant": {"org_id": "org-1", "user_id": "user-1", "client_id": "drop-me"},
        }
    )
    raw = (
        await database.fetch_all(
            "SELECT document_json FROM local_documents WHERE document_id = ?",
            ("manifest-1",),
        )
    )[0]
    metadata = await database.fetch_all(
        "SELECT key FROM local_document_metadata WHERE document_id = ? ORDER BY key",
        ("manifest-1",),
    )

    assert row["app_id"] == "legacy-app"
    assert row["tenant"] == {"org_id": "org-1", "user_id": "user-1"}
    document = json.loads(str(raw["document_json"]))
    assert "app_id" not in document
    assert document["compatibility_metadata"]["app_id"] == {
        "value": "legacy-app",
        "deprecated": True,
        "scheduled_removal": "future breaking release",
    }
    assert "app_id" not in {item["key"] for item in metadata}
    assert "client_id" not in str(document)
    await database.close()


@pytest.mark.asyncio
async def test_canonical_registry_filters_tenants_and_updates_with_revision_cas(
    tmp_path: Path,
) -> None:
    store, database, _clock = _store(tmp_path)
    await store.upsert_entry({"entry_id": "global", "tenant": None})
    await store.upsert_entry({"entry_id": "u1", "tenant": {"org_id": "o1", "user_id": "u1"}})
    await store.upsert_entry({"entry_id": "u2", "tenant": {"org_id": "o1", "user_id": "u2"}})
    await store.set_last_error(entry_id="u1", last_error="retry")

    with_global = await store.list_entries(
        tenant={"org_id": "o1", "user_id": "u1"}, include_global=True
    )
    tenant_only = await store.list_entries(
        tenant={"org_id": "o1", "user_id": "u1"}, include_global=False
    )

    assert {row["entry_id"] for row in with_global} == {"global", "u1"}
    assert [row["entry_id"] for row in tenant_only] == ["u1"]
    assert tenant_only[0]["last_error"] == "retry"
    persisted = (
        await database.fetch_all(
            "SELECT revision FROM local_documents WHERE document_id = ?", ("u1",)
        )
    )[0]
    assert persisted["revision"] == 2
    await database.close()


@pytest.mark.asyncio
async def test_canonical_registry_deletes_app_and_agent_entries_at_exact_revision(
    tmp_path: Path,
) -> None:
    store, database, _clock = _store(tmp_path)
    tenant = {"org_id": "o1", "user_id": "u1"}
    await store.upsert_entry(
        {"entry_id": "app", "app_id": "app-1", "agent_id": "other", "tenant": tenant}
    )
    await store.upsert_entry({"entry_id": "agent", "agent_id": "agent-1", "tenant": tenant})
    await store.upsert_entry({"entry_id": "global", "app_id": "app-1", "tenant": None})

    assert await store.delete_entries_for_app(app_id="app-1", tenant=tenant) == 1
    assert await store.delete_entries_for_agent(agent_id="agent-1", tenant=tenant) == 1
    assert await store.get_entry("app") is None
    assert await store.get_entry("agent") is None
    assert await store.get_entry("global") is not None
    await database.close()


@pytest.mark.asyncio
async def test_registration_service_consumes_canonical_manifest_projection(tmp_path: Path) -> None:
    store, database, _clock = _store(tmp_path)
    source = tmp_path / "canonical_registry_graph.py"
    source.write_text(_GRAPH_SOURCE, encoding="utf-8")
    service = RegistrationService(registry=UnifiedRegistry(), manifest_store=store)

    result = await service.register_by_file(
        str(source),
        app_config={"id": "compat-app", "name": "Compatibility App"},
    )
    manifests = await store.list_entries()

    assert result.success is True
    assert result.app_id == "compat-app"
    assert len(manifests) == 1
    assert manifests[0]["app_id"] == "compat-app"
    persisted = json.loads(
        str(
            (
                await database.fetch_all(
                    "SELECT document_json FROM local_documents WHERE document_id = ?",
                    (result.entry_id,),
                )
            )[0]["document_json"]
        )
    )
    assert "app_id" not in persisted
    await database.close()


@pytest.mark.parametrize("reserved", ["application_id", "client_id", "compatibility_metadata"])
@pytest.mark.asyncio
async def test_canonical_registry_rejects_aliases_and_caller_owned_compatibility_envelopes(
    tmp_path: Path,
    reserved: str,
) -> None:
    store, database, _clock = _store(tmp_path)
    with pytest.raises(ValueError, match=reserved):
        await store.upsert_entry({"entry_id": "bad", reserved: "forbidden"})
    await database.close()


def test_canonical_registry_factory_maps_only_exact_bundle_field_without_io() -> None:
    repository = object()
    bundle = SimpleNamespace(registry_manifests=repository)

    store = bind_canonical_registration_manifest_store(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=_OWNER,
        clock=lambda: datetime(2026, 8, 16, 6, tzinfo=UTC),
    )

    assert store._repository is repository
    assert store._owner_scope == _OWNER


def test_canonical_registry_public_docstrings_follow_strict_contract() -> None:
    methods = (
        CanonicalRegistrationManifestStore.__init__,
        CanonicalRegistrationManifestStore.upsert_entry,
        CanonicalRegistrationManifestStore.get_entry,
        CanonicalRegistrationManifestStore.set_last_error,
        CanonicalRegistrationManifestStore.list_entries,
        CanonicalRegistrationManifestStore.delete_entries_for_app,
        CanonicalRegistrationManifestStore.delete_entries_for_agent,
        bind_canonical_registration_manifest_store,
    )
    for method in methods:
        docstring = getdoc(method)
        assert docstring is not None
        assert docstring.count("```python") == 2
        positions = [
            docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
        ]
        assert positions == sorted(positions)
