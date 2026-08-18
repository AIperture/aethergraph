from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from inspect import getdoc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.auth.authn import DemoGrant, InviteCode
from aethergraph.services.auth.canonical_store import (
    CanonicalAuthStore,
    bind_canonical_auth_store,
)
from aethergraph.storage.contracts import (
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalKeyValueStore,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 17, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self) -> None:
        self.value = _NOW

    def now(self) -> datetime:
        return self.value


def _store(root: Path) -> tuple[CanonicalAuthStore, LocalSQLiteDatabase, _Clock]:
    clock = _Clock()
    database = LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    repository = LocalKeyValueStore(database=database, clock=clock)
    return (
        CanonicalAuthStore(
            grant_repository=repository,
            invite_repository=repository,
            owner_scope=_OWNER,
            clock=clock.now,
        ),
        database,
        clock,
    )


@pytest.mark.asyncio
async def test_auth_store_keeps_app_allowlist_only_in_marked_compatibility_metadata(
    tmp_path: Path,
) -> None:
    store, database, _clock = _store(tmp_path)
    grant = DemoGrant(
        grant_id="grant-1",
        org_id="org-1",
        allowed_apps=["legacy-app"],
        allowed_agents=["agent-1"],
        client_label="demo",
    )

    saved = await store.save_grant(grant)
    loaded = await store.get_grant(grant.grant_id)
    raw = (
        await database.fetch_all(
            "SELECT value_json FROM local_key_values WHERE namespace = ? AND key = ?",
            ("auth.grants", grant.grant_id),
        )
    )[0]
    payload = json.loads(str(raw["value_json"]))

    assert saved == loaded == grant
    assert "allowed_apps" not in {key for key in payload if key != "compatibility_metadata"}
    assert "app_id" not in payload
    assert "application_id" not in payload
    assert "client_id" not in payload
    assert payload["compatibility_metadata"]["allowed_apps"] == {
        "values": ["legacy-app"],
        "deprecated": True,
        "compatibility_only": True,
        "scheduled_removal": "future breaking release",
    }
    await database.close()


@pytest.mark.asyncio
async def test_auth_store_lists_provider_records_without_synthetic_index_and_survives_reopen(
    tmp_path: Path,
) -> None:
    store, database, clock = _store(tmp_path)
    await store.save_grant(DemoGrant(grant_id="grant-b", org_id="org-1"))
    await store.save_grant(DemoGrant(grant_id="grant-a", org_id="org-1"))
    await store.create_invite(InviteCode(code="DEMO-B", grant_id="grant-b"))
    await store.create_invite(InviteCode(code="DEMO-A", grant_id="grant-a"))
    await database.close()

    reopened_database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    repository = LocalKeyValueStore(database=reopened_database, clock=clock)
    reopened = CanonicalAuthStore(
        grant_repository=repository,
        invite_repository=repository,
        owner_scope=_OWNER,
        clock=clock.now,
    )
    keys = await reopened_database.fetch_all(
        "SELECT namespace, key FROM local_key_values ORDER BY namespace, key"
    )

    assert [item.grant_id for item in await reopened.list_grants()] == ["grant-a", "grant-b"]
    assert [item.code for item in await reopened.list_invites()] == ["DEMO-A", "DEMO-B"]
    assert all(row["key"] != "_index" for row in keys)
    await reopened_database.close()


@pytest.mark.asyncio
async def test_auth_store_claims_invite_usage_atomically_under_concurrency(tmp_path: Path) -> None:
    store, database, _clock = _store(tmp_path)
    expiry = _NOW + timedelta(hours=1)
    await store.save_grant(DemoGrant(grant_id="grant-1", org_id="org-1", expires_at=expiry))
    await store.create_invite(
        InviteCode(
            code="DEMO-LIMITED",
            grant_id="grant-1",
            max_uses=3,
            expires_at=expiry,
        )
    )

    outcomes = await asyncio.gather(
        *(store.claim_invite("DEMO-LIMITED") for _ in range(12)),
        return_exceptions=True,
    )
    successes = [item for item in outcomes if not isinstance(item, BaseException)]
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    persisted = await store.get_invite("DEMO-LIMITED")

    assert len(successes) == 3
    assert len(failures) == 9
    assert all(isinstance(item, ValueError) for item in failures)
    assert persisted is not None
    assert persisted.uses == 3
    await database.close()


@pytest.mark.asyncio
async def test_auth_store_duplicate_delete_and_expiry_are_provider_authoritative(
    tmp_path: Path,
) -> None:
    store, database, clock = _store(tmp_path)
    invite = InviteCode(
        code="DEMO-ONCE",
        grant_id="grant-1",
        expires_at=_NOW + timedelta(seconds=1),
    )
    await store.create_invite(invite)

    with pytest.raises(StorageConflictError):
        await store.create_invite(invite)

    clock.value = _NOW + timedelta(seconds=2)
    assert await store.get_invite(invite.code) is None
    assert await store.delete_invite(invite.code) is False
    assert await store.delete_grant("missing") is False
    await database.close()


@pytest.mark.asyncio
async def test_auth_store_rejects_unmarked_app_metadata_and_identity_mismatch(
    tmp_path: Path,
) -> None:
    store, database, _clock = _store(tmp_path)
    repository = store._grant_repository
    await repository.compare_and_set(
        _OWNER,
        "auth.grants",
        "grant-1",
        0,
        {
            "schema_version": 1,
            "grant_id": "other-grant",
            "org_id": "org-1",
            "compatibility_metadata": {
                "allowed_apps": {
                    "values": ["app-1"],
                    "deprecated": False,
                    "compatibility_only": True,
                    "scheduled_removal": "future breaking release",
                }
            },
        },
    )

    with pytest.raises(StorageIntegrityError, match="Unmarked"):
        await store.get_grant("grant-1")

    await repository.compare_and_set(
        _OWNER,
        "auth.grants",
        "grant-2",
        0,
        {
            "schema_version": 1,
            "grant_id": "other-grant",
            "org_id": "org-1",
            "allowed_agents": [],
            "revoked": False,
            "read_only": False,
        },
    )
    with pytest.raises(StorageIntegrityError, match="identity mismatch"):
        await store.get_grant("grant-2")
    await database.close()


def test_auth_store_binding_uses_exact_bundle_fields() -> None:
    grants = object()
    invites = object()
    bundle = SimpleNamespace(auth_grants=grants, auth_invites=invites)

    store = bind_canonical_auth_store(bundle=bundle, owner_scope=_OWNER, clock=lambda: _NOW)

    assert store._grant_repository is grants
    assert store._invite_repository is invites
    assert store._owner_scope == _OWNER


def test_auth_store_has_bounded_provider_neutral_structure_and_strict_docstrings() -> None:
    source = Path(CanonicalAuthStore.__module__.replace(".", "/") + ".py")
    source_text = (Path(__file__).parents[1] / "src" / source).read_text(encoding="utf-8")

    assert "SQLiteKVSync" not in source_text
    assert "limit=None" not in source_text
    assert "decode_cursor" not in source_text
    assert "build_kv_store" not in source_text
    for method_name in (
        "__init__",
        "get_grant",
        "save_grant",
        "delete_grant",
        "get_invite",
        "create_invite",
        "save_invite",
        "claim_invite",
        "delete_invite",
        "list_grants",
        "list_invites",
    ):
        doc = getdoc(getattr(CanonicalAuthStore, method_name)) or ""
        assert doc.splitlines()[0]
        assert "Intro:" in doc
        assert doc.count("```python") >= 2
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Notes:" in doc
    binding_doc = getdoc(bind_canonical_auth_store) or ""
    assert "Intro:" in binding_doc
    assert binding_doc.count("```python") >= 2
    assert "Args:" in binding_doc
    assert "Returns:" in binding_doc
    assert "Notes:" in binding_doc
