from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from inspect import getdoc
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.canonical_kv import (
    CanonicalKeyValueFacade,
    bind_canonical_key_value_facade,
)
from aethergraph.storage.contracts import (
    StorageIntegrityError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalKeyValueStore,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 19, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self) -> None:
        self.value = _NOW

    def now(self) -> datetime:
        return self.value


def _facade(
    root: Path,
) -> tuple[CanonicalKeyValueFacade, LocalKeyValueStore, LocalSQLiteDatabase, _Clock]:
    clock = _Clock()
    database = LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    repository = LocalKeyValueStore(database=database, clock=clock)
    return (
        CanonicalKeyValueFacade(
            repository=repository,
            owner_scope=_OWNER,
            namespace="runtime.kv",
            clock=clock.now,
        ),
        repository,
        database,
        clock,
    )


@pytest.mark.asyncio
async def test_key_value_facade_projects_crud_batch_prefix_and_ttl(tmp_path: Path) -> None:
    kv, _repository, database, clock = _facade(tmp_path)
    await kv.set("settings", {"enabled": True})
    await kv.set("cache:a", {"value": 1}, ttl_s=30)
    await kv.mset({"cache:b": 2, "other": 3})

    assert dict(await kv.get("settings")) == {"enabled": True}
    assert await kv.get("missing", "default") == "default"
    assert await kv.exists("settings") is True
    assert await kv.exists("missing") is False
    assert await kv.mget(["cache:a", "missing", "cache:b"]) == [
        {"value": 1},
        None,
        2,
    ]
    assert await kv.scan_prefix("cache:", limit=10) == ["cache:a", "cache:b"]
    assert await kv.scan_keys("cache:") == ["cache:a", "cache:b"]
    assert await kv.expire("other", 10) is True
    assert await kv.expire("missing", 10) is False

    clock.value = _NOW + timedelta(seconds=11)
    assert await kv.get("other") is None
    assert await kv.purge_expired(limit=10) == 1
    clock.value = _NOW + timedelta(seconds=31)
    assert await kv.purge_expired(limit=10) == 1
    assert await kv.get("settings") == {"enabled": True}
    await kv.delete("settings")
    await kv.delete("settings")
    assert await kv.exists("settings") is False
    await database.close()


@pytest.mark.asyncio
async def test_key_value_facade_increment_is_atomic_under_concurrency(tmp_path: Path) -> None:
    kv, _repository, database, _clock = _facade(tmp_path)

    values = await asyncio.gather(*(kv.incr("counter") for _ in range(20)))

    assert sorted(values) == list(range(1, 21))
    assert await kv.get("counter") == 20
    await kv.set("bad-counter", {"value": 1})
    with pytest.raises(StorageIntegrityError, match="integer"):
        await kv.incr("bad-counter")
    await database.close()


@pytest.mark.asyncio
async def test_key_value_facade_list_mutations_are_atomic_and_pop_once(tmp_path: Path) -> None:
    kv, _repository, database, _clock = _facade(tmp_path)
    await asyncio.gather(
        *(
            kv.list_append_unique(
                "inbox://ui:session",
                [{"id": f"item-{index}", "index": index}],
            )
            for index in range(16)
        )
    )
    await kv.list_append_unique(
        "inbox://ui:session",
        [{"id": "item-0", "index": 999}],
    )

    first, second = await asyncio.gather(
        kv.list_pop_all("inbox://ui:session"),
        kv.list_pop_all("inbox://ui:session"),
    )
    populated = first or second

    assert len(populated) == 16
    assert {item["id"] for item in populated} == {f"item-{index}" for index in range(16)}
    assert not (first and second)
    assert await kv.get("inbox://ui:session") is None
    await database.close()


@pytest.mark.asyncio
async def test_key_value_facade_deletes_malformed_list_content_and_fails_closed(
    tmp_path: Path,
) -> None:
    kv, repository, database, _clock = _facade(tmp_path)
    await repository.compare_and_set(_OWNER, "runtime.kv", "bad-list", 0, {"not": "a list"})

    with pytest.raises(StorageIntegrityError, match="list value"):
        await kv.list_pop_all("bad-list")
    assert await kv.exists("bad-list") is False

    await kv.set("mixed-list", [{"id": "ok"}, "bad"])
    with pytest.raises(StorageIntegrityError, match="mapping items"):
        await kv.list_append_unique("mixed-list", [{"id": "next"}])
    await database.close()


def test_key_value_facade_binding_uses_exact_bundle_field_and_validates_scope() -> None:
    repository = object()
    bundle = SimpleNamespace(kv=repository)

    facade = bind_canonical_key_value_facade(
        bundle=bundle,
        owner_scope=_OWNER,
        clock=lambda: _NOW,
        namespace="custom.kv",
    )

    assert facade._repository is repository
    assert facade._owner_scope == _OWNER
    assert facade._namespace == "custom.kv"
    with pytest.raises(ValueError, match="execution/external"):
        bind_canonical_key_value_facade(
            bundle=bundle,
            owner_scope=StorageScope(project_id="project-1", run_id="run-1"),
            clock=lambda: _NOW,
        )


def test_key_value_facade_is_provider_neutral_bounded_and_strictly_documented() -> None:
    source = Path(__file__).parents[1] / "src/aethergraph/services/canonical_kv.py"
    source_text = source.read_text(encoding="utf-8")

    for forbidden in (
        "SQLiteKVSync",
        "SqliteKV",
        "InMemoryKV",
        "LayeredKV",
        "build_kv_store",
        "limit=None",
        "decode_cursor",
    ):
        assert forbidden not in source_text
    for method_name in (
        "__init__",
        "get",
        "set",
        "delete",
        "mget",
        "mset",
        "incr",
        "exists",
        "expire",
        "list_append_unique",
        "list_pop_all",
        "scan_prefix",
        "scan_keys",
        "purge_expired",
    ):
        doc = getdoc(getattr(CanonicalKeyValueFacade, method_name)) or ""
        assert doc.splitlines()[0]
        assert "Intro:" in doc
        assert doc.count("```python") >= 2
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Notes:" in doc
    binding_doc = getdoc(bind_canonical_key_value_facade) or ""
    assert "Intro:" in binding_doc
    assert binding_doc.count("```python") >= 2
    assert "Args:" in binding_doc
    assert "Returns:" in binding_doc
    assert "Notes:" in binding_doc
