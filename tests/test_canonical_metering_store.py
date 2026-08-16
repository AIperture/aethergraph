from __future__ import annotations

from datetime import UTC, datetime
from inspect import getdoc
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.observability.metering import EventLogMeteringService
from aethergraph.services.canonical_metering import (
    CanonicalMeteringStore,
    bind_canonical_metering_store,
)
from aethergraph.storage.contracts import (
    ObservationQuery,
    PageRequest,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 21, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


def _store(
    root: Path,
) -> tuple[CanonicalMeteringStore, LocalObservationRepository, LocalSQLiteDatabase]:
    database = LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    repository = LocalObservationRepository(database=database)
    return (
        CanonicalMeteringStore(
            repository=repository,
            owner_scope=_OWNER,
            clock=lambda: _NOW,
        ),
        repository,
        database,
    )


@pytest.mark.asyncio
async def test_metering_store_projects_scope_filters_and_compatibility_metadata(
    tmp_path: Path,
) -> None:
    store, repository, database = _store(tmp_path)
    await store.append(
        {
            "event_id": "meter-1",
            "kind": "meter.llm",
            "ts": _NOW.isoformat(),
            "user_id": "user-1",
            "org_id": "org-1",
            "run_id": "run-1",
            "graph_id": "graph-1",
            "client_id": "legacy-client",
            "app_id": "legacy-app",
            "prompt_tokens": 12,
        }
    )
    await store.append(
        {
            "event_id": "meter-2",
            "kind": "meter.run",
            "user_id": "user-2",
            "org_id": "org-1",
            "status": "succeeded",
        }
    )

    rows = await store.query(kinds=["meter.llm"], user_id="user-1", org_id="org-1", limit=10)
    raw = await repository.query(
        ObservationQuery(
            scope=_OWNER,
            categories=("metering",),
            page=PageRequest(limit=10),
        )
    )

    assert rows == [
        {
            "prompt_tokens": 12,
            "kind": "meter.llm",
            "ts": _NOW.isoformat(),
            "user_id": "user-1",
            "org_id": "org-1",
            "run_id": "run-1",
            "graph_id": "graph-1",
            "client_id": "legacy-client",
            "app_id": "legacy-app",
        }
    ]
    llm = next(record for record in raw.items if record.name == "meter.llm")
    assert "app_id" not in llm.scope.as_filter()
    assert "client_id" not in llm.scope.as_filter()
    assert llm.attributes["compatibility_metadata"]["app_id"] == {
        "value": "legacy-app",
        "deprecated": True,
        "compatibility_only": True,
        "scheduled_removal": "future breaking release",
    }
    await database.close()


@pytest.mark.asyncio
async def test_metering_store_requires_explicit_bound_and_service_supplies_it(
    tmp_path: Path,
) -> None:
    store, _repository, database = _store(tmp_path)
    with pytest.raises(ValueError, match="between 1 and 10000"):
        await store.query(limit=None)

    class _CaptureStore:
        def __init__(self) -> None:
            self.limit = None

        async def query(self, **kwargs):
            self.limit = kwargs["limit"]
            return []

    capture = _CaptureStore()
    service = EventLogMeteringService(capture)
    await service._query(
        window="24h",
        kinds=["meter.llm"],
        user_id="local",
        org_id="local",
    )

    assert capture.limit == 10_000
    await database.close()


def test_metering_binding_and_provider_neutral_strict_docstrings() -> None:
    repository = object()
    store = bind_canonical_metering_store(
        bundle=SimpleNamespace(observations=repository),
        owner_scope=_OWNER,
        clock=lambda: _NOW,
    )
    source = Path(__file__).parents[1] / "src/aethergraph/services/canonical_metering.py"
    source_text = source.read_text(encoding="utf-8")

    assert store._repository is repository
    for forbidden in (
        "EventLogMeteringStore",
        "SqliteEventLog",
        "SQLiteObservationStore",
        "limit=None",
        "decode_cursor",
    ):
        assert forbidden not in source_text
    for method_name in ("__init__", "append", "query"):
        doc = getdoc(getattr(CanonicalMeteringStore, method_name)) or ""
        assert "Intro:" in doc
        assert doc.count("```python") >= 2
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "Notes:" in doc
    binding_doc = getdoc(bind_canonical_metering_store) or ""
    assert "Intro:" in binding_doc
    assert binding_doc.count("```python") >= 2
    assert "Args:" in binding_doc
    assert "Returns:" in binding_doc
    assert "Notes:" in binding_doc
