"""Compact observation selection excludes payloads before hydration."""

from dataclasses import replace

import pytest
from test_local_observation_repository import SCOPE, _database, _observation

from aethergraph.storage.contracts import (
    ObservationQuery,
    PageRequest,
    StorageMigrationRequiredError,
    StorageOpenMode,
)
from aethergraph.storage.providers.local_sqlite import LocalObservationRepository


@pytest.mark.asyncio
async def test_compact_indexed_observations_exclude_large_attributes(tmp_path, monkeypatch):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    try:
        await repository.append_many(
            tuple(
                _observation(
                    "span-" + str(index),
                    attributes={
                        "duration_ms": index * 10,
                        "error": {"code": "E"},
                        "large": "secret" * 100000,
                    },
                )
                for index in range(5)
            )
        )
        import aethergraph.storage.providers.local_sqlite.observation_repository as owner

        original = owner._json_object

        def bounded(value):
            assert len(value) < 200
            return original(value)

        monkeypatch.setattr(owner, "_json_object", bounded)
        query = ObservationQuery(
            scope=SCOPE,
            page=PageRequest(limit=1),
            include_payload_metadata=False,
            error_codes=("E",),
            duration_ms_at_least=10,
            duration_ms_at_most=30,
        )
        identities = []
        while True:
            page = await repository.query(query)
            identities.extend(record.observation_id for record in page.items)
            assert all(
                not record.resource_links and "large" not in record.attributes
                for record in page.items
            )
            assert len(page.item_cursors) == len(page.items)
            if not page.next_cursor:
                break
            query = replace(query, page=PageRequest(limit=1, cursor=page.next_cursor))
        assert identities == ["span-3", "span-2", "span-1"]
        exact = await repository.query(
            ObservationQuery(
                scope=SCOPE, observation_ids=("span-4",), include_payload_metadata=False
            )
        )
        assert exact.items[0].attributes["duration_ms"] == 40
        assert not (
            await repository.query(
                ObservationQuery(
                    scope=replace(SCOPE, run_id="foreign"),
                    observation_ids=("span-4",),
                    include_payload_metadata=False,
                )
            )
        ).items
        plans = await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT observation_id FROM local_observations INDEXED BY ix_local_observations_duration WHERE run_id=? AND json_extract(attributes_json, '$.duration_ms') >= ?",
            (SCOPE.run_id, 10),
        )
        assert any(
            "SEARCH" in row["detail"] and "ix_local_observations_duration" in row["detail"]
            for row in plans
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_old_readonly_observation_workspace_needs_explicit_scalar_index_preparation(tmp_path):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalObservationRepository(database=database)
    await database.transaction(
        lambda connection: connection.execute(
            "DELETE FROM ag_storage_components WHERE name='observation_compact_indexes'"
        )
    )
    await database.close()
    database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    repository = LocalObservationRepository(database=database)
    try:
        assert not (
            await repository.query(ObservationQuery(scope=SCOPE, include_payload_metadata=False))
        ).items
        with pytest.raises(StorageMigrationRequiredError, match="Prepare"):
            await repository.query(ObservationQuery(scope=SCOPE, duration_ms_at_least=10))
    finally:
        await database.close()


@pytest.mark.parametrize(
    "values",
    [
        {"duration_ms_at_least": -1},
        {"duration_ms_at_most": float("nan")},
        {"duration_ms_at_least": 10, "duration_ms_at_most": 9},
        {"include_payload_metadata": 1},
        {"observation_ids": ("",)},
    ],
)
def test_compact_query_validates_selection(values):
    with pytest.raises((ValueError, TypeError)):
        ObservationQuery(scope=SCOPE, **values)
