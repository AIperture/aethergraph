from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from storage_conformance.suite import check_search_backend_conformance

from aethergraph.storage.contracts import (
    SearchDocument,
    SearchMode,
    SearchQuery,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalSearchBackend,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 20, tzinfo=UTC)
TOKENS = ("canonical", "storage", "migration", "provider", "contract", "conformance")


class _Embedder:
    def __init__(self) -> None:
        self.dimension = len(TOKENS) + 1
        self.fail = False

    async def embed(self, texts, **_kwargs):
        if self.fail:
            raise RuntimeError("embedding failed")
        return [
            [float(text.lower().split().count(token)) for token in TOKENS]
            + [1.0] * (self.dimension - len(TOKENS))
            for text in texts
        ]


class _WrongCountEmbedder:
    async def embed(self, texts, **_kwargs):
        return [[1.0, 0.0]] if len(texts) > 1 else []


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.SEARCH,
        mode=mode,
    )


def _document(
    item_id: str,
    scope: StorageScope,
    text: str,
    *,
    occurred_at: datetime = NOW,
    metadata: dict[str, object] | None = None,
) -> SearchDocument:
    return SearchDocument(
        corpus="memory",
        item_id=item_id,
        text=text,
        scope=scope,
        occurred_at=occurred_at,
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_local_search_backend_passes_shared_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    search = LocalSearchBackend(database=database, embedder=_Embedder())

    await check_search_backend_conformance(search)

    await database.close()


@pytest.mark.asyncio
async def test_exact_modes_apply_scope_metadata_and_time_before_ranking(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    search = LocalSearchBackend(database=database, embedder=_Embedder())
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_scope = replace(scope, project_id="project-2")
    documents = (
        _document(
            "older",
            scope,
            "canonical storage migration",
            occurred_at=NOW - timedelta(hours=1),
            metadata={"kind": "note", "labels": ["storage", "migration"]},
        ),
        _document(
            "newer",
            scope,
            "canonical provider contract",
            metadata={"kind": "note", "labels": ["provider"]},
        ),
        _document(
            "other-kind",
            scope,
            "canonical storage migration",
            metadata={"kind": "log"},
        ),
        _document(
            "hidden",
            other_scope,
            "canonical storage migration",
            metadata={"kind": "note"},
        ),
    )
    cursor = await search.upsert_many(documents)
    assert cursor is not None

    structural = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.STRUCTURAL,
            scope=scope,
            metadata={"kind": "note"},
            occurred_at_min=NOW - timedelta(hours=2),
            require_indexed_cursor=cursor,
        )
    )
    assert [row.item_id for row in structural] == ["newer", "older"]
    lexical = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.LEXICAL,
            scope=scope,
            query="storage migration",
            metadata={"kind": "note"},
        )
    )
    assert [row.item_id for row in lexical] == ["older"]
    semantic = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.SEMANTIC,
            scope=scope,
            query="storage migration",
            metadata={"kind": "note"},
        )
    )
    assert semantic[0].item_id == "older"
    hybrid = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.HYBRID,
            scope=scope,
            query="storage migration",
            metadata={"kind": "note"},
        )
    )
    assert hybrid[0].item_id == "older"
    assert {row.mode for row in (*structural, *lexical, *semantic, *hybrid)} == {
        SearchMode.STRUCTURAL,
        SearchMode.LEXICAL,
        SearchMode.SEMANTIC,
        SearchMode.HYBRID,
    }
    await database.close()


@pytest.mark.asyncio
async def test_upsert_is_idempotent_and_embedding_failure_is_atomic(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    embedder = _Embedder()
    search = LocalSearchBackend(database=database, embedder=embedder)
    scope = StorageScope(project_id="project-1")
    first = _document("first", scope, "canonical storage")

    first_cursor = await search.upsert(first)
    assert await search.upsert(first) == first_cursor
    changed_cursor = await search.upsert(replace(first, text="canonical migration"))
    assert changed_cursor != first_cursor
    assert (
        await search.query(
            SearchQuery(
                corpus="memory",
                mode=SearchMode.LEXICAL,
                scope=scope,
                query="storage",
            )
        )
        == ()
    )
    assert [
        row.item_id
        for row in await search.query(
            SearchQuery(
                corpus="memory",
                mode=SearchMode.LEXICAL,
                scope=scope,
                query="migration",
            )
        )
    ] == ["first"]

    embedder.fail = True
    with pytest.raises(RuntimeError, match="embedding failed"):
        await search.upsert(_document("failed", scope, "provider contract"))
    embedder.fail = False
    assert await search.indexed_cursor("memory") == changed_cursor
    rows = await search.query(SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope))
    assert [row.item_id for row in rows] == ["first"]
    await database.close()


@pytest.mark.asyncio
async def test_invalid_batches_and_vectors_publish_nothing(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    scope = StorageScope(project_id="project-1")
    search = LocalSearchBackend(database=database, embedder=_Embedder())
    with pytest.raises(StorageConfigurationError, match="one exact corpus"):
        await search.upsert_many(
            (
                _document("first", scope, "one"),
                replace(_document("second", scope, "two"), corpus="other"),
            )
        )
    assert await search.indexed_cursor("memory") is None

    invalid = LocalSearchBackend(database=database, embedder=_WrongCountEmbedder())
    with pytest.raises(StorageIntegrityError, match="count"):
        await invalid.upsert_many(
            (
                _document("first", scope, "one"),
                _document("second", scope, "two"),
            )
        )
    assert await search.indexed_cursor("memory") is None
    await database.close()


@pytest.mark.asyncio
async def test_missing_embedder_never_falls_back_from_semantic_or_hybrid(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    search = LocalSearchBackend(database=database, embedder=None)
    scope = StorageScope(project_id="project-1")
    await search.upsert(_document("first", scope, "canonical storage migration"))

    lexical = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.LEXICAL,
            scope=scope,
            query="canonical migration",
        )
    )
    assert [row.item_id for row in lexical] == ["first"]
    for mode, capability in (
        (SearchMode.SEMANTIC, "search_semantic"),
        (SearchMode.HYBRID, "search_hybrid"),
    ):
        with pytest.raises(StorageCapabilityError) as failure:
            await search.query(
                SearchQuery(
                    corpus="memory",
                    mode=mode,
                    scope=scope,
                    query="canonical migration",
                )
            )
        assert failure.value.missing == (capability,)
    await database.close()


@pytest.mark.asyncio
async def test_delete_and_freshness_cursors_are_scope_and_corpus_bound(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    search = LocalSearchBackend(database=database, embedder=_Embedder())
    scope = StorageScope(project_id="project-1")
    other_scope = StorageScope(project_id="project-2")
    first_cursor = await search.upsert_many(
        (
            _document("shared", scope, "canonical storage"),
            _document("shared", other_scope, "provider contract"),
        )
    )
    assert first_cursor is not None

    delete_cursor = await search.delete(scope, "memory", ("shared",))
    assert delete_cursor is not None and delete_cursor != first_cursor
    assert await search.delete(scope, "memory", ("shared",)) == delete_cursor
    assert (
        await search.query(SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope))
        == ()
    )
    remaining = await search.query(
        SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=other_scope)
    )
    assert [row.item_id for row in remaining] == ["shared"]

    other_cursor = await search.upsert(
        replace(_document("other", scope, "canonical"), corpus="other")
    )
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await search.wait_until_indexed("memory", other_cursor, 0.0)
    with pytest.raises(StorageConfigurationError, match="Invalid"):
        await search.wait_until_indexed("memory", "not-a-cursor", 0.0)
    await database.close()


@pytest.mark.asyncio
async def test_concurrent_writes_have_unique_monotonic_corpus_cursors(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    search = LocalSearchBackend(database=database, embedder=_Embedder())
    scope = StorageScope(project_id="project-1")

    cursors = await asyncio.gather(
        *(
            search.upsert(_document(f"item-{index}", scope, f"canonical {index}"))
            for index in range(20)
        )
    )

    assert len(set(cursors)) == 20
    assert await search.indexed_cursor("memory") in cursors
    rows = await search.query(
        SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope, top_k=20)
    )
    assert len(rows) == 20
    await database.close()


@pytest.mark.asyncio
async def test_schema_is_canonical_indexed_and_vector_corruption_fails_typed(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    embedder = _Embedder()
    search = LocalSearchBackend(database=database, embedder=embedder)
    scope = StorageScope(project_id="project-1")
    await search.upsert(_document("first", scope, "canonical storage"))

    columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_search_documents)")
    }
    assert not {"app_id", "application_id", "client_id", "scope_id"} & columns
    query_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM local_search_documents
            WHERE corpus = ? AND scope_identity = ?
            ORDER BY occurred_at DESC, document_id DESC LIMIT ?
            """,
            ("memory", '{"project_id":"project-1"}', 10),
        )
    )
    assert "ix_local_search_scope_time" in query_plan
    assert "SCAN local_search_documents" not in query_plan

    embedder.dimension += 1
    with pytest.raises(StorageIntegrityError, match="dimension differs"):
        await search.query(
            SearchQuery(
                corpus="memory",
                mode=SearchMode.SEMANTIC,
                scope=scope,
                query="canonical",
            )
        )
    await database.close()


@pytest.mark.asyncio
async def test_read_only_search_reads_and_rejects_mutation(tmp_path: Path) -> None:
    scope = StorageScope(project_id="project-1")
    document = _document("first", scope, "canonical storage migration")
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable = LocalSearchBackend(database=writable_database, embedder=_Embedder())
    cursor = await writable.upsert(document)
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = LocalSearchBackend(database=readonly_database, embedder=_Embedder())
    assert await readonly.indexed_cursor("memory") == cursor
    assert [
        row.item_id
        for row in await readonly.query(
            SearchQuery(
                corpus="memory",
                mode=SearchMode.LEXICAL,
                scope=scope,
                query="canonical migration",
            )
        )
    ] == ["first"]
    with pytest.raises(StorageReadOnlyError):
        await readonly.upsert(_document("new", scope, "provider contract"))
    with pytest.raises(StorageReadOnlyError):
        await readonly.delete(scope, "memory", ("first",))
    await readonly_database.close()
