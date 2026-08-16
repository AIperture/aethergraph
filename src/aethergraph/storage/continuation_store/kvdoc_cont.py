"""Single-document atomic continuation store for legacy DocStore deployments."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeVar

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.kv import AsyncKV
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.continuations.continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationPage,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
    CreatedContinuation,
)
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore

_T = TypeVar("_T")


class KVDocContinuationStore(InMemoryContinuationStore, AsyncContinuationStore):
    """Persist the token-safe runtime contract in one atomic document."""

    def __init__(
        self,
        *,
        doc_store: DocStore,
        kv: AsyncKV,
        event_log: EventLog | None = None,
        secret: bytes,
        namespace: str = "cont",
    ):
        super().__init__(secret=secret)
        self._docs = doc_store
        self._kv = kv
        self._log = event_log
        self._ns = namespace.rstrip("/")
        self._manifest_id = f"{self._ns}/continuations-v2"
        self._loaded = False
        self._load_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create and persist a token-safe continuation.

        Intro:
            Commits record and protected indexes in one versioned document.

        Examples:
            Create a wait:
            ```python
            created = await store.create(draft)
            ```
            Resolve its token:
            ```python
            wait = await store.resolve_token(created.token)
            ```

        Args:
            draft: Immutable tokenless continuation content.

        Returns:
            CreatedContinuation: Revision-one record and one-time raw token.

        Notes:
            The KV dependency is retained for constructor compatibility but stores no token index.
        """
        return await self._mutate(lambda: super(KVDocContinuationStore, self).create(draft))

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        """Read one exact tokenless continuation.

        Intro:
            Loads the manifest once and resolves the scheduler-facing identity.

        Examples:
            Read a wait:
            ```python
            wait = await store.get("run-1", "node-1")
            ```
            Detect absence:
            ```python
            assert await store.get("missing", "node") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.

        Returns:
            Continuation | None: Current record or `None`.

        Notes:
            The manifest is loaded once before serving reads.
        """
        await self._ensure_loaded()
        return await super().get(run_id, node_id)

    async def get_by_id(self, continuation_id: str) -> Continuation | None:
        """Read one record by stable continuation identity.

        Intro:
            Resolves trusted delivery identity from the loaded manifest.

        Examples:
            Read a timer candidate:
            ```python
            wait = await store.get_by_id("cont-1")
            ```
            Detect absence:
            ```python
            assert await store.get_by_id("missing") is None
            ```

        Args:
            continuation_id: Stable continuation identity.

        Returns:
            Continuation | None: Current record or `None`.

        Notes:
            No bearer token is required for trusted internal lookup.
        """
        await self._ensure_loaded()
        return await super().get_by_id(continuation_id)

    async def resolve_token(self, token: str) -> Continuation | None:
        """Resolve an external token through the protected manifest index.

        Intro:
            Hashes bearer material before consulting the persisted index.

        Examples:
            Resolve an issued token:
            ```python
            wait = await store.resolve_token(created.token)
            ```
            Reject an invalid token:
            ```python
            assert await store.resolve_token("invalid") is None
            ```

        Args:
            token: Raw external bearer token.

        Returns:
            Continuation | None: Matching tokenless record or `None`.

        Notes:
            Raw token material is not stored in DocStore or AsyncKV.
        """
        await self._ensure_loaded()
        return await super().resolve_token(token)

    async def update(self, continuation: Continuation, *, expected_revision: int) -> Continuation:
        """Compare, replace, and persist one exact revision.

        Intro:
            Couples compare-and-set semantics to a single document write.

        Examples:
            Reschedule a wait:
            ```python
            stored = await store.update(changed, expected_revision=current.revision)
            ```
            Detect conflict:
            ```python
            await store.update(changed, expected_revision=1)
            ```

        Args:
            continuation: Complete next revision.
            expected_revision: Required authoritative revision.

        Returns:
            Continuation: Newly persisted record.

        Notes:
            A failed document write restores the in-process snapshot.
        """
        return await self._mutate(
            lambda: super(KVDocContinuationStore, self).update(
                continuation, expected_revision=expected_revision
            )
        )

    async def close(
        self,
        continuation: Continuation,
        *,
        status: ContinuationStatus,
        closed_at: datetime,
    ) -> Continuation:
        """Persist a retained terminal continuation receipt.

        Intro:
            Records terminal lifecycle without deleting continuation identity.

        Examples:
            Mark a resume:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.RESUMED, closed_at=now)
            ```
            Cancel a wait:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.CANCELED, closed_at=now)
            ```

        Args:
            continuation: Exact current waiting record.
            status: Requested terminal status.
            closed_at: Timezone-aware terminal time.

        Returns:
            Continuation: Durable terminal record.

        Notes:
            Terminal records are retained instead of physically deleted.
        """
        return await self._mutate(
            lambda: super(KVDocContinuationStore, self).close(
                continuation, status=status, closed_at=closed_at
            )
        )

    async def bind_correlator(
        self, *, continuation: Continuation, corr: Correlator
    ) -> Continuation:
        """Persist an exact correlator and record revision together.

        Intro:
            Stores both correlation directions in one manifest mutation.

        Examples:
            Bind an interaction:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=interaction)
            ```
            Replay a binding:
            ```python
            same = await store.bind_correlator(continuation=wait, corr=interaction)
            ```

        Args:
            continuation: Exact current continuation record.
            corr: Exact correlator to add.

        Returns:
            Continuation: Current or newly revised durable record.

        Notes:
            Correlator lookup uses stable continuation identity, not bearer tokens.
        """
        return await self._mutate(
            lambda: super(KVDocContinuationStore, self).bind_correlator(
                continuation=continuation, corr=corr
            )
        )

    async def query(self, query: ContinuationQuery) -> ContinuationPage:
        """Execute one bounded manifest query.

        Intro:
            Applies explicit filters and limits after one manifest load.

        Examples:
            Query timers:
            ```python
            page = await store.query(ContinuationQuery(due_at_or_before=now, limit=50))
            ```
            Query an interaction:
            ```python
            page = await store.query(ContinuationQuery(correlator=corr, limit=2))
            ```

        Args:
            query: Exact filters, bound, and optional cursor.

        Returns:
            ContinuationPage: Bounded records and optional cursor.

        Notes:
            This legacy backend reads one bounded-domain manifest; canonical providers use indexes.
        """
        await self._ensure_loaded()
        return await super().query(query)

    async def _mutate(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        await self._ensure_loaded()
        async with self._mutation_lock:
            snapshot = self._snapshot()
            result = await operation()
            try:
                await self._docs.put(self._manifest_id, self._payload())
            except Exception:
                self._restore(snapshot)
                raise
            return result

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            raw = await self._docs.get(self._manifest_id)
            if raw is not None:
                self._hydrate(raw)
            self._loaded = True

    def _snapshot(self) -> tuple[dict, dict, dict, dict]:
        return (
            dict(self._records),
            dict(self._by_run_node),
            dict(self._by_digest),
            defaultdict(list, {key: list(value) for key, value in self._corr_index.items()}),
        )

    def _restore(self, snapshot: tuple[dict, dict, dict, dict]) -> None:
        self._records, self._by_run_node, self._by_digest, self._corr_index = snapshot

    def _payload(self) -> dict[str, Any]:
        return {
            "version": 2,
            "records": {key: value.to_dict() for key, value in self._records.items()},
            "by_run_node": [
                {"run_id": key[0], "node_id": key[1], "continuation_id": value}
                for key, value in self._by_run_node.items()
            ],
            "by_digest": dict(self._by_digest),
            "correlators": [
                {
                    "correlator": {
                        "scheme": key[0],
                        "channel": key[1],
                        "thread": key[2],
                        "message": key[3],
                    },
                    "continuation_ids": list(value),
                }
                for key, value in self._corr_index.items()
            ],
        }

    def _hydrate(self, raw: dict[str, Any]) -> None:
        if raw.get("version") != 2:
            raise ValueError("Unsupported continuation manifest version")
        self._records = {
            str(key): _continuation(value) for key, value in raw.get("records", {}).items()
        }
        self._by_run_node = {
            (str(value["run_id"]), str(value["node_id"])): str(value["continuation_id"])
            for value in raw.get("by_run_node", ())
        }
        self._by_digest = {str(key): str(value) for key, value in raw.get("by_digest", {}).items()}
        correlators: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for value in raw.get("correlators", ()):
            corr = Correlator(**value["correlator"])
            correlators[corr.key()] = [str(item) for item in value["continuation_ids"]]
        self._corr_index = defaultdict(list, correlators)


def _continuation(raw: dict[str, Any]) -> Continuation:
    value = dict(raw)
    for key in ("created_at", "deadline", "next_wakeup_at", "closed_at"):
        if value.get(key):
            value[key] = datetime.fromisoformat(value[key])
    value["status"] = ContinuationStatus(value["status"])
    value["correlators"] = tuple(Correlator(**item) for item in value.pop("correlators", ()))
    metadata = value.pop("metadata", {})
    compatibility = metadata.get("compatibility_metadata", {})
    app_envelope = compatibility.get("app_id") if isinstance(compatibility, dict) else None
    value["app_id"] = app_envelope.get("value") if isinstance(app_envelope, dict) else None
    return Continuation(**value)
