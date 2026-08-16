"""Atomic token-safe filesystem continuation store."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
import json
from pathlib import Path
from typing import Any, TypeVar

from ..continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationStatus,
    Correlator,
    CreatedContinuation,
)
from .inmem_store import InMemoryContinuationStore

_T = TypeVar("_T")


class FSContinuationStore(InMemoryContinuationStore):
    """Persist the atomic runtime contract in one replace-on-commit manifest."""

    def __init__(self, root: str | Path, secret: bytes):
        super().__init__(secret=secret)
        self.root = Path(root)
        self._manifest = self.root / "continuations.v2.json"
        self._load()

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create and durably commit one continuation.

        Intro:
            Persists the in-memory atomic mutation through one manifest replacement.

        Examples:
            Create a wait:
            ```python
            created = await store.create(draft)
            ```
            Reopen it after restart:
            ```python
            wait = await reopened.get(draft.run_id, draft.node_id)
            ```

        Args:
            draft: Immutable tokenless continuation content.

        Returns:
            CreatedContinuation: Revision-one record and one-time raw token.

        Notes:
            The manifest stores only a protected token digest and tokenless record.
        """
        return await self._mutate(lambda: super(FSContinuationStore, self).create(draft))

    async def update(self, continuation: Continuation, *, expected_revision: int) -> Continuation:
        """Compare, replace, and durably commit one revision.

        Intro:
            Couples revision validation with one atomic manifest replacement.

        Examples:
            Reschedule a timer:
            ```python
            stored = await store.update(changed, expected_revision=current.revision)
            ```
            Detect a stale write:
            ```python
            await store.update(changed, expected_revision=1)
            ```

        Args:
            continuation: Complete next continuation revision.
            expected_revision: Required current revision.

        Returns:
            Continuation: Newly committed record.

        Notes:
            Failed replacement restores the in-process snapshot before propagating.
        """
        return await self._mutate(
            lambda: super(FSContinuationStore, self).update(
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
        """Commit one terminal continuation receipt.

        Intro:
            Retains terminal state through the filesystem durability boundary.

        Examples:
            Complete a resume:
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
            Terminal records remain in the manifest as delivery receipts.
        """
        return await self._mutate(
            lambda: super(FSContinuationStore, self).close(
                continuation, status=status, closed_at=closed_at
            )
        )

    async def bind_correlator(
        self, *, continuation: Continuation, corr: Correlator
    ) -> Continuation:
        """Commit a record and exact correlator index together.

        Intro:
            Persists both sides of correlation lookup in one manifest version.

        Examples:
            Bind a message:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=message)
            ```
            Replay a binding:
            ```python
            same = await store.bind_correlator(continuation=wait, corr=message)
            ```

        Args:
            continuation: Exact current continuation record.
            corr: Exact correlator to add.

        Returns:
            Continuation: Current or newly revised durable record.

        Notes:
            Bearer tokens are not used as correlator identities.
        """
        return await self._mutate(
            lambda: super(FSContinuationStore, self).bind_correlator(
                continuation=continuation, corr=corr
            )
        )

    async def _mutate(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        with self._lock:
            snapshot = self._snapshot()
            result = await operation()
            try:
                await asyncio.to_thread(self._persist)
            except Exception:
                self._restore(snapshot)
                raise
            return result

    def _snapshot(self) -> tuple[dict, dict, dict, dict]:
        return (
            dict(self._records),
            dict(self._by_run_node),
            dict(self._by_digest),
            defaultdict(list, {key: list(value) for key, value in self._corr_index.items()}),
        )

    def _restore(self, snapshot: tuple[dict, dict, dict, dict]) -> None:
        self._records, self._by_run_node, self._by_digest, self._corr_index = snapshot

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "records": {key: value.to_dict() for key, value in self._records.items()},
            "by_run_node": [
                {"run_id": key[0], "node_id": key[1], "continuation_id": value}
                for key, value in self._by_run_node.items()
            ],
            "by_digest": self._by_digest,
            "correlators": [
                {
                    "correlator": {
                        "scheme": key[0],
                        "channel": key[1],
                        "thread": key[2],
                        "message": key[3],
                    },
                    "continuation_ids": value,
                }
                for key, value in self._corr_index.items()
            ],
        }
        temporary = self._manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._manifest)

    def _load(self) -> None:
        if not self._manifest.exists():
            return
        raw = json.loads(self._manifest.read_text(encoding="utf-8"))
        if raw.get("version") != 2:
            raise ValueError("Unsupported continuation manifest version")
        records = {str(key): _continuation(value) for key, value in raw.get("records", {}).items()}
        by_run_node = {
            (str(value["run_id"]), str(value["node_id"])): str(value["continuation_id"])
            for value in raw.get("by_run_node", ())
        }
        by_digest = {str(key): str(value) for key, value in raw.get("by_digest", {}).items()}
        correlators: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for value in raw.get("correlators", ()):
            corr = Correlator(**value["correlator"])
            correlators[corr.key()] = [str(item) for item in value["continuation_ids"]]
        self._records = records
        self._by_run_node = by_run_node
        self._by_digest = by_digest
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
