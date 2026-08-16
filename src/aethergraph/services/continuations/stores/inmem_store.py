"""Token-safe in-memory continuation store for tests and embedded use."""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import hmac
import os
import threading

from ..continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationPage,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
    CreatedContinuation,
)


class InMemoryContinuationStore:
    """Atomic in-process implementation of the runtime continuation contract."""

    def __init__(self, secret: bytes | None = None):
        self.secret = secret or os.urandom(32)
        self._records: dict[str, Continuation] = {}
        self._by_run_node: dict[tuple[str, str], str] = {}
        self._by_digest: dict[str, str] = {}
        self._corr_index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def _hmac(self, *parts: str) -> str:
        digest = hmac.new(self.secret, digestmod=hashlib.sha256)
        for part in parts:
            digest.update(part.encode("utf-8"))
        return digest.hexdigest()

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create a record and its protected token index.

        Intro:
            Commits the tokenless record and all lookup indexes under one lock.

        Examples:
            Create a wait:
            ```python
            created = await store.create(draft)
            ```
            Read its tokenless record:
            ```python
            assert created.record == await store.get(draft.run_id, draft.node_id)
            ```

        Args:
            draft: Immutable tokenless continuation content.

        Returns:
            CreatedContinuation: Revision-one record and one-time raw token.

        Notes:
            Identity collision fails before any index is changed.
        """
        token = self._hmac(
            draft.continuation_id,
            draft.run_id,
            draft.node_id,
            str(draft.attempts),
            os.urandom(16).hex(),
        )
        digest = self._hmac("token", token)
        record = Continuation(
            continuation_id=draft.continuation_id,
            revision=1,
            run_id=draft.run_id,
            node_id=draft.node_id,
            kind=draft.kind,
            prompt=draft.prompt,
            resume_schema=draft.resume_schema,
            deadline=draft.deadline,
            poll=draft.poll,
            next_wakeup_at=draft.next_wakeup_at,
            attempts=draft.attempts,
            channel=draft.channel,
            created_at=draft.created_at,
            payload=draft.payload,
            session_id=draft.session_id,
            agent_id=draft.agent_id,
            app_id=draft.app_id,
            graph_id=draft.graph_id,
            correlators=draft.correlators,
        )
        with self._lock:
            key = (record.run_id, record.node_id)
            if record.continuation_id in self._records or key in self._by_run_node:
                raise ValueError("Continuation identity conflicts with an existing record")
            self._records[record.continuation_id] = record
            self._by_run_node[key] = record.continuation_id
            self._by_digest[digest] = record.continuation_id
            for corr in record.correlators:
                self._corr_index[corr.key()].append(record.continuation_id)
        return CreatedContinuation(record=record, token=token)

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        """Read one current record by run and node.

        Intro:
            Resolves the exact scheduler-facing identity to its current record.

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
            Continuation | None: Tokenless record or `None`.

        Notes:
            Terminal records remain readable as durable receipts.
        """
        with self._lock:
            continuation_id = self._by_run_node.get((run_id, node_id))
            return self._records.get(continuation_id) if continuation_id else None

    async def get_by_id(
        self, run_id: str, node_id: str, continuation_id: str
    ) -> Continuation | None:
        """Read one current record by stable continuation identity.

        Intro:
            Resolves trusted internal delivery identity without token material.

        Examples:
            Read a candidate:
            ```python
            wait = await store.get_by_id("run-1", "node-1", "cont-1")
            ```
            Detect absence:
            ```python
            assert await store.get_by_id("run-1", "node-1", "missing") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.
            continuation_id: Stable continuation identity.

        Returns:
            Continuation | None: Tokenless record or `None`.

        Notes:
            No token index participates in this trusted lookup.
        """
        with self._lock:
            record = self._records.get(continuation_id)
            if record is None or (record.run_id, record.node_id) != (run_id, node_id):
                return None
            return record

    async def resolve_token(self, token: str) -> Continuation | None:
        """Resolve an external bearer token through its protected digest.

        Intro:
            Hashes the supplied token and returns only the tokenless record.

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
            Raw token material is neither a mapping key nor a record field.
        """
        digest = self._hmac("token", token)
        with self._lock:
            continuation_id = self._by_digest.get(digest)
            return self._records.get(continuation_id) if continuation_id else None

    async def update(self, continuation: Continuation, *, expected_revision: int) -> Continuation:
        """Compare and set one complete continuation revision.

        Intro:
            Applies one immutable revision after validating identity and lifecycle.

        Examples:
            Reschedule a wait:
            ```python
            stored = await store.update(changed, expected_revision=current.revision)
            ```
            Detect a stale writer:
            ```python
            await store.update(changed, expected_revision=1)
            ```

        Args:
            continuation: Complete next revision.
            expected_revision: Required current revision.

        Returns:
            Continuation: Newly authoritative record.

        Notes:
            Stable identity, scope, creation time, and terminal state cannot change.
        """
        with self._lock:
            current = self._records.get(continuation.continuation_id)
            if current is None:
                raise KeyError(continuation.continuation_id)
            if current.revision != expected_revision:
                raise RuntimeError("Continuation revision conflict")
            if continuation.revision != expected_revision + 1:
                raise ValueError("Continuation revision must advance by one")
            if _identity(current) != _identity(continuation):
                raise ValueError("Continuation identity is immutable")
            if current.closed:
                raise RuntimeError("Terminal continuation cannot be updated")
            self._records[continuation.continuation_id] = continuation
            return continuation

    async def close(
        self,
        continuation: Continuation,
        *,
        status: ContinuationStatus,
        closed_at: datetime,
    ) -> Continuation:
        """Transition one waiting continuation to a retained terminal receipt.

        Intro:
            Advances lifecycle once while preserving the continuation record.

        Examples:
            Mark delivery complete:
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
            closed_at: Timezone-aware close time.

        Returns:
            Continuation: Newly stored terminal record.

        Notes:
            Repeating the same transition with the current terminal record is idempotent.
        """
        if status is ContinuationStatus.WAITING:
            raise ValueError("close requires a terminal continuation status")
        if closed_at.tzinfo is None:
            raise ValueError("closed_at must be timezone-aware")
        with self._lock:
            current = self._records.get(continuation.continuation_id)
            if current is None:
                raise KeyError(continuation.continuation_id)
            if current.closed:
                if current.status is status:
                    return current
                raise RuntimeError("Continuation is already terminal")
            if current.revision != continuation.revision:
                raise RuntimeError("Continuation revision conflict")
            changed = replace(
                current,
                revision=current.revision + 1,
                status=status,
                closed_at=closed_at.astimezone(UTC),
            )
            self._records[current.continuation_id] = changed
            return changed

    async def bind_correlator(
        self, *, continuation: Continuation, corr: Correlator
    ) -> Continuation:
        """Bind an exact correlator using stable continuation identity.

        Intro:
            Adds a correlator and reverse index in one locked mutation.

        Examples:
            Bind a message receipt:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=message)
            ```
            Replay an existing binding:
            ```python
            same = await store.bind_correlator(continuation=wait, corr=message)
            ```

        Args:
            continuation: Exact current continuation record.
            corr: Exact correlator value.

        Returns:
            Continuation: Current record for a replay or newly revised record.

        Notes:
            The record and reverse lookup become visible under the same lock.
        """
        with self._lock:
            current = self._records.get(continuation.continuation_id)
            if current is None:
                raise KeyError(continuation.continuation_id)
            if corr in current.correlators:
                return current
            if current.revision != continuation.revision or current.closed:
                raise RuntimeError("Continuation revision conflict")
            changed = replace(
                current,
                revision=current.revision + 1,
                correlators=(*current.correlators, corr),
            )
            self._records[current.continuation_id] = changed
            self._corr_index[corr.key()].append(current.continuation_id)
            return changed

    async def query(self, query: ContinuationQuery) -> ContinuationPage:
        """Return one bounded filtered continuation page.

        Intro:
            Applies the runtime query shape with an explicit maximum result bound.

        Examples:
            Query due waits:
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
            ContinuationPage: Bounded matching records and next cursor.

        Notes:
            The in-memory index is test infrastructure; production providers own query plans.
        """
        with self._lock:
            if query.correlator is not None:
                identities = tuple(self._corr_index.get(query.correlator.key(), ()))
                records = [self._records[value] for value in identities]
            else:
                records = list(self._records.values())
        records = [value for value in records if _matches(value, query)]
        if query.due_at_or_before is not None:
            records.sort(key=lambda value: (value.next_wakeup_at, value.continuation_id))
        else:
            records.sort(key=lambda value: (value.created_at, value.continuation_id), reverse=True)
        offset = _decode_cursor(query.cursor)
        visible = records[offset : offset + query.limit]
        next_offset = offset + len(visible)
        cursor = _encode_cursor(next_offset) if next_offset < len(records) else None
        return ContinuationPage(items=tuple(visible), next_cursor=cursor)


def _identity(value: Continuation) -> tuple[object, ...]:
    return (
        value.continuation_id,
        value.run_id,
        value.node_id,
        value.created_at,
        value.session_id,
        value.agent_id,
        value.graph_id,
        value.app_id,
    )


def _matches(value: Continuation, query: ContinuationQuery) -> bool:
    if query.statuses and value.status not in query.statuses:
        return False
    if query.kinds and value.kind not in query.kinds:
        return False
    if query.session_id is not None and value.session_id != query.session_id:
        return False
    if query.correlator is not None and query.correlator not in value.correlators:
        return False
    if query.due_at_or_before is not None and (
        value.next_wakeup_at is None or value.next_wakeup_at > query.due_at_or_before
    ):
        return False
    if query.open_at is not None and value.deadline is not None and value.deadline < query.open_at:
        return False
    return True


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii")


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        value = int(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid continuation cursor") from exc
    if value < 0:
        raise ValueError("Invalid continuation cursor")
    return value
