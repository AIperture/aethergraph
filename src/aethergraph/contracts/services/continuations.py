"""Frozen runtime continuation service contract."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from aethergraph.services.continuations.continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationPage,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
    CreatedContinuation,
)


class AsyncContinuationStore(Protocol):
    """Atomic token-safe continuation persistence used by the runtime."""

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically mint and persist one continuation.

        Intro:
            Defines the single creation boundary for record and secret indexes.

        Examples:
            Create an approval wait:
            ```python
            created = await store.create(draft)
            ```
            Register its one-time token:
            ```python
            future = waits.register(created.record.continuation_id)
            ```

        Args:
            draft: Immutable tokenless continuation content.

        Returns:
            CreatedContinuation: Revision-one record and one-time raw token.

        Notes:
            The raw token must not be persisted in the continuation document.
        """
        ...

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        """Read one current continuation by exact run/node identity.

        Intro:
            Retrieves a tokenless record through its scheduler-facing identity.

        Examples:
            Read a wait:
            ```python
            wait = await store.get("run-1", "approval")
            ```
            Detect absence:
            ```python
            assert await store.get("missing", "node") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.

        Returns:
            Continuation | None: Current tokenless record or `None`.

        Notes:
            This lookup never authorizes bearer-token input.
        """
        ...

    async def get_by_id(self, continuation_id: str) -> Continuation | None:
        """Read one continuation by stable identity.

        Intro:
            Retrieves a tokenless record for trusted internal delivery.

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
            Continuation | None: Current tokenless record or `None`.

        Notes:
            Trusted internal delivery uses this identity, never a persisted token.
        """
        ...

    async def resolve_token(self, token: str) -> Continuation | None:
        """Resolve one external bearer token at the authorization boundary.

        Intro:
            Delegates secret protection and exact lookup to the configured store.

        Examples:
            Resolve an inbound token:
            ```python
            wait = await store.resolve_token(token)
            ```
            Reject an unknown token:
            ```python
            assert await store.resolve_token("invalid") is None
            ```

        Args:
            token: Raw token supplied by the external caller.

        Returns:
            Continuation | None: Matching tokenless record or `None`.

        Notes:
            Providers own token protection and comparison; callers do not revalidate it.
        """
        ...

    async def update(self, continuation: Continuation, *, expected_revision: int) -> Continuation:
        """Replace one continuation with an exact next revision.

        Intro:
            Defines compare-and-set mutation for non-creation state changes.

        Examples:
            Reschedule a poll:
            ```python
            stored = await store.update(changed, expected_revision=current.revision)
            ```
            Close a delivered wait:
            ```python
            stored = await store.update(resumed, expected_revision=current.revision)
            ```

        Args:
            continuation: Complete immutable next revision.
            expected_revision: Revision that must currently be authoritative.

        Returns:
            Continuation: Newly stored authoritative revision.

        Notes:
            Identity and terminal lifecycle are immutable; stale updates fail explicitly.
        """
        ...

    async def close(
        self,
        continuation: Continuation,
        *,
        status: ContinuationStatus,
        closed_at: datetime,
    ) -> Continuation:
        """Atomically transition one waiting continuation to a terminal state.

        Intro:
            Retains a durable terminal receipt after successful or canceled delivery.

        Examples:
            Mark a successful resume:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.RESUMED, closed_at=now)
            ```
            Cancel a wait:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.CANCELED, closed_at=now)
            ```

        Args:
            continuation: Exact current tokenless record.
            status: Requested non-waiting terminal state.
            closed_at: Timezone-aware terminal transition time.

        Returns:
            Continuation: Newly stored terminal revision.

        Notes:
            Successful resumes retain a durable terminal receipt; records are not deleted.
        """
        ...

    async def bind_correlator(
        self, *, continuation: Continuation, corr: Correlator
    ) -> Continuation:
        """Atomically bind one exact correlator to a continuation.

        Intro:
            Commits record revision and reverse lookup as one logical mutation.

        Examples:
            Bind a public interaction identity:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=interaction)
            ```
            Bind a channel delivery receipt:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=message)
            ```

        Args:
            continuation: Exact current record and expected revision.
            corr: Exact correlator to bind idempotently.

        Returns:
            Continuation: Current or newly revised tokenless record.

        Notes:
            Binding uses stable continuation identity rather than bearer-token lookup.
        """
        ...

    async def query(self, query: ContinuationQuery) -> ContinuationPage:
        """Execute one bounded continuation query.

        Intro:
            Selects indexed due, interaction, session, kind, or status records.

        Examples:
            Query due timers:
            ```python
            page = await store.query(ContinuationQuery(due_at_or_before=now, limit=50))
            ```
            Query one interaction:
            ```python
            page = await store.query(ContinuationQuery(correlator=interaction, limit=2))
            ```

        Args:
            query: Exact indexed filters, bound, and optional cursor.

        Returns:
            ContinuationPage: Bounded records and optional continuation cursor.

        Notes:
            Implementations must not replace indexed filters with unbounded global scans.
        """
        ...
