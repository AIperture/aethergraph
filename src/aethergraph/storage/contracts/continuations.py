"""Canonical continuation and durable delivery-lease contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .pagination import Page, PageRequest
from .records import FrozenJson, _freeze_mapping, _nonempty, _optional_nonempty, _utc
from .scope import StorageScope


class ContinuationStatus(StrEnum):
    """Canonical lifecycle state for a durable continuation."""

    WAITING = "waiting"
    RESUMED = "resumed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class ContinuationLeaseStatus(StrEnum):
    """Canonical delivery state for one scheduled continuation fire."""

    LEASED = "leased"
    RETRY = "retry"
    DELIVERED = "delivered"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True, slots=True)
class ContinuationCorrelator:
    """Transport correlation identity stored without transport-specific behavior."""

    scheme: str
    channel: str
    thread: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        _nonempty("scheme", self.scheme)
        _nonempty("channel", self.channel)
        for name in ("thread", "message"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationDraft:
    """Canonical continuation content before token material is minted."""

    continuation_id: str
    kind: str
    scope: StorageScope
    created_at: datetime
    prompt: str | None = None
    resume_schema: Mapping[str, FrozenJson] = field(default_factory=dict)
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    poll_payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    deadline: datetime | None = None
    next_wakeup_at: datetime | None = None
    channel: str | None = None
    correlators: tuple[ContinuationCorrelator, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_continuation_identity(
            continuation_id=self.continuation_id,
            kind=self.kind,
            scope=self.scope,
            created_at=self.created_at,
            prompt=self.prompt,
            channel=self.channel,
            correlators=self.correlators,
            schema_version=self.schema_version,
        )
        _validate_schedule(
            created_at=self.created_at,
            deadline=self.deadline,
            next_wakeup_at=self.next_wakeup_at,
        )
        object.__setattr__(
            self,
            "resume_schema",
            _freeze_mapping(self.resume_schema, path="resume_schema"),
        )
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))
        object.__setattr__(
            self,
            "poll_payload",
            _freeze_mapping(self.poll_payload, path="poll_payload"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="metadata"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationRecord:
    """Revisioned continuation with a non-secret token digest and indexes."""

    continuation_id: str
    kind: str
    scope: StorageScope
    created_at: datetime
    token_digest: str
    revision: int
    status: ContinuationStatus = ContinuationStatus.WAITING
    prompt: str | None = None
    resume_schema: Mapping[str, FrozenJson] = field(default_factory=dict)
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    poll_payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    deadline: datetime | None = None
    next_wakeup_at: datetime | None = None
    channel: str | None = None
    correlators: tuple[ContinuationCorrelator, ...] = ()
    attempts: int = 0
    closed_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_continuation_identity(
            continuation_id=self.continuation_id,
            kind=self.kind,
            scope=self.scope,
            created_at=self.created_at,
            prompt=self.prompt,
            channel=self.channel,
            correlators=self.correlators,
            schema_version=self.schema_version,
        )
        _nonempty("token_digest", self.token_digest)
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.status, ContinuationStatus):
            raise TypeError("status must be a ContinuationStatus")
        if isinstance(self.attempts, bool) or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        _validate_schedule(
            created_at=self.created_at,
            deadline=self.deadline,
            next_wakeup_at=self.next_wakeup_at,
        )
        if self.status is ContinuationStatus.WAITING:
            if self.closed_at is not None:
                raise ValueError("waiting continuations must not have closed_at")
        elif self.closed_at is None:
            raise ValueError("terminal continuations require closed_at")
        if self.closed_at is not None:
            _utc("closed_at", self.closed_at)
            if self.closed_at < self.created_at:
                raise ValueError("closed_at must not precede created_at")
        object.__setattr__(
            self,
            "resume_schema",
            _freeze_mapping(self.resume_schema, path="resume_schema"),
        )
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))
        object.__setattr__(
            self,
            "poll_payload",
            _freeze_mapping(self.poll_payload, path="poll_payload"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="metadata"))


def _validate_continuation_identity(
    *,
    continuation_id: str,
    kind: str,
    scope: StorageScope,
    created_at: datetime,
    prompt: str | None,
    channel: str | None,
    correlators: tuple[ContinuationCorrelator, ...],
    schema_version: int,
) -> None:
    _nonempty("continuation_id", continuation_id)
    _nonempty("kind", kind)
    scope.require("run_id", "node_id")
    _utc("created_at", created_at)
    _optional_nonempty("prompt", prompt)
    _optional_nonempty("channel", channel)
    if not isinstance(correlators, tuple):
        raise TypeError("correlators must be an immutable tuple")
    if any(not isinstance(value, ContinuationCorrelator) for value in correlators):
        raise TypeError("correlators must contain ContinuationCorrelator values")
    if len(set(correlators)) != len(correlators):
        raise ValueError("correlators must not contain duplicates")
    if isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")


def _validate_schedule(
    *,
    created_at: datetime,
    deadline: datetime | None,
    next_wakeup_at: datetime | None,
) -> None:
    for name, value in (("deadline", deadline), ("next_wakeup_at", next_wakeup_at)):
        if value is None:
            continue
        _utc(name, value)
        if value < created_at:
            raise ValueError(f"{name} must not precede created_at")


@dataclass(frozen=True, slots=True)
class CreatedContinuation:
    """Atomic continuation-creation result containing the one-time raw token."""

    record: ContinuationRecord
    token: str

    def __post_init__(self) -> None:
        _nonempty("token", self.token)
        if self.record.status is not ContinuationStatus.WAITING:
            raise ValueError("created continuation must be waiting")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationQuery:
    """Bounded indexed continuation query with an opaque cursor."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    statuses: tuple[ContinuationStatus, ...] = ()
    kinds: tuple[str, ...] = ()
    channel: str | None = None
    correlator: ContinuationCorrelator | None = None
    due_at_or_before: datetime | None = None
    open_at: datetime | None = None

    def __post_init__(self) -> None:
        for name, values in (("statuses", self.statuses), ("kinds", self.kinds)):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in self.kinds):
            raise ValueError("kinds must contain non-empty strings")
        if any(not isinstance(value, ContinuationStatus) for value in self.statuses):
            raise TypeError("statuses must contain ContinuationStatus values")
        _optional_nonempty("channel", self.channel)
        if self.due_at_or_before is not None:
            _utc("due_at_or_before", self.due_at_or_before)
        if self.open_at is not None:
            _utc("open_at", self.open_at)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationLeaseRequest:
    """Exact atomic claim request for one scheduled continuation occurrence."""

    fire_id: str
    continuation_id: str
    scope: StorageScope
    scheduled_for: datetime
    worker_id: str
    now: datetime
    lease_until: datetime

    def __post_init__(self) -> None:
        _nonempty("fire_id", self.fire_id)
        _nonempty("continuation_id", self.continuation_id)
        _nonempty("worker_id", self.worker_id)
        self.scope.require("run_id", "node_id")
        for name in ("scheduled_for", "now", "lease_until"):
            _utc(name, getattr(self, name))
        if self.scheduled_for > self.now:
            raise ValueError("scheduled_for must not be after now")
        if self.lease_until <= self.now:
            raise ValueError("lease_until must be after now")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationLeaseRecord:
    """Revisioned durable claim, retry state, or terminal delivery receipt."""

    fire_id: str
    continuation_id: str
    scope: StorageScope
    scheduled_for: datetime
    status: ContinuationLeaseStatus
    attempts: int
    revision: int
    updated_at: datetime
    worker_id: str | None = None
    lease_until: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    finished_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("fire_id", self.fire_id)
        _nonempty("continuation_id", self.continuation_id)
        self.scope.require("run_id", "node_id")
        _utc("scheduled_for", self.scheduled_for)
        _utc("updated_at", self.updated_at)
        if self.updated_at < self.scheduled_for:
            raise ValueError("updated_at must not precede scheduled_for")
        if not isinstance(self.status, ContinuationLeaseStatus):
            raise TypeError("status must be a ContinuationLeaseStatus")
        if isinstance(self.attempts, bool) or self.attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _optional_nonempty("worker_id", self.worker_id)
        _optional_nonempty("last_error", self.last_error)
        for name in ("lease_until", "next_attempt_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if self.status is ContinuationLeaseStatus.LEASED:
            if self.worker_id is None or self.lease_until is None:
                raise ValueError("leased records require worker_id and lease_until")
            if self.lease_until <= self.updated_at:
                raise ValueError("lease_until must be after updated_at")
            if self.next_attempt_at is not None or self.finished_at is not None:
                raise ValueError("leased records cannot carry retry or terminal timestamps")
        elif self.worker_id is not None or self.lease_until is not None:
            raise ValueError("non-leased records must not retain lease ownership")
        if self.status is ContinuationLeaseStatus.RETRY:
            if self.next_attempt_at is None or self.last_error is None:
                raise ValueError("retry records require next_attempt_at and last_error")
            if self.next_attempt_at <= self.updated_at:
                raise ValueError("next_attempt_at must be after updated_at")
            if self.finished_at is not None:
                raise ValueError("retry records must not have finished_at")
        elif self.next_attempt_at is not None:
            raise ValueError("only retry records may have next_attempt_at")
        if self.status in {
            ContinuationLeaseStatus.DELIVERED,
            ContinuationLeaseStatus.DEAD_LETTER,
        }:
            if self.finished_at is None:
                raise ValueError("terminal lease records require finished_at")
            if self.finished_at < self.updated_at:
                raise ValueError("finished_at must not precede updated_at")
        elif self.finished_at is not None:
            raise ValueError("nonterminal lease records must not have finished_at")
        if self.status is ContinuationLeaseStatus.DEAD_LETTER and self.last_error is None:
            raise ValueError("dead-letter records require last_error")
        if self.status is ContinuationLeaseStatus.DELIVERED and self.last_error is not None:
            raise ValueError("delivered records must not retain last_error")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationLeaseQuery:
    """Bounded lease/receipt query using canonical scope and opaque cursor."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    statuses: tuple[ContinuationLeaseStatus, ...] = ()
    continuation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.statuses, tuple):
            raise TypeError("statuses must be an immutable tuple")
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("statuses must not contain duplicates")
        if any(not isinstance(value, ContinuationLeaseStatus) for value in self.statuses):
            raise TypeError("statuses must contain ContinuationLeaseStatus values")
        _optional_nonempty("continuation_id", self.continuation_id)


class ContinuationRepository(Protocol):
    """Atomic repository for durable continuations and their secret indexes."""

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create a continuation and every initial lookup index.

        The provider mints a strong raw token, persists only its protected digest,
        and commits token and correlator lookup entries with the record.

        Examples:
            Create an approval wait:
                ```python
                created = await continuations.create(draft)
                ```

            Pass the token to a channel boundary:
                ```python
                await channel.publish_wait(token=created.token)
                ```

        Args:
            draft: Immutable continuation content and initial correlators.

        Returns:
            CreatedContinuation: Stored revision-one record and one-time raw token.

        Notes:
            Identity collision raises `StorageIntegrityError`; partial index writes are
            forbidden and raw tokens are absent from canonical persisted records.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        continuation_id: str,
    ) -> ContinuationRecord | None:
        """Read one current continuation by exact canonical identity.

        Lookup remains within the supplied scope and does not scan another owner,
        project, run, or node after a miss.

        Examples:
            Read a wait:
                ```python
                wait = await continuations.get(scope, "cont-1")
                ```

            Detect absence:
                ```python
                assert await continuations.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and run/node scope constraining access.
            continuation_id: Exact stable continuation identity.

        Returns:
            ContinuationRecord | None: Current record or `None` when absent.

        Notes:
            Deprecated App identity is not a lookup dimension.
        """
        ...

    async def resolve_token(self, token: str) -> ContinuationRecord | None:
        """Resolve an exact bearer token through the provider-owned secret index.

        The provider protects and compares the supplied token without exposing stored
        token material to callers.

        Examples:
            Resolve an inbound response:
                ```python
                wait = await continuations.resolve_token(token)
                ```

            Reject an unknown token:
                ```python
                assert await continuations.resolve_token("invalid") is None
                ```

        Args:
            token: Exact raw bearer token received from the trusted resume boundary.

        Returns:
            ContinuationRecord | None: Matching current record or `None`.

        Notes:
            Implementations use constant-time digest comparison where applicable and
            never fall back to run/node scans.
        """
        ...

    async def compare_and_set(
        self,
        record: ContinuationRecord,
        expected_revision: int,
    ) -> ContinuationRecord:
        """Atomically replace a continuation with its exact next revision.

        Lifecycle status, retry schedule, payload, and correlator indexes commit as
        one provider transaction.

        Examples:
            Close a resumed wait:
                ```python
                stored = await continuations.compare_and_set(resumed, wait.revision)
                ```

            Reschedule a poll:
                ```python
                stored = await continuations.compare_and_set(rescheduled, wait.revision)
                ```

        Args:
            record: Complete canonical next continuation revision.
            expected_revision: Current revision required for the update.

        Returns:
            ContinuationRecord: Newly committed authoritative revision.

        Notes:
            Stale expectations raise `StorageConflictError`; token identity is
            immutable across revisions.
        """
        ...

    async def bind_correlator(
        self,
        scope: StorageScope,
        continuation_id: str,
        correlator: ContinuationCorrelator,
        expected_revision: int,
    ) -> ContinuationRecord:
        """Atomically add one idempotent correlator and advance the revision.

        Record and reverse index update together so a successful bind is immediately
        queryable and a failed bind leaves neither side changed.

        Examples:
            Bind the sent message:
                ```python
                wait = await continuations.bind_correlator(scope, wait_id, corr, revision)
                ```

            Retry an already committed bind:
                ```python
                wait = await continuations.bind_correlator(scope, wait_id, corr, revision)
                ```

        Args:
            scope: Canonical owner and run/node scope constraining the update.
            continuation_id: Exact stable continuation identity.
            correlator: Transport-neutral correlation identity to add.
            expected_revision: Current revision required for a new binding.

        Returns:
            ContinuationRecord: Current record including the correlator.

        Notes:
            Repeating an existing binding is idempotent; competing new bindings use
            CAS and stale expectations raise `StorageConflictError`.
        """
        ...

    async def query(self, query: ContinuationQuery) -> Page[ContinuationRecord]:
        """Query a bounded stable cursor page using canonical continuation indexes.

        Scope and optional lifecycle, kind, channel, correlator, and due-time filters
        apply before provider-defined stable ordering.

        Examples:
            Read due waits:
                ```python
                page = await continuations.query(ContinuationQuery(scope=scope, due_at_or_before=now))
                ```

            Resolve a correlator page:
                ```python
                page = await continuations.query(ContinuationQuery(scope=scope, correlator=corr))
                ```

        Args:
            query: Exact scope, indexed filters, and opaque page request.

        Returns:
            Page[ContinuationRecord]: Matching records and continuation cursor.

        Notes:
            Unbounded `list_waits`, newest-wait guessing, and method probing are not
            part of the canonical protocol.
        """
        ...


class ContinuationLeaseRepository(Protocol):
    """Durable provider-owned claims and receipts for continuation timers."""

    async def claim(self, request: ContinuationLeaseRequest) -> ContinuationLeaseRecord | None:
        """Atomically create, retry, or reclaim one eligible timer fire.

        A delivered or dead-lettered receipt, active lease, or backoff-delayed retry
        returns no claim. Each successful claim advances attempts and revision.

        Examples:
            Claim a due fire:
                ```python
                lease = await leases.claim(request)
                ```

            Detect competing ownership:
                ```python
                if await leases.claim(request) is None:
                    return
                ```

        Args:
            request: Exact occurrence, worker, clock, and lease interval.

        Returns:
            ContinuationLeaseRecord | None: Worker-owned lease or `None` if ineligible.

        Notes:
            Claim and stale-lease reclamation are one provider transaction; raw
            continuation tokens are not stored in lease records.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        fire_id: str,
    ) -> ContinuationLeaseRecord | None:
        """Read one durable claim or terminal receipt by exact identity.

        The read is scope constrained and does not mutate worker ownership or retry
        eligibility.

        Examples:
            Inspect a receipt:
                ```python
                receipt = await leases.get(scope, "fire-1")
                ```

            Detect an unknown fire:
                ```python
                assert await leases.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and run/node scope constraining access.
            fire_id: Stable scheduled occurrence identity.

        Returns:
            ContinuationLeaseRecord | None: Current lease/receipt or `None`.

        Notes:
            Reads never renew leases implicitly.
        """
        ...

    async def compare_and_set(
        self,
        record: ContinuationLeaseRecord,
        expected_revision: int,
    ) -> ContinuationLeaseRecord:
        """Atomically renew, release, or terminalize a worker-owned lease.

        The provider validates the legal state transition, expected revision, and
        current worker ownership before committing the complete next record.

        Examples:
            Record delivery:
                ```python
                receipt = await leases.compare_and_set(delivered, lease.revision)
                ```

            Schedule retry backoff:
                ```python
                retry = await leases.compare_and_set(failed, lease.revision)
                ```

        Args:
            record: Complete canonical next lease or receipt revision.
            expected_revision: Current revision required for the transition.

        Returns:
            ContinuationLeaseRecord: Newly committed authoritative revision.

        Notes:
            Stale revision or ownership raises `StorageConflictError`; terminal
            receipts are immutable and remain as durable deduplication evidence.
        """
        ...

    async def query(self, query: ContinuationLeaseQuery) -> Page[ContinuationLeaseRecord]:
        """Query a bounded stable cursor page of leases and receipts.

        Canonical scope and optional status/continuation filters apply before stable
        provider ordering.

        Examples:
            Inspect dead letters:
                ```python
                page = await leases.query(ContinuationLeaseQuery(scope=scope, statuses=(status,)))
                ```

            Continue receipt inspection:
                ```python
                page = await leases.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, filters, and opaque page request.

        Returns:
            Page[ContinuationLeaseRecord]: Matching records and continuation cursor.

        Notes:
            This diagnostic query is bounded; workers claim exact due occurrences
            through `claim` rather than scanning receipts.
        """
        ...
