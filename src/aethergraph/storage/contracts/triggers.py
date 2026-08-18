"""Canonical trigger definition and atomic occurrence-claim contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .pagination import Page, PageRequest
from .records import FrozenJson, _freeze_mapping, _nonempty, _optional_nonempty, _utc
from .scope import StorageScope


class TriggerKind(StrEnum):
    """Canonical trigger scheduling mechanism."""

    CRON = "cron"
    INTERVAL = "interval"
    ONE_SHOT = "one_shot"
    EVENT = "event"


class TriggerClaimStatus(StrEnum):
    """Canonical lifecycle state for one trigger occurrence."""

    LEASED = "leased"
    RETRY = "retry"
    DELIVERED = "delivered"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerRecord:
    """Revisioned canonical trigger definition with provider-neutral scheduling."""

    trigger_id: str
    graph_id: str
    scope: StorageScope
    kind: TriggerKind
    revision: int
    created_at: datetime
    updated_at: datetime
    default_inputs: Mapping[str, FrozenJson] = field(default_factory=dict)
    name: str | None = None
    origin: str = "schedule"
    cron_expression: str | None = None
    interval_seconds: int | None = None
    run_at: datetime | None = None
    event_key: str | None = None
    timezone: str | None = None
    max_overlap_runs: int | None = None
    catch_up_missed: bool = False
    active: bool = True
    last_fired_at: datetime | None = None
    next_fire_at: datetime | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("trigger_id", self.trigger_id)
        _nonempty("graph_id", self.graph_id)
        self.scope.require("graph_id")
        if self.scope.graph_id != self.graph_id:
            raise ValueError("graph_id must match canonical scope")
        if self.scope.run_id is not None or self.scope.node_id is not None:
            raise ValueError("trigger scope must not contain run_id or node_id")
        if not isinstance(self.kind, TriggerKind):
            raise TypeError("kind must be a TriggerKind")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("created_at", self.created_at)
        _utc("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        _optional_nonempty("name", self.name)
        _nonempty("origin", self.origin)
        _optional_nonempty("timezone", self.timezone)
        if not isinstance(self.catch_up_missed, bool) or not isinstance(self.active, bool):
            raise TypeError("catch_up_missed and active must be booleans")
        if self.max_overlap_runs is not None and (
            isinstance(self.max_overlap_runs, bool) or self.max_overlap_runs < 0
        ):
            raise ValueError("max_overlap_runs must be non-negative when supplied")
        for name in ("run_at", "last_fired_at", "next_fire_at"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if self.last_fired_at is not None and self.last_fired_at < self.created_at:
            raise ValueError("last_fired_at must not precede created_at")
        self._validate_schedule()
        object.__setattr__(
            self,
            "default_inputs",
            _freeze_mapping(self.default_inputs, path="default_inputs"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )

    def _validate_schedule(self) -> None:
        configured = {
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "run_at": self.run_at,
            "event_key": self.event_key,
        }
        expected = {
            TriggerKind.CRON: "cron_expression",
            TriggerKind.INTERVAL: "interval_seconds",
            TriggerKind.ONE_SHOT: "run_at",
            TriggerKind.EVENT: "event_key",
        }[self.kind]
        for name, value in configured.items():
            if name == expected:
                if value is None:
                    raise ValueError(f"{self.kind.value} triggers require {name}")
            elif value is not None:
                raise ValueError(f"{self.kind.value} triggers must not define {name}")
        if self.cron_expression is not None:
            _nonempty("cron_expression", self.cron_expression)
        if self.interval_seconds is not None and (
            isinstance(self.interval_seconds, bool) or self.interval_seconds < 1
        ):
            raise ValueError("interval_seconds must be positive")
        if self.event_key is not None:
            _nonempty("event_key", self.event_key)
        if self.kind is TriggerKind.EVENT:
            if self.next_fire_at is not None:
                raise ValueError("event triggers must not define next_fire_at")
        elif self.active and self.next_fire_at is None:
            raise ValueError("active scheduled triggers require next_fire_at")
        if self.kind is TriggerKind.ONE_SHOT and self.active and self.next_fire_at != self.run_at:
            raise ValueError("active one-shot next_fire_at must equal run_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerQuery:
    """Bounded indexed trigger-definition query with opaque pagination."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    kinds: tuple[TriggerKind, ...] = ()
    active: bool | None = None
    event_key: str | None = None

    def __post_init__(self) -> None:
        if not self.scope.as_filter():
            raise ValueError("trigger queries require an explicit canonical scope")
        if self.scope.run_id is not None or self.scope.node_id is not None:
            raise ValueError("trigger query scope must not contain run_id or node_id")
        if not isinstance(self.kinds, tuple):
            raise TypeError("kinds must be an immutable tuple")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must not contain duplicates")
        if any(not isinstance(value, TriggerKind) for value in self.kinds):
            raise TypeError("kinds must contain TriggerKind values")
        if self.active is not None and not isinstance(self.active, bool):
            raise TypeError("active must be a boolean when supplied")
        _optional_nonempty("event_key", self.event_key)
        if self.event_key is not None and self.kinds not in ((), (TriggerKind.EVENT,)):
            raise ValueError("event_key may only be combined with event trigger kind")


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerClaimRequest:
    """Exact bounded request for atomically claiming due trigger occurrences."""

    now: datetime
    worker_id: str
    lease_until: datetime
    limit: int
    scope: StorageScope | None = None
    skip_missed_before: datetime | None = None

    def __post_init__(self) -> None:
        _utc("now", self.now)
        _nonempty("worker_id", self.worker_id)
        _utc("lease_until", self.lease_until)
        if self.lease_until <= self.now:
            raise ValueError("lease_until must be after now")
        if isinstance(self.limit, bool) or not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if self.skip_missed_before is not None:
            _utc("skip_missed_before", self.skip_missed_before)
            if self.skip_missed_before > self.now:
                raise ValueError("skip_missed_before must not be after now")
        if self.scope is not None and not self.scope.as_filter():
            raise ValueError("claim scope must be populated when supplied")


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerClaimRecord:
    """Revisioned durable trigger lease, retry, delivery, or skip receipt."""

    fire_id: str
    trigger_id: str
    scope: StorageScope
    scheduled_for: datetime
    status: TriggerClaimStatus
    attempts: int
    revision: int
    updated_at: datetime
    worker_id: str | None = None
    lease_until: datetime | None = None
    retry_at: datetime | None = None
    run_id: str | None = None
    last_error: str | None = None
    skip_reason: str | None = None
    finished_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("fire_id", self.fire_id)
        _nonempty("trigger_id", self.trigger_id)
        self.scope.require("graph_id")
        _utc("scheduled_for", self.scheduled_for)
        _utc("updated_at", self.updated_at)
        if self.updated_at < self.scheduled_for:
            raise ValueError("updated_at must not precede scheduled_for")
        if not isinstance(self.status, TriggerClaimStatus):
            raise TypeError("status must be a TriggerClaimStatus")
        if isinstance(self.attempts, bool) or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        for name in ("worker_id", "run_id", "last_error", "skip_reason"):
            _optional_nonempty(name, getattr(self, name))
        for name in ("lease_until", "retry_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        self._validate_status()
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")

    def _validate_status(self) -> None:
        if self.status is TriggerClaimStatus.LEASED:
            if self.attempts < 1 or self.worker_id is None or self.lease_until is None:
                raise ValueError("leased claims require attempts, worker_id, and lease_until")
            if self.lease_until <= self.updated_at:
                raise ValueError("lease_until must be after updated_at")
            if any(
                value is not None
                for value in (
                    self.retry_at,
                    self.run_id,
                    self.last_error,
                    self.skip_reason,
                    self.finished_at,
                )
            ):
                raise ValueError("leased claims must not carry retry or terminal fields")
            return
        if self.worker_id is not None or self.lease_until is not None:
            raise ValueError("non-leased claims must not retain lease ownership")
        if self.status is TriggerClaimStatus.RETRY:
            if self.attempts < 1 or self.retry_at is None or self.last_error is None:
                raise ValueError("retry claims require attempts, retry_at, and last_error")
            if self.retry_at <= self.updated_at:
                raise ValueError("retry_at must be after updated_at")
            if (
                self.run_id is not None
                or self.skip_reason is not None
                or self.finished_at is not None
            ):
                raise ValueError("retry claims must not carry delivery or skip fields")
            return
        if self.finished_at is None:
            raise ValueError("terminal trigger claims require finished_at")
        if self.finished_at < self.updated_at:
            raise ValueError("finished_at must not precede updated_at")
        if self.retry_at is not None or self.last_error is not None:
            raise ValueError("terminal trigger claims must not retain retry fields")
        if self.status is TriggerClaimStatus.DELIVERED:
            if self.attempts < 1 or self.run_id is None or self.skip_reason is not None:
                raise ValueError("delivered claims require attempts and run_id only")
        elif self.run_id is not None or self.skip_reason is None:
            raise ValueError("skipped claims require skip_reason and no run_id")


@dataclass(frozen=True, slots=True)
class ClaimedTrigger:
    """Trigger definition and worker-owned occurrence returned from one claim."""

    trigger: TriggerRecord
    claim: TriggerClaimRecord
    reclaimed: bool = False

    def __post_init__(self) -> None:
        if self.trigger.trigger_id != self.claim.trigger_id:
            raise ValueError("trigger and claim identities must match")
        if self.trigger.scope != self.claim.scope:
            raise ValueError("trigger and claim scopes must match")
        if self.claim.status is not TriggerClaimStatus.LEASED:
            raise ValueError("claimed trigger must contain a leased claim")
        if not isinstance(self.reclaimed, bool):
            raise TypeError("reclaimed must be a boolean")


class TriggerRepository(Protocol):
    """Transactional repository for trigger definitions and occurrence claims."""

    async def create(self, record: TriggerRecord) -> TriggerRecord:
        """Idempotently create one revision-one canonical trigger.

        Exact identity and content retries succeed; conflicting identity reuse fails
        without updating the existing definition.

        Examples:
            Create a scheduled trigger:
                ```python
                stored = await triggers.create(record)
                ```

            Retry creation safely:
                ```python
                assert await triggers.create(record) == stored
                ```

        Args:
            record: Complete canonical trigger at revision one.

        Returns:
            TriggerRecord: Authoritative stored trigger definition.

        Notes:
            Identity conflict raises `StorageIntegrityError`; there is no upsert.
        """
        ...

    async def get(self, scope: StorageScope, trigger_id: str) -> TriggerRecord | None:
        """Read one current trigger by exact canonical identity.

        The lookup remains within the supplied owner/graph scope and never retries
        with deprecated App or client identities.

        Examples:
            Read a trigger:
                ```python
                trigger = await triggers.get(scope, "trigger-1")
                ```

            Detect absence:
                ```python
                assert await triggers.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and graph scope constraining access.
            trigger_id: Exact stable trigger identity.

        Returns:
            TriggerRecord | None: Current definition or `None` when absent.

        Notes:
            Scope mismatch behaves as absence and does not disclose another owner.
        """
        ...

    async def compare_and_set(
        self,
        record: TriggerRecord,
        expected_revision: int,
    ) -> TriggerRecord:
        """Atomically replace a trigger with its exact next revision.

        Schedule, active state, next-fire time, inputs, and metadata commit together.

        Examples:
            Pause a trigger:
                ```python
                stored = await triggers.compare_and_set(paused, current.revision)
                ```

            Advance its schedule:
                ```python
                stored = await triggers.compare_and_set(advanced, current.revision)
                ```

        Args:
            record: Complete canonical next trigger revision.
            expected_revision: Current revision required for the update.

        Returns:
            TriggerRecord: Newly committed authoritative revision.

        Notes:
            Stale expectations raise `StorageConflictError`; read-modify-write without
            CAS is absent from the protocol.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        trigger_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete an exact trigger only at the expected current revision.

        The provider applies trigger and associated nonterminal-claim cleanup in one
        transaction while preserving any required terminal audit evidence.

        Examples:
            Delete an owned trigger:
                ```python
                removed = await triggers.delete(scope, trigger_id, revision)
                ```

            Detect absence:
                ```python
                assert not await triggers.delete(scope, "missing", 1)
                ```

        Args:
            scope: Canonical owner and graph scope constraining deletion.
            trigger_id: Exact stable trigger identity.
            expected_revision: Current definition revision required for deletion.

        Returns:
            bool: Whether the exact scoped trigger was deleted.

        Notes:
            Scope never broadens after a miss; stale revision raises
            `StorageConflictError`.
        """
        ...

    async def query(self, query: TriggerQuery) -> Page[TriggerRecord]:
        """Query a bounded stable cursor page using promoted trigger indexes.

        Owner/graph scope and optional kind, activity, and event-key filters apply
        before provider-defined stable ordering.

        Examples:
            List active owner triggers:
                ```python
                page = await triggers.query(TriggerQuery(scope=scope, active=True))
                ```

            Resolve event triggers:
                ```python
                page = await triggers.query(TriggerQuery(scope=scope, event_key="invoice.paid"))
                ```

        Args:
            query: Exact scope, indexed filters, and opaque page request.

        Returns:
            Page[TriggerRecord]: Matching definitions and continuation cursor.

        Notes:
            Unbounded `list_all` and legacy client/user alias matching are absent.
        """
        ...

    async def claim_due(self, request: TriggerClaimRequest) -> tuple[ClaimedTrigger, ...]:
        """Atomically claim a bounded batch and advance trigger schedules.

        Eligible retries and expired leases are claimed before newly due definitions.
        New claims, missed-run receipts, and next-fire revisions commit together.

        Examples:
            Claim globally due work:
                ```python
                claims = await triggers.claim_due(request)
                ```

            Claim one owner partition:
                ```python
                claims = await triggers.claim_due(replace(request, scope=owner_scope))
                ```

        Args:
            request: Worker, clock, bound, optional scope, and catch-up boundary.

        Returns:
            tuple[ClaimedTrigger, ...]: Worker-owned claims up to the requested limit.

        Notes:
            A `None` scope is an explicit trusted runtime-wide scan, not a fallback
            after a scoped miss. `reclaimed` is authored atomically and distinguishes
            stale-lease recovery from an ordinary retry.
        """
        ...

    async def get_claim(
        self,
        scope: StorageScope,
        fire_id: str,
    ) -> TriggerClaimRecord | None:
        """Read one durable trigger occurrence claim or receipt.

        Exact scope and fire identity constrain the read without changing ownership
        or retry eligibility.

        Examples:
            Inspect delivery:
                ```python
                receipt = await triggers.get_claim(scope, "fire-1")
                ```

            Detect an unknown occurrence:
                ```python
                assert await triggers.get_claim(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and graph scope constraining access.
            fire_id: Stable trigger occurrence identity.

        Returns:
            TriggerClaimRecord | None: Current claim/receipt or `None`.

        Notes:
            Reads never renew leases or claim retries implicitly.
        """
        ...

    async def compare_and_set_claim(
        self,
        record: TriggerClaimRecord,
        expected_revision: int,
    ) -> TriggerClaimRecord:
        """Atomically renew, retry, deliver, or skip a worker-owned claim.

        The provider validates revision and worker ownership. Delivery also updates
        the trigger's last-fired metadata in the same transaction.

        Examples:
            Commit a delivered run:
                ```python
                receipt = await triggers.compare_and_set_claim(delivered, claim.revision)
                ```

            Release into retry backoff:
                ```python
                retry = await triggers.compare_and_set_claim(failed, claim.revision)
                ```

        Args:
            record: Complete canonical next claim or receipt revision.
            expected_revision: Current claim revision required for transition.

        Returns:
            TriggerClaimRecord: Newly committed authoritative revision.

        Notes:
            Stale ownership/revision raises `StorageConflictError`; delivered and
            skipped receipts are immutable deduplication evidence.
        """
        ...
