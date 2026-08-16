"""Canonical run, result, and session control-store contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .pagination import Page, PageRequest
from .records import (
    FrozenJson,
    _freeze_json,
    _freeze_mapping,
    _nonempty,
    _optional_nonempty,
    _optional_text,
    _utc,
)
from .scope import StorageScope


class RunStatus(StrEnum):
    """Canonical durable lifecycle status for one runtime run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    WAITING = "waiting"
    CANCELED = "canceled"
    CANCELLATION_REQUESTED = "cancellation_requested"


class SessionKind(StrEnum):
    """Canonical product session kind stored independently from API schemas."""

    CHAT = "chat"
    PLAYGROUND = "playground"
    NOTEBOOK = "notebook"
    PIPELINE = "pipeline"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecord:
    """Current revisioned canonical runtime run and transactional counters."""

    run_id: str
    graph_id: str
    kind: str
    status: RunStatus
    scope: StorageScope
    revision: int
    started_at: datetime
    finished_at: datetime | None = None
    tags: tuple[str, ...] = ()
    error: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    artifact_count: int = 0
    first_artifact_at: datetime | None = None
    last_artifact_at: datetime | None = None
    result_available: bool = False
    result_updated_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("run_id", self.run_id)
        _nonempty("graph_id", self.graph_id)
        _nonempty("kind", self.kind)
        self.scope.require("run_id", "graph_id")
        if self.scope.run_id != self.run_id or self.scope.graph_id != self.graph_id:
            raise ValueError("run_id and graph_id must match canonical scope")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("started_at", self.started_at)
        if self.finished_at is not None:
            _utc("finished_at", self.finished_at)
            if self.finished_at < self.started_at:
                raise ValueError("finished_at must not precede started_at")
        if self.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED}:
            if self.finished_at is None:
                raise ValueError("terminal runs require finished_at")
        elif self.finished_at is not None:
            raise ValueError("nonterminal runs must not have finished_at")
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be an immutable tuple")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("tags must contain non-empty strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must not contain duplicates")
        _optional_nonempty("error", self.error)
        if isinstance(self.artifact_count, bool) or self.artifact_count < 0:
            raise ValueError("artifact_count must be a non-negative integer")
        for name in ("first_artifact_at", "last_artifact_at", "result_updated_at"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if self.artifact_count == 0 and (
            self.first_artifact_at is not None or self.last_artifact_at is not None
        ):
            raise ValueError("zero artifact_count must not carry artifact timestamps")
        if self.artifact_count > 0 and (
            self.first_artifact_at is None or self.last_artifact_at is None
        ):
            raise ValueError("positive artifact_count requires artifact timestamps")
        if (
            self.first_artifact_at is not None
            and self.last_artifact_at is not None
            and self.first_artifact_at > self.last_artifact_at
        ):
            raise ValueError("first_artifact_at must not follow last_artifact_at")
        if self.result_available != (self.result_updated_at is not None):
            raise ValueError("result_available must match result_updated_at presence")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResultRecord:
    """Revisioned durable final output for one canonical runtime run."""

    run_id: str
    graph_id: str
    scope: StorageScope
    status: RunStatus
    outputs: FrozenJson
    revision: int
    created_at: datetime
    updated_at: datetime
    source: str
    snapshot_revision: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("run_id", self.run_id)
        _nonempty("graph_id", self.graph_id)
        _nonempty("source", self.source)
        self.scope.require("run_id", "graph_id")
        if self.scope.run_id != self.run_id or self.scope.graph_id != self.graph_id:
            raise ValueError("run_id and graph_id must match canonical scope")
        if self.status is not RunStatus.SUCCEEDED:
            raise ValueError("durable run results require succeeded status")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("created_at", self.created_at)
        _utc("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.snapshot_revision is not None and (
            isinstance(self.snapshot_revision, bool) or self.snapshot_revision < 1
        ):
            raise ValueError("snapshot_revision must be positive when supplied")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(self, "outputs", _freeze_json(self.outputs, path="outputs"))


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRecord:
    """Current revisioned canonical user/product session and artifact counters."""

    session_id: str
    kind: SessionKind
    scope: StorageScope
    revision: int
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    source: str = "runtime"
    external_reference: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    artifact_count: int = 0
    last_artifact_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("session_id", self.session_id)
        self.scope.require("session_id")
        if self.scope.session_id != self.session_id:
            raise ValueError("session_id must match canonical scope")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("created_at", self.created_at)
        _utc("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        _optional_text("title", self.title)
        _nonempty("source", self.source)
        _optional_text("external_reference", self.external_reference)
        if isinstance(self.artifact_count, bool) or self.artifact_count < 0:
            raise ValueError("artifact_count must be a non-negative integer")
        if self.last_artifact_at is not None:
            _utc("last_artifact_at", self.last_artifact_at)
        if (self.artifact_count == 0) != (self.last_artifact_at is None):
            raise ValueError("artifact_count and last_artifact_at must agree")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunQuery:
    """Bounded canonical run query using scope, status, kind, and opaque cursor."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    statuses: tuple[RunStatus, ...] = ()
    kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, values in (("statuses", self.statuses), ("kinds", self.kinds)):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in self.kinds):
            raise ValueError("kinds must contain non-empty strings")


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionQuery:
    """Bounded canonical session query using scope, kind, and opaque cursor."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    kinds: tuple[SessionKind, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kinds, tuple):
            raise TypeError("kinds must be an immutable tuple")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must not contain duplicates")


class RunRepository(Protocol):
    """Transactional repository for canonical runs and their artifact counters."""

    async def create(self, record: RunRecord) -> RunRecord:
        """Idempotently create one initial canonical run record.

        The initial record must have revision one. Repeating identical identity and
        content succeeds; conflicting reuse fails directly.

        Examples:
            Create a run:
                ```python
                stored = await runs.create(record)
                ```

            Retry creation:
                ```python
                assert await runs.create(record) == stored
                ```

        Args:
            record: Complete canonical run at revision one.

        Returns:
            RunRecord: Authoritative stored run record.

        Notes:
            Identity conflict raises `StorageIntegrityError`; there is no overwrite.
        """
        ...

    async def get(self, scope: StorageScope, run_id: str) -> RunRecord | None:
        """Read one current run record within canonical scope.

        The provider performs an indexed exact lookup and does not search another
        tenant or project for the same run identifier.

        Examples:
            Read a run:
                ```python
                run = await runs.get(scope, "run-1")
                ```

            Detect absence:
                ```python
                assert await runs.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining access.
            run_id: Exact stable run identifier.

        Returns:
            RunRecord | None: Current run record or `None` when absent.

        Notes:
            Deprecated App identity is not a lookup dimension.
        """
        ...

    async def compare_and_set(
        self,
        record: RunRecord,
        expected_revision: int,
    ) -> RunRecord:
        """Atomically replace a run with its exact next revision.

        The record revision must equal expected revision plus one. Status, terminal
        timestamp, result availability, and metadata commit together.

        Examples:
            Advance run status:
                ```python
                updated = await runs.compare_and_set(next_record, current.revision)
                ```

            Complete a run:
                ```python
                stored = await runs.compare_and_set(completed, running.revision)
                ```

        Args:
            record: Complete canonical next run revision.
            expected_revision: Current revision required for the update.

        Returns:
            RunRecord: Newly committed authoritative run revision.

        Notes:
            Stale expectations raise `StorageConflictError`; read-modify-write without
            CAS is not part of the protocol.
        """
        ...

    async def query(self, query: RunQuery) -> Page[RunRecord]:
        """Query a bounded stable cursor page of canonical runs.

        Canonical scope and optional status/kind filters apply before ordering by the
        provider's documented recent-run cursor.

        Examples:
            List recent runs:
                ```python
                page = await runs.query(RunQuery(scope=scope))
                ```

            Continue failed runs:
                ```python
                page = await runs.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, filters, and opaque page request.

        Returns:
            Page[RunRecord]: Matching run records and continuation cursor.

        Notes:
            Offset pagination and unbounded production lists are absent.
        """
        ...

    async def record_artifact(
        self,
        scope: StorageScope,
        run_id: str,
        occurrence_id: str,
        occurred_at: datetime,
    ) -> RunRecord:
        """Atomically count one idempotent artifact occurrence for a run.

        The occurrence identity prevents double counting across retries. Count and
        first/last timestamps update in the same transaction.

        Examples:
            Count produced content:
                ```python
                run = await runs.record_artifact(scope, run_id, occurrence_id, now)
                ```

            Retry the same occurrence:
                ```python
                assert await runs.record_artifact(scope, run_id, occurrence_id, now) == run
                ```

        Args:
            scope: Canonical scope owning the run and occurrence.
            run_id: Exact stable run identifier.
            occurrence_id: Stable artifact occurrence idempotency identity.
            occurred_at: Timezone-aware UTC occurrence time.

        Returns:
            RunRecord: Current run after the transactional counter update.

        Notes:
            Missing runs raise `StorageNotFoundError`; duplicate occurrences do not
            increment the count again.
        """
        ...


class RunResultRepository(Protocol):
    """Revisioned durable repository for canonical successful run outputs."""

    async def compare_and_set(
        self,
        record: RunResultRecord,
        expected_revision: int,
    ) -> RunResultRecord:
        """Atomically create or advance one durable run result.

        Revision zero creates the first result. Revisions support controlled final
        output refinement without separate result database behavior.

        Examples:
            Create final output:
                ```python
                stored = await results.compare_and_set(record, 0)
                ```

            Advance output metadata:
                ```python
                stored = await results.compare_and_set(next_record, current.revision)
                ```

        Args:
            record: Complete canonical next result revision.
            expected_revision: Current revision required, or zero for creation.

        Returns:
            RunResultRecord: Newly committed durable result revision.

        Notes:
            The provider coordinates run `result_available` updates transactionally.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> RunResultRecord | None:
        """Read the current durable result for one run.

        The exact lookup is canonical-scope constrained and does not infer results
        from run preview fields.

        Examples:
            Read final output:
                ```python
                result = await results.get(scope, "run-1")
                ```

            Detect no output:
                ```python
                assert await results.get(scope, "running") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining access.
            run_id: Exact stable run identifier.

        Returns:
            RunResultRecord | None: Current durable result or `None` when absent.

        Notes:
            Result storage shares provider transaction ownership with run metadata.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        run_id: str,
        expected_revision: int,
    ) -> bool:
        """Atomically delete one exact result and clear its run marker.

        Deletion is canonical-scope constrained and revision guarded so recovery
        tooling cannot erase a newer successful output accidentally.

        Examples:
            Delete a corrupt output:
                ```python
                deleted = await results.delete(scope, run_id, result.revision)
                ```

            Detect an absent output:
                ```python
                assert not await results.delete(scope, "missing", 1)
                ```

        Args:
            scope: Canonical owner/execution scope constraining deletion.
            run_id: Exact stable run identifier.
            expected_revision: Exact current result revision required.

        Returns:
            bool: `True` when deleted, or `False` when absent or unauthorized.

        Notes:
            The provider advances the owning run revision and clears its result
            availability fields in the same transaction. Stale expectations raise
            `StorageConflictError`.
        """
        ...


class SessionRepository(Protocol):
    """Transactional repository for canonical sessions and artifact counters."""

    async def create(self, record: SessionRecord) -> SessionRecord:
        """Idempotently create one initial canonical session.

        Exact identity reuse with different scope or immutable creation fields fails
        instead of silently updating the existing session.

        Examples:
            Create a chat session:
                ```python
                stored = await sessions.create(record)
                ```

            Retry creation:
                ```python
                assert await sessions.create(record) == stored
                ```

        Args:
            record: Complete canonical session at revision one.

        Returns:
            SessionRecord: Authoritative stored session.

        Notes:
            Identity collision raises `StorageIntegrityError`.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        session_id: str,
    ) -> SessionRecord | None:
        """Read one current session within canonical scope.

        The provider performs an indexed exact lookup and never broadens scope after
        a miss.

        Examples:
            Read a session:
                ```python
                session = await sessions.get(scope, "session-1")
                ```

            Detect absence:
                ```python
                assert await sessions.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/session scope constraining access.
            session_id: Exact stable session identifier.

        Returns:
            SessionRecord | None: Current session or `None` when absent.

        Notes:
            Deprecated App identity is not a lookup dimension.
        """
        ...

    async def compare_and_set(
        self,
        record: SessionRecord,
        expected_revision: int,
    ) -> SessionRecord:
        """Atomically replace a session with its exact next revision.

        Mutable title, external reference, metadata, and updated time commit together.
        Artifact counters use their dedicated idempotent transaction method.

        Examples:
            Rename a session:
                ```python
                stored = await sessions.compare_and_set(renamed, current.revision)
                ```

            Update metadata:
                ```python
                stored = await sessions.compare_and_set(updated, current.revision)
                ```

        Args:
            record: Complete canonical next session revision.
            expected_revision: Current revision required for the update.

        Returns:
            SessionRecord: Newly committed authoritative session revision.

        Notes:
            Stale expectations raise `StorageConflictError`.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        session_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete one exact current session using revision CAS.

        The operation preserves unrelated runs and artifacts while provider-owned
        session artifact occurrence receipts are removed transactionally.

        Examples:
            Delete a session:
                ```python
                deleted = await sessions.delete(scope, session_id, session.revision)
                ```

            Detect an absent session:
                ```python
                assert not await sessions.delete(scope, "missing", 1)
                ```

        Args:
            scope: Canonical owner/session scope constraining deletion.
            session_id: Exact stable session identifier.
            expected_revision: Exact current session revision required.

        Returns:
            bool: `True` when deleted, or `False` when absent or unauthorized.

        Notes:
            Stale expectations raise `StorageConflictError`; scope is never
            broadened after an absent or unauthorized lookup.
        """
        ...

    async def query(self, query: SessionQuery) -> Page[SessionRecord]:
        """Query a bounded stable cursor page of canonical sessions.

        Canonical owner scope and optional kind filters apply before recent-session
        cursor pagination.

        Examples:
            List user sessions:
                ```python
                page = await sessions.query(SessionQuery(scope=user_scope))
                ```

            Continue chat sessions:
                ```python
                page = await sessions.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact owner scope, kind filters, and opaque page request.

        Returns:
            Page[SessionRecord]: Matching sessions and continuation cursor.

        Notes:
            Offset pagination and unbounded production lists are absent.
        """
        ...

    async def record_artifact(
        self,
        scope: StorageScope,
        session_id: str,
        occurrence_id: str,
        occurred_at: datetime,
    ) -> SessionRecord:
        """Atomically count one idempotent artifact occurrence for a session.

        The occurrence identity prevents retry double counting. Count, last-artifact
        time, updated time, and revision commit in one transaction.

        Examples:
            Count an attachment:
                ```python
                session = await sessions.record_artifact(scope, session_id, occurrence_id, now)
                ```

            Retry the occurrence:
                ```python
                assert await sessions.record_artifact(scope, session_id, occurrence_id, now) == session
                ```

        Args:
            scope: Canonical scope owning the session and occurrence.
            session_id: Exact stable session identifier.
            occurrence_id: Stable artifact occurrence idempotency identity.
            occurred_at: Timezone-aware UTC occurrence time.

        Returns:
            SessionRecord: Current session after transactional counter update.

        Notes:
            Missing sessions raise `StorageNotFoundError`; duplicate occurrences do
            not increment the count again.
        """
        ...
