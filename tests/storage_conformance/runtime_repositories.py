"""Filesystem-free canonical repositories for external runtime qualification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields, replace
import hashlib
from typing import TypeVar

import aethergraph.storage.contracts as storage

_RecordT = TypeVar("_RecordT")


class InMemoryDeliveryCursorAllocator:
    def __init__(self) -> None:
        self._cursor = 0

    def next(self) -> int:
        self._cursor += 1
        return self._cursor


def _scope_matches(candidate: storage.StorageScope, expected: storage.StorageScope) -> bool:
    return all(getattr(candidate, name) == value for name, value in expected.as_filter().items())


def _promote(source: object, record_type: type[_RecordT], **extra: object) -> _RecordT:
    values = {
        item.name: getattr(source, item.name)
        for item in fields(record_type)
        if hasattr(source, item.name)
    }
    values.update(extra)
    return record_type(**values)


def _page(items: Iterable[_RecordT], request: storage.PageRequest) -> storage.Page[_RecordT]:
    ordered = tuple(items)
    offset = 0
    if request.cursor is not None:
        prefix, separator, raw_offset = request.cursor.partition(":")
        if prefix != "external" or separator != ":" or not raw_offset.isdigit():
            raise storage.StorageIntegrityError("external test cursor is malformed")
        offset = int(raw_offset)
    selected = ordered[offset : offset + request.limit]
    next_offset = offset + len(selected)
    next_cursor = f"external:{next_offset}" if next_offset < len(ordered) else None
    return storage.Page(items=selected, next_cursor=next_cursor)


def _cas_revision(current: object | None, expected_revision: int) -> int:
    actual = getattr(current, "revision", 0) if current is not None else 0
    if actual != expected_revision:
        raise storage.StorageConflictError(
            f"external test revision conflict: expected {expected_revision}, found {actual}"
        )
    return expected_revision + 1


class InMemoryKeyValueStore:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._records: dict[tuple[storage.StorageScope, str, str], storage.KeyValueRecord] = {}

    async def get(self, scope, namespace, key):
        record = self._records.get((scope, namespace, key))
        if (
            record is not None
            and record.expires_at is not None
            and record.expires_at <= self._clock.now()
        ):
            return None
        return record

    async def compare_and_set(
        self,
        scope,
        namespace,
        key,
        expected_revision,
        value,
        expires_at=None,
    ):
        identity = (scope, namespace, key)
        current = self._records.get(identity)
        revision = _cas_revision(current, expected_revision)
        record = storage.KeyValueRecord(
            namespace=namespace,
            key=key,
            value=value,
            revision=revision,
            scope=scope,
            updated_at=self._clock.now(),
            expires_at=expires_at,
        )
        self._records[identity] = record
        return record

    async def delete(self, scope, namespace, key, expected_revision):
        identity = (scope, namespace, key)
        current = self._records.get(identity)
        if current is None:
            if expected_revision:
                raise storage.StorageConflictError("external test KV record is absent")
            return False
        _cas_revision(current, expected_revision)
        del self._records[identity]
        return True

    async def scan(self, query):
        rows = (
            record
            for record in self._records.values()
            if record.scope == query.scope
            and record.namespace == query.namespace
            and (query.key_prefix is None or record.key.startswith(query.key_prefix))
            and (record.expires_at is None or record.expires_at > self._clock.now())
        )
        return _page(sorted(rows, key=lambda item: item.key), query.page)

    async def purge_expired(self, scope, namespace, limit):
        identities = [
            identity
            for identity, record in self._records.items()
            if record.scope == scope
            and record.namespace == namespace
            and record.expires_at is not None
            and record.expires_at <= self._clock.now()
        ][:limit]
        for identity in identities:
            del self._records[identity]
        return len(identities)


class InMemoryDocumentStore:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._records: dict[tuple[storage.StorageScope, str, str], storage.DocumentRecord] = {}

    async def get(self, scope, namespace, document_id):
        return self._records.get((scope, namespace, document_id))

    async def compare_and_set(
        self,
        scope,
        namespace,
        document_id,
        expected_revision,
        document,
        schema_version,
    ):
        identity = (scope, namespace, document_id)
        current = self._records.get(identity)
        record = storage.DocumentRecord(
            namespace=namespace,
            document_id=document_id,
            document=document,
            revision=_cas_revision(current, expected_revision),
            scope=scope,
            updated_at=self._clock.now(),
            schema_version=schema_version,
        )
        self._records[identity] = record
        return record

    async def delete(self, scope, namespace, document_id, expected_revision):
        identity = (scope, namespace, document_id)
        current = self._records.get(identity)
        if current is None:
            if expected_revision:
                raise storage.StorageConflictError("external test document is absent")
            return False
        _cas_revision(current, expected_revision)
        del self._records[identity]
        return True

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if record.scope == query.scope
            and record.namespace == query.namespace
            and (query.id_prefix is None or record.document_id.startswith(query.id_prefix))
            and all(record.document.get(key) == value for key, value in query.metadata.items())
        )
        return _page(sorted(rows, key=lambda item: item.document_id), query.page)


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.RunRecord] = {}

    async def create(self, record):
        current = self._records.get(record.run_id)
        if current is not None and current != record:
            raise storage.StorageIntegrityError("external test run identity conflicts")
        self._records[record.run_id] = record
        return record

    async def get(self, scope, run_id):
        record = self._records.get(run_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.run_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external test run next revision is invalid")
        self._records[record.run_id] = record
        return record

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.statuses or record.status in query.statuses)
            and (not query.kinds or record.kind in query.kinds)
            and (query.channel is None or record.channel == query.channel)
            and (query.correlator is None or query.correlator in record.correlators)
            and (
                query.due_at_or_before is None
                or (
                    record.next_wakeup_at is not None
                    and record.next_wakeup_at <= query.due_at_or_before
                )
            )
            and (
                query.open_at is None
                or (record.created_at <= query.open_at)
                and (record.deadline is None or record.deadline > query.open_at)
            )
        )
        return _page(sorted(rows, key=lambda item: item.run_id), query.page)

    async def record_artifact(self, scope, run_id, artifact_id, occurrence_id, occurred_at):
        current = await self.get(scope, run_id)
        if current is None:
            raise storage.StorageNotFoundError("external test run is absent")
        recent = tuple((*current.recent_artifact_ids, artifact_id)[-10:])
        updated = replace(
            current,
            revision=current.revision + 1,
            artifact_count=current.artifact_count + 1,
            first_artifact_at=current.first_artifact_at or occurred_at,
            last_artifact_at=occurred_at,
            recent_artifact_ids=recent,
        )
        self._records[run_id] = updated
        return updated


class InMemoryRunResultRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.RunResultRecord] = {}

    async def get(self, scope, run_id):
        record = self._records.get(run_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.run_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external test result next revision is invalid")
        self._records[record.run_id] = record
        return record

    async def delete(self, scope, run_id, expected_revision):
        current = await self.get(scope, run_id)
        if current is None:
            if expected_revision:
                raise storage.StorageConflictError("external test result is absent")
            return False
        _cas_revision(current, expected_revision)
        del self._records[run_id]
        return True


class InMemorySessionRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.SessionRecord] = {}

    async def create(self, record):
        current = self._records.get(record.session_id)
        if current is not None and current != record:
            raise storage.StorageIntegrityError("external test session identity conflicts")
        self._records[record.session_id] = record
        return record

    async def get(self, scope, session_id):
        record = self._records.get(session_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.session_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external test session next revision is invalid")
        self._records[record.session_id] = record
        return record

    async def delete(self, scope, session_id, expected_revision):
        current = await self.get(scope, session_id)
        if current is None:
            if expected_revision:
                raise storage.StorageConflictError("external test session is absent")
            return False
        _cas_revision(current, expected_revision)
        del self._records[session_id]
        return True

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.kinds or record.kind in query.kinds)
        )
        return _page(sorted(rows, key=lambda item: item.session_id), query.page)

    async def record_artifact(self, scope, session_id, occurrence_id, occurred_at):
        current = await self.get(scope, session_id)
        if current is None:
            raise storage.StorageNotFoundError("external test session is absent")
        updated = replace(
            current,
            revision=current.revision + 1,
            artifact_count=current.artifact_count + 1,
            last_artifact_at=occurred_at,
        )
        self._records[session_id] = updated
        return updated


class InMemoryContinuationRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.ContinuationRecord] = {}
        self._tokens: dict[str, str] = {}

    async def create(self, draft):
        if draft.continuation_id in self._records:
            raise storage.StorageIntegrityError("external test continuation already exists")
        token = f"external-token-{draft.continuation_id}"
        record = _promote(
            draft,
            storage.ContinuationRecord,
            token_digest=hashlib.sha256(token.encode()).hexdigest(),
            revision=1,
        )
        self._records[draft.continuation_id] = record
        self._tokens[token] = draft.continuation_id
        return storage.CreatedContinuation(record=record, token=token)

    async def get(self, scope, continuation_id):
        record = self._records.get(continuation_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def resolve_token(self, token):
        continuation_id = self._tokens.get(token)
        return self._records.get(continuation_id) if continuation_id is not None else None

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.continuation_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external continuation next revision is invalid")
        self._records[record.continuation_id] = record
        return record

    async def bind_correlator(self, scope, continuation_id, correlator, expected_revision):
        current = await self.get(scope, continuation_id)
        if current is None:
            raise storage.StorageNotFoundError("external test continuation is absent")
        _cas_revision(current, expected_revision)
        if correlator in current.correlators:
            return current
        updated = replace(
            current,
            correlators=(*current.correlators, correlator),
            revision=current.revision + 1,
        )
        self._records[continuation_id] = updated
        return updated

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.statuses or record.status in query.statuses)
            and (query.continuation_id is None or record.continuation_id == query.continuation_id)
            and (not query.kinds or record.kind in query.kinds)
        )
        return _page(sorted(rows, key=lambda item: item.continuation_id), query.page)


class InMemoryContinuationLeaseRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.ContinuationLeaseRecord] = {}

    async def get(self, scope, fire_id):
        record = self._records.get(fire_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def claim(self, request):
        current = self._records.get(request.fire_id)
        if (
            current is not None
            and current.lease_until is not None
            and current.lease_until > request.now
        ):
            return None
        record = storage.ContinuationLeaseRecord(
            fire_id=request.fire_id,
            continuation_id=request.continuation_id,
            scope=request.scope,
            scheduled_for=request.scheduled_for,
            status=storage.ContinuationLeaseStatus.LEASED,
            attempts=(current.attempts + 1 if current is not None else 1),
            revision=(current.revision + 1 if current is not None else 1),
            updated_at=request.now,
            worker_id=request.worker_id,
            lease_until=request.lease_until,
        )
        self._records[request.fire_id] = record
        return storage.ClaimedContinuationLease(record=record, reclaimed=current is not None)

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.fire_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external lease next revision is invalid")
        self._records[record.fire_id] = record
        return record

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.statuses or record.status in query.statuses)
        )
        return _page(sorted(rows, key=lambda item: item.fire_id), query.page)


class InMemoryTriggerRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.TriggerRecord] = {}
        self._claims: dict[str, storage.TriggerClaimRecord] = {}

    async def create(self, record):
        current = self._records.get(record.trigger_id)
        if current is not None and current != record:
            raise storage.StorageIntegrityError("external test trigger identity conflicts")
        self._records[record.trigger_id] = record
        return record

    async def get(self, scope, trigger_id):
        record = self._records.get(trigger_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def compare_and_set(self, record, expected_revision):
        current = self._records.get(record.trigger_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external trigger next revision is invalid")
        self._records[record.trigger_id] = record
        return record

    async def delete(self, scope, trigger_id, expected_revision):
        current = await self.get(scope, trigger_id)
        if current is None:
            if expected_revision:
                raise storage.StorageConflictError("external test trigger is absent")
            return False
        _cas_revision(current, expected_revision)
        del self._records[trigger_id]
        return True

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.kinds or record.kind in query.kinds)
            and (query.active is None or record.active is query.active)
            and (query.event_key is None or record.event_key == query.event_key)
        )
        return _page(sorted(rows, key=lambda item: item.trigger_id), query.page)

    async def claim_due(self, request):
        claimed: list[storage.ClaimedTrigger] = []
        for trigger in sorted(self._records.values(), key=lambda item: item.trigger_id):
            if len(claimed) >= request.limit:
                break
            if (
                not trigger.active
                or trigger.next_fire_at is None
                or trigger.next_fire_at > request.now
            ):
                continue
            if request.scope is not None and not _scope_matches(trigger.scope, request.scope):
                continue
            fire_id = f"{trigger.trigger_id}:{trigger.next_fire_at.isoformat()}"
            current = self._claims.get(fire_id)
            if (
                current is not None
                and current.lease_until is not None
                and current.lease_until > request.now
            ):
                continue
            claim = storage.TriggerClaimRecord(
                fire_id=fire_id,
                trigger_id=trigger.trigger_id,
                scope=trigger.scope,
                scheduled_for=trigger.next_fire_at,
                status=storage.TriggerClaimStatus.LEASED,
                attempts=(current.attempts + 1 if current is not None else 1),
                revision=(current.revision + 1 if current is not None else 1),
                updated_at=request.now,
                worker_id=request.worker_id,
                lease_until=request.lease_until,
            )
            self._claims[fire_id] = claim
            claimed.append(
                storage.ClaimedTrigger(
                    trigger=trigger,
                    claim=claim,
                    reclaimed=current is not None,
                )
            )
        return tuple(claimed)

    async def get_claim(self, scope, fire_id):
        record = self._claims.get(fire_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def compare_and_set_claim(self, record, expected_revision):
        current = self._claims.get(record.fire_id)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external trigger claim next revision is invalid")
        self._claims[record.fire_id] = record
        return record


class InMemoryIngressIdempotencyRepository:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], storage.IngressClaimRecord] = {}

    async def get(self, scope, deployment_id, integration_id, idempotency_key):
        record = self._records.get((deployment_id, integration_id, idempotency_key))
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def claim(self, request):
        identity = (request.deployment_id, request.integration_id, request.idempotency_key)
        current = self._records.get(identity)
        if current is not None:
            if (
                current.external_event_id != request.external_event_id
                or current.envelope_digest != request.envelope_digest
            ):
                raise storage.StorageIntegrityError("external ingress claim conflicts")
            return storage.IngressClaimResult(record=current, acquired=False)
        record = _promote(
            request,
            storage.IngressClaimRecord,
            status=storage.IngressClaimStatus.PENDING,
            revision=1,
        )
        self._records[identity] = record
        return storage.IngressClaimResult(record=record, acquired=True)

    async def complete(self, record, expected_revision):
        identity = (record.deployment_id, record.integration_id, record.idempotency_key)
        current = self._records.get(identity)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external ingress next revision is invalid")
        self._records[identity] = record
        return record


class InMemoryIntegrationSessionRepository:
    def __init__(self, sessions: InMemorySessionRepository) -> None:
        self._sessions = sessions
        self._records: dict[str, storage.ExternalSessionBindingRecord] = {}

    async def get_binding(self, scope, route_id):
        record = self._records.get(route_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def provision(self, request, session):
        current = self._records.get(request.route_id)
        if current is not None and (
            current.binding_id != request.binding_id
            or current.build_id != request.build_id
            or current.ag_session_id != request.ag_session_id
            or current.scope != request.scope
        ):
            raise storage.StorageIntegrityError("external session binding conflicts")
        stored_session = self._sessions._records.get(request.ag_session_id)
        if stored_session is not None and (
            stored_session.kind != session.kind
            or stored_session.scope != session.scope
            or stored_session.source != session.source
            or stored_session.external_reference != session.external_reference
        ):
            raise storage.StorageIntegrityError("external integration session conflicts")
        session_created = stored_session is None
        if session_created:
            self._sessions._records[session.session_id] = session
            stored_session = session
        if current is not None:
            updated = replace(current, last_seen_at=request.now, revision=current.revision + 1)
            self._records[request.route_id] = updated
            return storage.IntegrationSessionProvisioningResult(
                session=stored_session,
                binding=updated,
                session_created=session_created,
                binding_created=False,
            )
        record = storage.ExternalSessionBindingRecord(
            binding_id=request.binding_id,
            route_id=request.route_id,
            build_id=request.build_id,
            ag_session_id=request.ag_session_id,
            scope=request.scope,
            revision=1,
            created_at=request.now,
            last_seen_at=request.now,
        )
        self._records[request.route_id] = record
        return storage.IntegrationSessionProvisioningResult(
            session=stored_session,
            binding=record,
            session_created=session_created,
            binding_created=True,
        )


class InMemoryInboundEventRepository:
    def __init__(self) -> None:
        self._records: dict[str, storage.InboundEventRecord] = {}
        self._cursor = 0

    async def append(self, event):
        current = self._records.get(event.event_id)
        if current is not None:
            comparable = _promote(
                event,
                storage.InboundEventRecord,
                delivery_cursor=current.delivery_cursor,
                cursor=current.cursor,
            )
            if current != comparable:
                raise storage.StorageIntegrityError("external inbound event conflicts")
            return current
        self._cursor += 1
        record = _promote(
            event,
            storage.InboundEventRecord,
            delivery_cursor=self._cursor,
            cursor=f"external-inbound:{self._cursor}",
        )
        self._records[event.event_id] = record
        return record

    async def get(self, scope, event_id):
        record = self._records.get(event_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None


class InMemorySemanticEventRepository:
    def __init__(self, delivery_cursors: InMemoryDeliveryCursorAllocator) -> None:
        self._records: dict[str, storage.SemanticEventRecord] = {}
        self._delivery_cursors = delivery_cursors

    async def append(self, event):
        current = self._records.get(event.event_id)
        if current is not None:
            raise storage.StorageIntegrityError("external semantic event identity conflicts")
        delivery_cursor = self._delivery_cursors.next()
        record = _promote(
            event,
            storage.SemanticEventRecord,
            delivery_cursor=delivery_cursor,
            cursor=f"external-semantic:{delivery_cursor}",
        )
        self._records[event.event_id] = record
        return record

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and record.deployment_id == query.deployment_id
            and (not query.kinds or record.kind in query.kinds)
            and (
                query.after_delivery_cursor is None
                or record.delivery_cursor > query.after_delivery_cursor
            )
            and (query.turn_id is None or record.turn_id == query.turn_id)
        )
        return _page(sorted(rows, key=lambda item: item.delivery_cursor), query.page)


class InMemoryRuntimeOutputSink:
    def __init__(self, delivery_cursors: InMemoryDeliveryCursorAllocator) -> None:
        self._delivery_cursors = delivery_cursors
        self.frames: list[storage.RuntimeOutputFrame] = []
        self._records: dict[str, storage.RuntimeOutputRecord] = {}
        self.flushed_executions: list[str] = []
        self.flushed_runs: list[str] = []

    def emit(self, frame):
        if any(item.output_id == frame.output_id for item in self.frames):
            raise storage.StorageIntegrityError("external runtime output identity conflicts")
        self.frames.append(frame)

    async def flush_execution(self, execution_id):
        self._commit(lambda frame: frame.execution_id == execution_id)
        self.flushed_executions.append(execution_id)

    async def flush_run(self, run_id):
        self._commit(lambda frame: frame.scope.run_id == run_id)
        self.flushed_runs.append(run_id)

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (
                query.after_delivery_cursor is None
                or record.delivery_cursor > query.after_delivery_cursor
            )
            and (not query.streams or record.stream in query.streams)
            and (query.execution_id is None or record.execution_id == query.execution_id)
        )
        return _page(sorted(rows, key=lambda item: item.delivery_cursor), query.page)

    def _commit(self, selected) -> None:
        selected_frames = tuple(frame for frame in self.frames if selected(frame))
        committed_sequences = {
            (record.execution_id, record.sequence): record.output_id
            for record in self._records.values()
        }
        staged_sequences: dict[tuple[str, int], str] = {}
        for frame in selected_frames:
            current = self._records.get(frame.output_id)
            if current is not None:
                comparable = _promote(
                    frame,
                    storage.RuntimeOutputRecord,
                    delivery_cursor=current.delivery_cursor,
                    cursor=current.cursor,
                )
                if current != comparable:
                    raise storage.StorageIntegrityError(
                        "external runtime output identity conflicts"
                    )
                continue
            key = (frame.execution_id, frame.sequence)
            conflicting_id = committed_sequences.get(key) or staged_sequences.get(key)
            if conflicting_id is not None and conflicting_id != frame.output_id:
                raise storage.StorageIntegrityError(
                    "external runtime output execution sequence conflicts"
                )
            staged_sequences[key] = frame.output_id

        remaining: list[storage.RuntimeOutputFrame] = []
        for frame in self.frames:
            if not selected(frame):
                remaining.append(frame)
                continue
            current = self._records.get(frame.output_id)
            if current is not None:
                comparable = _promote(
                    frame,
                    storage.RuntimeOutputRecord,
                    delivery_cursor=current.delivery_cursor,
                    cursor=current.cursor,
                )
                continue
            delivery_cursor = self._delivery_cursors.next()
            self._records[frame.output_id] = _promote(
                frame,
                storage.RuntimeOutputRecord,
                delivery_cursor=delivery_cursor,
                cursor=f"external-runtime-output:{delivery_cursor}",
            )
        self.frames = remaining


class InMemoryObservationRepository:
    def __init__(self, clock) -> None:
        self._clock = clock
        self._cursor = 0
        self._records: dict[str, storage.ObservationRecord] = {}
        self._llm_calls: dict[str, storage.LLMCallDetail] = {}
        self._management: dict[
            tuple[storage.StorageScope, str], storage.ObservationScopeManagementRecord
        ] = {}
        self.trace_queries: list[storage.ObservationTraceSummaryQuery] = []
        self.llm_queries: list[storage.ObservationLLMSummaryQuery] = []

    async def append_many(self, observations):
        committed: list[storage.ObservationRecord] = []
        for draft in observations:
            current = self._records.get(draft.observation_id)
            if current is not None:
                comparable = _promote(
                    draft,
                    storage.ObservationRecord,
                    cursor=current.cursor,
                )
                if current != comparable:
                    raise storage.StorageIntegrityError("external observation identity conflicts")
                committed.append(current)
                continue
            self._cursor += 1
            record = _promote(
                draft,
                storage.ObservationRecord,
                cursor=f"external-observation:{self._cursor}",
            )
            self._records[draft.observation_id] = record
            committed.append(record)
        return tuple(committed)

    async def get(self, scope, observation_id):
        record = self._records.get(observation_id)
        return record if record is not None and _scope_matches(record.scope, scope) else None

    async def query(self, query):
        rows = (
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and (not query.categories or record.category in query.categories)
            and (not query.names or record.name in query.names)
            and (not query.producers or record.producer in query.producers)
            and (not query.statuses or record.status in query.statuses)
            and (not query.severities or record.severity in query.severities)
            and (query.trace_id is None or record.trace_id == query.trace_id)
            and (query.turn_id is None or record.turn_id == query.turn_id)
            and (
                query.resource_key is None
                or any(
                    resource.resource_key == query.resource_key
                    and (
                        query.resource_relation is None
                        or resource.relation is query.resource_relation
                    )
                    for resource in record.resource_links
                )
            )
            and (
                query.occurred_at_or_after is None
                or record.occurred_at >= query.occurred_at_or_after
            )
            and (
                query.occurred_at_or_before is None
                or record.occurred_at <= query.occurred_at_or_before
            )
        )
        return _page(sorted(rows, key=lambda item: item.cursor), query.page)

    async def append_llm_call(self, call):
        observations = await self.append_many((call.observation,))
        observation = observations[0]
        values = {
            item.name: getattr(call, item.name)
            for item in fields(storage.LLMCallRecord)
            if hasattr(call, item.name)
        }
        values["observation"] = observation
        values["trace_payload_preview"] = call.trace_payload
        record = storage.LLMCallRecord(**values)
        detail = storage.LLMCallDetail(
            record=record,
            captured_request=call.captured_request,
            captured_response=call.captured_response,
            trace_payload=call.trace_payload,
        )
        current = self._llm_calls.get(call.llm_call_id)
        if current is not None and current != detail:
            raise storage.StorageIntegrityError("external LLM call identity conflicts")
        self._llm_calls[call.llm_call_id] = detail
        return record

    async def get_llm_call(self, scope, llm_call_id):
        detail = self._llm_calls.get(llm_call_id)
        if detail is None or not _scope_matches(detail.record.observation.scope, scope):
            return None
        return detail

    async def query_llm_calls(self, query):
        rows = (
            detail.record
            for detail in self._llm_calls.values()
            if _scope_matches(detail.record.observation.scope, query.scope)
            and (query.trace_id is None or detail.record.observation.trace_id == query.trace_id)
            and (not query.providers or detail.record.provider in query.providers)
            and (not query.models or detail.record.model in query.models)
            and (not query.call_types or detail.record.call_type in query.call_types)
            and (
                not query.prompt_manifest_ids
                or detail.record.prompt_manifest_id in query.prompt_manifest_ids
            )
            and (not query.statuses or detail.record.observation.status in query.statuses)
            and (
                query.occurred_at_or_after is None
                or detail.record.observation.occurred_at >= query.occurred_at_or_after
            )
            and (
                query.occurred_at_or_before is None
                or detail.record.observation.occurred_at <= query.occurred_at_or_before
            )
        )
        return _page(sorted(rows, key=lambda item: item.observation.cursor), query.page)

    async def get_scope_management(self, scope, scope_key):
        return self._management.get((scope, scope_key))

    async def compare_and_set_scope_management(self, record, expected_revision):
        identity = (record.scope, record.scope_key)
        current = self._management.get(identity)
        _cas_revision(current, expected_revision)
        if record.revision != expected_revision + 1:
            raise storage.StorageConflictError("external management next revision is invalid")
        self._management[identity] = record
        return record

    async def query_scope_management(self, query):
        rows = (
            record
            for record in self._management.values()
            if _scope_matches(record.scope, query.scope)
            and (query.trace_id is None or record.trace_id == query.trace_id)
            and (query.pinned is None or record.pinned is query.pinned)
            and (query.hidden is None or record.hidden is query.hidden)
            and (query.deleted is None or record.deleted is query.deleted)
            and (not query.retention_classes or record.retention_class in query.retention_classes)
        )
        return _page(sorted(rows, key=lambda item: item.scope_key), query.page)

    async def query_scope_usage(self, query):
        grouped: dict[str, list[storage.ObservationRecord]] = {}
        for record in self._records.values():
            if not _scope_matches(record.scope, query.scope):
                continue
            scope_id = (
                record.trace_id
                if query.dimension is storage.ObservationUsageDimension.TRACE
                else record.scope.run_id
            )
            if scope_id is not None:
                grouped.setdefault(scope_id, []).append(record)
        rows = (
            storage.ObservationScopeUsageRecord(
                dimension=query.dimension,
                scope_id=scope_id,
                latest_at=max(item.occurred_at for item in records),
                observation_count=len(records),
                logical_bytes=sum(len(item.summary.encode()) for item in records),
            )
            for scope_id, records in grouped.items()
        )
        return _page(sorted(rows, key=lambda item: item.scope_id), query.page)

    async def purge(self, request):
        matching = [
            identity
            for identity, record in self._records.items()
            if _scope_matches(record.scope, request.scope)
        ][: request.max_observations]
        if not request.dry_run:
            for identity in matching:
                del self._records[identity]
        count = len(matching)
        return storage.ObservationPurgeResult(
            dry_run=request.dry_run,
            matching_traces=0,
            matching_observations=count,
            matching_manifests=0,
            exclusive_fragment_bytes=0,
            shared_fragment_bytes_retained=0,
            estimated_reclaimed_bytes=0,
            deleted_observations=0 if request.dry_run else count,
        )

    async def storage_stats(self, scope):
        observations = [
            record for record in self._records.values() if _scope_matches(record.scope, scope)
        ]
        llm_calls = [
            detail
            for detail in self._llm_calls.values()
            if _scope_matches(detail.record.observation.scope, scope)
        ]
        logical_bytes = sum(len(record.summary.encode()) for record in observations)
        return storage.ObservationStorageStats(
            observations=len(observations),
            llm_calls=len(llm_calls),
            manifests=0,
            fragments=0,
            fragment_bytes=0,
            logical_bytes=logical_bytes,
        )

    async def summarize_traces(self, query):
        self.trace_queries.append(query)
        rows = [
            record
            for record in self._records.values()
            if _scope_matches(record.scope, query.scope)
            and record.category == "trace"
            and (
                query.occurred_at_or_after is None
                or record.occurred_at >= query.occurred_at_or_after
            )
            and (
                query.occurred_at_or_before is None
                or record.occurred_at <= query.occurred_at_or_before
            )
        ]
        trace_ids = tuple(
            sorted({record.trace_id for record in rows if record.trace_id is not None})
        )
        errors = [record for record in rows if record.status is storage.ObservationStatus.ERROR]
        failing_services: dict[str, int] = {}
        for record in errors:
            if record.producer is not None:
                failing_services[record.producer] = failing_services.get(record.producer, 0) + 1
        ordered_services = sorted(
            failing_services.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return storage.ObservationTraceSummaryRecord(
            span_count=len(rows),
            error_count=len(errors),
            total_duration_ms=sum(
                value
                for record in rows
                if isinstance((value := record.attributes.get("duration_ms")), int)
                and not isinstance(value, bool)
            ),
            trace_id_count=len(trace_ids),
            trace_ids=trace_ids[: query.trace_id_limit],
            trace_ids_truncated=len(trace_ids) > query.trace_id_limit,
            top_failing_services=dict(ordered_services[: query.failing_service_limit]),
            latest_error_at=max((record.occurred_at for record in errors), default=None),
        )

    async def summarize_llm_calls(self, query):
        self.llm_queries.append(query)
        rows = [
            detail.record
            for detail in self._llm_calls.values()
            if _scope_matches(detail.record.observation.scope, query.scope)
            and (
                query.occurred_at_or_after is None
                or detail.record.observation.occurred_at >= query.occurred_at_or_after
            )
            and (
                query.occurred_at_or_before is None
                or detail.record.observation.occurred_at <= query.occurred_at_or_before
            )
        ]
        by_model = {
            model: sum(record.model == model for record in rows)
            for model in sorted({record.model for record in rows})
        }
        ordered_models = sorted(by_model.items(), key=lambda item: (-item[1], item[0]))

        def _usage(record, *names):
            for name in names:
                value = record.usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
            return 0

        prompt_tokens = sum(_usage(record, "input_tokens", "prompt_tokens") for record in rows)
        completion_tokens = sum(
            _usage(record, "output_tokens", "completion_tokens") for record in rows
        )
        return storage.ObservationLLMSummaryRecord(
            total_calls=len(rows),
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            error_count=sum(record.error_type is not None for record in rows),
            model_count=len(by_model),
            by_model=dict(ordered_models[: query.model_limit]),
            by_model_truncated=len(by_model) > query.model_limit,
        )
