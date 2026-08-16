"""Inactive canonical auth grant and invite persistence for the S9 cut."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from aethergraph.services.auth.authn import DemoGrant, InviteCode
from aethergraph.services.canonical_storage_scope import validate_storage_owner_scope
from aethergraph.storage.contracts import (
    FrozenJson,
    KeyValueQuery,
    KeyValueRecord,
    KeyValueStore,
    PageRequest,
    StorageBundle,
    StorageCapacityError,
    StorageConflictError,
    StorageIntegrityError,
    StorageScope,
)

_GRANT_NAMESPACE = "auth.grants"
_INVITE_NAMESPACE = "auth.invites"
_SCHEMA_VERSION = 1
_PAGE_SIZE = 100
_MAX_RECORDS = 10_000
_MAX_CAS_ATTEMPTS = 8
_COMPATIBILITY_METADATA = "compatibility_metadata"
_ALLOWED_APPS = "allowed_apps"
_REMOVAL = "future breaking release"


class CanonicalAuthStore:
    """Persist demo grants and invites through canonical revisioned KV stores."""

    def __init__(
        self,
        *,
        grant_repository: KeyValueStore,
        invite_repository: KeyValueStore,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind auth persistence to exact repositories from one open bundle.

        Intro:
            Captures provider-owned grant and invite repositories plus one trusted
            owner scope without selecting a provider or performing storage I/O.

        Examples:
            Bind repositories from a production bundle:
            ```python
            store = CanonicalAuthStore(
                grant_repository=bundle.auth_grants,
                invite_repository=bundle.auth_invites,
                owner_scope=owner_scope,
                clock=clock,
            )
            ```

            Bind deterministic test repositories:
            ```python
            store = CanonicalAuthStore(
                grant_repository=fake_kv,
                invite_repository=fake_kv,
                owner_scope=StorageScope(project_id="project-1"),
                clock=lambda: fixed_now,
            )
            ```

        Args:
            grant_repository: Exact canonical auth-grant KV repository.
            invite_repository: Exact canonical auth-invite KV repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC clock used for domain expiry checks.

        Returns:
            None: The inactive-until-S9 auth projection is ready.

        Notes:
            The bundle owns repository lifecycle. This service has no opener,
            fallback, cache authority, or close path.
        """
        validate_storage_owner_scope(owner_scope)
        _utc(clock(), field="clock")
        self._grant_repository = grant_repository
        self._invite_repository = invite_repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def get_grant(self, grant_id: str) -> DemoGrant | None:
        """Read one unexpired grant by exact provider identity.

        Intro:
            Uses one scoped key lookup and validates the canonical grant payload
            before returning the frozen auth-domain model.

        Examples:
            Load a grant during invite redemption:
            ```python
            grant = await store.get_grant("grant-1")
            ```

            Detect an absent or provider-expired grant:
            ```python
            assert await store.get_grant("missing") is None
            ```

        Args:
            grant_id: Exact stable grant identity.

        Returns:
            DemoGrant | None: Validated grant or `None` when absent or expired.

        Notes:
            Revocation remains visible in the returned domain record so the caller
            can distinguish policy denial from storage absence.
        """
        record = await self._grant_repository.get(
            self._owner_scope,
            _GRANT_NAMESPACE,
            _identity("grant_id", grant_id),
        )
        return _decode_grant(record) if record is not None else None

    async def save_grant(self, grant: DemoGrant) -> DemoGrant:
        """Create or replace one grant through bounded revision-CAS retry.

        Intro:
            Preserves the frozen unconditional auth-service operation while making
            the provider record and its revision the only persistence authority.

        Examples:
            Persist a newly issued grant:
            ```python
            saved = await store.save_grant(grant)
            ```

            Persist a revoked replacement:
            ```python
            saved = await store.save_grant(grant.model_copy(update={"revoked": True}))
            ```

        Args:
            grant: Complete next grant value.

        Returns:
            DemoGrant: Detached validated grant committed by the provider.

        Notes:
            Legacy App allowlists are stored only in marked deprecated compatibility
            metadata and never in provider scope, columns, indexes, or key identity.
        """
        record = await self._replace(
            repository=self._grant_repository,
            namespace=_GRANT_NAMESPACE,
            key=_identity("grant_id", grant.grant_id),
            value=_encode_grant(grant),
            expires_at=grant.expires_at,
        )
        return _decode_grant(record)

    async def delete_grant(self, grant_id: str) -> bool:
        """Delete one grant through an exact bounded revision-CAS operation.

        Intro:
            Resolves the current provider revision and deletes only that revision,
            retrying a bounded number of times when a concurrent update wins.

        Examples:
            Delete an existing grant:
            ```python
            deleted = await store.delete_grant("grant-1")
            ```

            Delete an absent grant idempotently:
            ```python
            assert await store.delete_grant("missing") is False
            ```

        Args:
            grant_id: Exact stable grant identity.

        Returns:
            bool: `True` when a provider record was deleted; otherwise `False`.

        Notes:
            Exhausted conflicts propagate. No alternate store or tombstone is used.
        """
        return await self._delete(
            repository=self._grant_repository,
            namespace=_GRANT_NAMESPACE,
            key=_identity("grant_id", grant_id),
        )

    async def get_invite(self, code: str) -> InviteCode | None:
        """Read one unexpired invite by exact provider identity.

        Intro:
            Performs one scoped lookup and validates the stored invite payload
            without consulting an index record or in-memory cache.

        Examples:
            Load an invite for display:
            ```python
            invite = await store.get_invite("DEMO-ABC")
            ```

            Detect an absent invite:
            ```python
            assert await store.get_invite("missing") is None
            ```

        Args:
            code: Exact invite code.

        Returns:
            InviteCode | None: Validated invite or `None` when absent or expired.

        Notes:
            The invite code is already a bearer secret in the frozen API; this
            migration does not add a second token or lookup identity.
        """
        record = await self._invite_repository.get(
            self._owner_scope,
            _INVITE_NAMESPACE,
            _identity("code", code),
        )
        return _decode_invite(record) if record is not None else None

    async def create_invite(self, invite: InviteCode) -> InviteCode:
        """Create one invite only when its exact code is absent.

        Intro:
            Uses provider revision zero as the uniqueness check, replacing the
            legacy process-local dictionary and synthetic `_index` authority.

        Examples:
            Create a generated invite:
            ```python
            created = await store.create_invite(invite)
            ```

            Reject a duplicate custom code:
            ```python
            await store.create_invite(existing_invite)
            ```

        Args:
            invite: Complete new invite value with a stable code.

        Returns:
            InviteCode: Detached validated invite committed at revision one.

        Notes:
            Duplicate identities raise `StorageConflictError` directly; callers may
            translate that domain outcome at the HTTP boundary during S9.
        """
        record = await self._invite_repository.compare_and_set(
            self._owner_scope,
            _INVITE_NAMESPACE,
            _identity("code", invite.code),
            0,
            _encode_invite(invite),
            invite.expires_at,
        )
        return _decode_invite(record)

    async def save_invite(self, invite: InviteCode) -> InviteCode:
        """Replace one invite through bounded revision-CAS retry.

        Intro:
            Supports explicit administrative update and deactivation while keeping
            the provider record authoritative under concurrent mutation.

        Examples:
            Deactivate an invite:
            ```python
            saved = await store.save_invite(invite.model_copy(update={"active": False}))
            ```

            Change the usage limit:
            ```python
            saved = await store.save_invite(invite.model_copy(update={"max_uses": 5}))
            ```

        Args:
            invite: Complete next invite value.

        Returns:
            InviteCode: Detached validated invite committed by the provider.

        Notes:
            This compatibility operation is bounded and revisioned; it never writes
            a synthetic invite index.
        """
        record = await self._replace(
            repository=self._invite_repository,
            namespace=_INVITE_NAMESPACE,
            key=_identity("code", invite.code),
            value=_encode_invite(invite),
            expires_at=invite.expires_at,
        )
        return _decode_invite(record)

    async def claim_invite(self, code: str) -> tuple[InviteCode, DemoGrant]:
        """Atomically claim one allowed invite use and resolve its current grant.

        Intro:
            Validates the current provider records, then advances the invite use
            counter with exact revision CAS so concurrent redemptions cannot exceed
            the authored usage limit.

        Examples:
            Claim a reusable invite:
            ```python
            invite, grant = await store.claim_invite("DEMO-ABC")
            ```

            Surface an exhausted invite as a domain error:
            ```python
            await store.claim_invite("DEMO-EXHAUSTED")
            ```

        Args:
            code: Exact invite code to claim.

        Returns:
            tuple[InviteCode, DemoGrant]: Committed invite revision and valid grant.

        Notes:
            Provider conflicts retry only the exact claim. Missing, inactive,
            expired, exhausted, revoked, or expired-grant states fail closed.
        """
        key = _identity("code", code)
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await self._invite_repository.get(
                self._owner_scope,
                _INVITE_NAMESPACE,
                key,
            )
            if current is None:
                raise ValueError("Invalid invite code")
            invite = _decode_invite(current)
            now = _utc(self._clock(), field="clock")
            _validate_invite_claim(invite, now)
            grant = await self.get_grant(invite.grant_id)
            if grant is None or grant.revoked:
                raise ValueError("Invite code grant is no longer valid")
            if grant.expires_at is not None and grant.expires_at <= now:
                raise ValueError("Invite code grant is no longer valid")
            claimed = invite.model_copy(update={"uses": invite.uses + 1})
            try:
                record = await self._invite_repository.compare_and_set(
                    self._owner_scope,
                    _INVITE_NAMESPACE,
                    key,
                    current.revision,
                    _encode_invite(claimed),
                    claimed.expires_at,
                )
            except StorageConflictError:
                continue
            return _decode_invite(record), grant
        raise StorageConflictError("Invite claim exceeded bounded CAS retries")

    async def delete_invite(self, code: str) -> bool:
        """Delete one invite through an exact bounded revision-CAS operation.

        Intro:
            Removes only the provider-authoritative invite key and requires no
            companion index maintenance.

        Examples:
            Delete an existing invite:
            ```python
            deleted = await store.delete_invite("DEMO-ABC")
            ```

            Delete an absent invite idempotently:
            ```python
            assert await store.delete_invite("missing") is False
            ```

        Args:
            code: Exact invite code.

        Returns:
            bool: `True` when deleted; otherwise `False`.

        Notes:
            Exhausted conflicts propagate without fallback or a second authority.
        """
        return await self._delete(
            repository=self._invite_repository,
            namespace=_INVITE_NAMESPACE,
            key=_identity("code", code),
        )

    async def list_grants(self) -> list[DemoGrant]:
        """List all grants through bounded provider-cursor pages.

        Intro:
            Scans only the canonical grant namespace and fails explicitly if the
            administrative safety bound would be exceeded.

        Examples:
            List grants for an admin response:
            ```python
            grants = await store.list_grants()
            ```

            Build an id lookup from the bounded result:
            ```python
            grants_by_id = {item.grant_id: item for item in await store.list_grants()}
            ```

        Args:
            None: The store is already bound to one trusted owner scope.

        Returns:
            list[DemoGrant]: Key-ordered detached grant models.

        Notes:
            There is no unbounded request and no legacy `_index` record.
        """
        records = await self._scan(self._grant_repository, _GRANT_NAMESPACE)
        return [_decode_grant(record) for record in records]

    async def list_invites(self) -> list[InviteCode]:
        """List all invites through bounded provider-cursor pages.

        Intro:
            Scans the exact invite namespace instead of trusting a mutable synthetic
            index that can omit provider records.

        Examples:
            List invites for administration:
            ```python
            invites = await store.list_invites()
            ```

            Select active invites from the bounded result:
            ```python
            active = [item for item in await store.list_invites() if item.active]
            ```

        Args:
            None: The store is already bound to one trusted owner scope.

        Returns:
            list[InviteCode]: Key-ordered detached invite models.

        Notes:
            The provider applies TTL visibility before each bounded page.
        """
        records = await self._scan(self._invite_repository, _INVITE_NAMESPACE)
        return [_decode_invite(record) for record in records]

    async def _replace(
        self,
        *,
        repository: KeyValueStore,
        namespace: str,
        key: str,
        value: FrozenJson,
        expires_at: datetime | None,
    ) -> KeyValueRecord:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await repository.get(self._owner_scope, namespace, key)
            try:
                return await repository.compare_and_set(
                    self._owner_scope,
                    namespace,
                    key,
                    current.revision if current is not None else 0,
                    value,
                    expires_at,
                )
            except StorageConflictError:
                continue
        raise StorageConflictError(f"{namespace} replacement exceeded bounded CAS retries")

    async def _delete(
        self,
        *,
        repository: KeyValueStore,
        namespace: str,
        key: str,
    ) -> bool:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await repository.get(self._owner_scope, namespace, key)
            if current is None:
                return False
            try:
                return await repository.delete(
                    self._owner_scope,
                    namespace,
                    key,
                    current.revision,
                )
            except StorageConflictError:
                continue
        raise StorageConflictError(f"{namespace} deletion exceeded bounded CAS retries")

    async def _scan(
        self,
        repository: KeyValueStore,
        namespace: str,
    ) -> list[KeyValueRecord]:
        records: list[KeyValueRecord] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            remaining = _MAX_RECORDS - len(records)
            if remaining == 0:
                raise StorageCapacityError(f"{namespace} exceeds {_MAX_RECORDS} records")
            page = await repository.scan(
                KeyValueQuery(
                    scope=self._owner_scope,
                    namespace=namespace,
                    page=PageRequest(limit=min(_PAGE_SIZE, remaining), cursor=cursor),
                )
            )
            records.extend(page.items)
            if page.next_cursor is None:
                return records
            if not page.items or page.next_cursor in seen_cursors:
                raise StorageIntegrityError(
                    f"{namespace} returned a non-progressing provider cursor"
                )
            seen_cursors.add(page.next_cursor)
            cursor = page.next_cursor


def bind_canonical_auth_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalAuthStore:
    """Bind canonical auth persistence to exact fields from one open bundle.

    Intro:
        Constructs the inactive auth projection without provider selection, I/O,
        lifecycle ownership, or legacy-store discovery.

    Examples:
        Bind production composition inputs:
        ```python
        auth_store = bind_canonical_auth_store(
            bundle=bundle, owner_scope=owner_scope, clock=clock
        )
        ```

        Bind a conformance fake bundle:
        ```python
        auth_store = bind_canonical_auth_store(
            bundle=fake_bundle, owner_scope=test_scope, clock=lambda: fixed_now
        )
        ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC clock for domain expiry checks.

    Returns:
        CanonicalAuthStore: Exact grant and invite service projection.

    Notes:
        Binding has no fallback and does not activate the S9 composition path.
    """
    return CanonicalAuthStore(
        grant_repository=bundle.auth_grants,
        invite_repository=bundle.auth_invites,
        owner_scope=owner_scope,
        clock=clock,
    )


def _encode_grant(grant: DemoGrant) -> dict[str, FrozenJson]:
    payload = grant.model_dump(mode="json", exclude={_ALLOWED_APPS})
    payload["schema_version"] = _SCHEMA_VERSION
    if grant.allowed_apps:
        payload[_COMPATIBILITY_METADATA] = {
            _ALLOWED_APPS: {
                "values": list(grant.allowed_apps),
                "deprecated": True,
                "compatibility_only": True,
                "scheduled_removal": _REMOVAL,
            }
        }
    return payload


def _decode_grant(record: KeyValueRecord) -> DemoGrant:
    payload = _mapping(record, expected_namespace=_GRANT_NAMESPACE)
    if payload.pop("schema_version", None) != _SCHEMA_VERSION:
        raise StorageIntegrityError("Unsupported canonical auth grant schema")
    for forbidden in (_ALLOWED_APPS, "app_id", "application_id", "client_id"):
        if forbidden in payload:
            raise StorageIntegrityError(f"Canonical auth grant contains forbidden {forbidden}")
    allowed_apps: list[str] = []
    compatibility = payload.pop(_COMPATIBILITY_METADATA, None)
    if compatibility is not None:
        if not isinstance(compatibility, Mapping) or set(compatibility) != {_ALLOWED_APPS}:
            raise StorageIntegrityError("Malformed canonical auth compatibility metadata")
        app_entry = compatibility[_ALLOWED_APPS]
        if not isinstance(app_entry, Mapping):
            raise StorageIntegrityError("Malformed canonical auth App compatibility metadata")
        if (
            app_entry.get("deprecated") is not True
            or app_entry.get("compatibility_only") is not True
            or app_entry.get("scheduled_removal") != _REMOVAL
        ):
            raise StorageIntegrityError("Unmarked canonical auth App compatibility metadata")
        values = app_entry.get("values")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not all(isinstance(value, str) and value.strip() for value in values)
        ):
            raise StorageIntegrityError("Malformed canonical auth App compatibility values")
        allowed_apps = values
    payload[_ALLOWED_APPS] = allowed_apps
    try:
        grant = DemoGrant.model_validate(payload)
    except Exception as exc:
        raise StorageIntegrityError("Malformed canonical auth grant") from exc
    if grant.grant_id != record.key:
        raise StorageIntegrityError("Canonical auth grant identity mismatch")
    return grant


def _encode_invite(invite: InviteCode) -> dict[str, FrozenJson]:
    payload = invite.model_dump(mode="json")
    payload["schema_version"] = _SCHEMA_VERSION
    return payload


def _decode_invite(record: KeyValueRecord) -> InviteCode:
    payload = _mapping(record, expected_namespace=_INVITE_NAMESPACE)
    if payload.pop("schema_version", None) != _SCHEMA_VERSION:
        raise StorageIntegrityError("Unsupported canonical auth invite schema")
    for forbidden in ("app_id", "application_id", "client_id", _COMPATIBILITY_METADATA):
        if forbidden in payload:
            raise StorageIntegrityError(f"Canonical auth invite contains forbidden {forbidden}")
    try:
        invite = InviteCode.model_validate(payload)
    except Exception as exc:
        raise StorageIntegrityError("Malformed canonical auth invite") from exc
    if invite.code != record.key:
        raise StorageIntegrityError("Canonical auth invite identity mismatch")
    return invite


def _mapping(record: KeyValueRecord, *, expected_namespace: str) -> dict[str, object]:
    if record.namespace != expected_namespace:
        raise StorageIntegrityError("Canonical auth record namespace mismatch")
    if not isinstance(record.value, Mapping):
        raise StorageIntegrityError("Canonical auth record must contain a JSON object")
    return dict(record.value)


def _validate_invite_claim(invite: InviteCode, now: datetime) -> None:
    if not invite.active:
        raise ValueError("Invite code is deactivated")
    if invite.expires_at is not None and invite.expires_at <= now:
        raise ValueError("Invite code expired")
    if invite.max_uses is not None and invite.uses >= invite.max_uses:
        raise ValueError("Invite code has reached its usage limit")


def _identity(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value
