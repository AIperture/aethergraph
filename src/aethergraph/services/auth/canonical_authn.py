"""Asynchronous public Authn behavior over canonical provider persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
import secrets
from typing import Any, Literal

from aethergraph.services.auth.authn import (
    AuthenticationRejected,
    AuthSession,
    DemoGrant,
    InviteCode,
    ResolvedAuth,
)
from aethergraph.storage.contracts import StorageConflictError

from .canonical_store import CanonicalAuthStore


class CanonicalAuthnService:
    """Resolve sessions and demo grants through one canonical Auth store."""

    def __init__(
        self,
        *,
        store: CanonicalAuthStore,
        secret: str,
        cookie_name: str = "ag_auth_session",
        cookie_secure: bool = False,
        cookie_samesite: Literal["lax", "strict", "none"] = "lax",
        session_ttl_seconds: int = 24 * 3600,
        grant_ttl_seconds: int = 7 * 24 * 3600,
        public_demo_fallback_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        guest_id_factory: Callable[[], str] | None = None,
        session_id_factory: Callable[[], str] | None = None,
        invite_code_factory: Callable[[], str] | None = None,
    ) -> None:
        """Bind asynchronous Authn behavior to one canonical Auth store.

        Intro:
            Construction retains configuration and an already-bound provider service.
            Only short-lived HTTP session state remains process-local.

        Examples:
            Bind production Authn:
                ```python
                authn = CanonicalAuthnService(
                    store=auth_store,
                    secret=settings.auth.secret.get_secret_value(),
                )
                ```

            Bind deterministic tests:
                ```python
                authn = CanonicalAuthnService(
                    store=auth_store,
                    secret="test-secret",
                    clock=lambda: fixed_now,
                    session_id_factory=lambda: "session-1",
                )
                ```

        Args:
            store: Exact canonical Auth persistence service.
            secret: Non-empty Authn secret retained for the public service contract.
            cookie_name: Non-empty HTTP session-cookie name.
            cookie_secure: Whether the HTTP session cookie requires a secure transport.
            cookie_samesite: Exact cookie same-site policy.
            session_ttl_seconds: Positive in-memory session lifetime in seconds.
            grant_ttl_seconds: Positive default provider grant lifetime in seconds.
            public_demo_fallback_enabled: Whether authored public-demo resolution is enabled.
            clock: Optional timezone-aware UTC clock.
            guest_id_factory: Optional exact non-empty guest identity factory.
            session_id_factory: Optional exact non-empty session identity factory.
            invite_code_factory: Optional exact non-empty invite-code factory.

        Returns:
            None: The service is ready without provider I/O.

        Notes:
            The provider store remains the only grant and invite authority. Session
            state is intentionally ephemeral and creates no second persistence path.
        """
        if not isinstance(secret, str) or not secret:
            raise ValueError("secret must be a non-empty string")
        if not isinstance(cookie_name, str) or not cookie_name.strip():
            raise ValueError("cookie_name must be a non-empty string")
        if cookie_samesite not in {"lax", "strict", "none"}:
            raise ValueError("cookie_samesite must be exactly 'lax', 'strict', or 'none'")
        _positive_int("session_ttl_seconds", session_ttl_seconds)
        _positive_int("grant_ttl_seconds", grant_ttl_seconds)
        self.store = store
        self.secret = secret
        self.cookie_name = cookie_name
        self.cookie_secure = cookie_secure
        self.cookie_samesite = cookie_samesite
        self.session_ttl_seconds = session_ttl_seconds
        self.grant_ttl_seconds = grant_ttl_seconds
        self.public_demo_fallback_enabled = public_demo_fallback_enabled
        self._clock = clock or (lambda: datetime.now(UTC))
        _utc(self._clock())
        self._guest_id_factory = guest_id_factory or (lambda: secrets.token_urlsafe(10))
        self._session_id_factory = session_id_factory or (lambda: secrets.token_urlsafe(24))
        self._invite_code_factory = invite_code_factory or _random_invite_code
        self._sessions: dict[str, AuthSession] = {}

    async def create_demo_session(
        self,
        *,
        grant: DemoGrant,
        client_id: str | None = None,
    ) -> AuthSession:
        """Persist one valid grant and create one ephemeral demo session.

        Intro:
            The grant commits through canonical provider persistence before a unique
            in-memory session is made visible to the caller.

        Examples:
            Create a demo session:
                ```python
                session = await authn.create_demo_session(grant=grant)
                ```

            Retain deprecated client compatibility metadata:
                ```python
                session = await authn.create_demo_session(
                    grant=grant,
                    client_id="browser-1",
                )
                ```

        Args:
            grant: Complete non-revoked, unexpired demo grant.
            client_id: Optional deprecated session-only client compatibility metadata.

        Returns:
            AuthSession: New ephemeral demo session after provider grant persistence.

        Notes:
            Deprecated client identity never enters provider scope, grant identity,
            invite identity, or authorization lookup.
        """
        self._require_valid_grant(grant)
        persisted = await self.store.save_grant(grant)
        return self._create_session(grant=persisted, client_id=client_id)

    def get_session(self, session_id: str | None) -> AuthSession | None:
        """Read one unexpired process-local Auth session.

        Intro:
            The session cache is checked without provider I/O and expired state is
            removed before returning.

        Examples:
            Read a session:
                ```python
                session = authn.get_session("session-1")
                ```

            Read an absent session:
                ```python
                assert authn.get_session(None) is None
                ```

        Args:
            session_id: Optional exact session identity.

        Returns:
            AuthSession | None: Unexpired session or `None` when absent or expired.

        Notes:
            Grant validity is deliberately checked by `resolve`, not by this local
            cache accessor.
        """
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.expires_at is not None and session.expires_at <= self._now():
            self._sessions.pop(session_id, None)
            return None
        return session

    def delete_session(self, session_id: str | None) -> None:
        """Delete one exact process-local Auth session idempotently.

        Intro:
            Logout and invalid-grant rejection remove only the presented ephemeral
            session and perform no provider mutation.

        Examples:
            Delete a session:
                ```python
                authn.delete_session("session-1")
                ```

            Delete an absent cookie value:
                ```python
                authn.delete_session(None)
                ```

        Args:
            session_id: Optional exact session identity.

        Returns:
            None: The session is absent after the call.

        Notes:
            Grants and invites remain unchanged in canonical persistence.
        """
        if session_id:
            self._sessions.pop(session_id, None)

    async def get_grant(self, grant_id: str | None) -> DemoGrant | None:
        """Read one currently valid grant from canonical provider authority.

        Intro:
            Every call reads provider state and filters revoked or domain-expired
            grants without consulting a service-owned grant cache.

        Examples:
            Read a valid grant:
                ```python
                grant = await authn.get_grant("grant-1")
                ```

            Read an absent grant:
                ```python
                assert await authn.get_grant(None) is None
                ```

        Args:
            grant_id: Optional exact grant identity.

        Returns:
            DemoGrant | None: Valid provider grant or `None` when unavailable.

        Notes:
            Revocation remains provider state and can invalidate an existing session
            on its next request.
        """
        if not grant_id:
            return None
        grant = await self.store.get_grant(grant_id)
        if grant is None or grant.revoked:
            return None
        if grant.expires_at is not None and grant.expires_at <= self._now():
            return None
        return grant

    async def create_invite_code(
        self,
        grant: DemoGrant,
        *,
        max_uses: int | None = None,
        expires_in_seconds: int | None = None,
        code: str | None = None,
    ) -> InviteCode:
        """Persist a grant and create one unique provider-authoritative invite.

        Intro:
            Default expiry is authored before the grant commit, and invite uniqueness
            is enforced by the canonical store's revision-zero create operation.

        Examples:
            Create a generated invite:
                ```python
                invite = await authn.create_invite_code(grant)
                ```

            Create a bounded custom invite:
                ```python
                invite = await authn.create_invite_code(
                    grant,
                    max_uses=3,
                    expires_in_seconds=3600,
                    code="DEMO-TEAM",
                )
                ```

        Args:
            grant: Complete non-revoked, unexpired demo grant.
            max_uses: Optional positive maximum successful claims.
            expires_in_seconds: Optional positive invite lifetime in seconds.
            code: Optional exact unique invite code.

        Returns:
            InviteCode: Provider-committed invite at its initial revision.

        Notes:
            Duplicate codes raise `ValueError`; no synthetic invite index or
            in-memory invite authority is created.
        """
        if max_uses is not None:
            _positive_int("max_uses", max_uses)
        if expires_in_seconds is not None:
            _positive_int("expires_in_seconds", expires_in_seconds)
        now = self._now()
        if grant.expires_at is None:
            grant = grant.model_copy(
                update={"expires_at": now + timedelta(seconds=self.grant_ttl_seconds)}
            )
        self._require_valid_grant(grant)
        await self.store.save_grant(grant)
        invite_code = _identity("code", code or self._invite_code_factory())
        invite = InviteCode(
            code=invite_code,
            grant_id=grant.grant_id,
            max_uses=max_uses,
            expires_at=(
                now + timedelta(seconds=expires_in_seconds)
                if expires_in_seconds is not None
                else grant.expires_at
            ),
        )
        try:
            return await self.store.create_invite(invite)
        except StorageConflictError as exc:
            raise ValueError(f"Invite code already exists: {invite_code}") from exc

    async def redeem_invite_code(
        self,
        code: str,
        *,
        client_id: str | None = None,
    ) -> AuthSession:
        """Atomically claim one invite use and create one ephemeral session.

        Intro:
            Provider CAS validates invite status, usage, expiry, and its current grant
            before any process-local session is created.

        Examples:
            Redeem an invite:
                ```python
                session = await authn.redeem_invite_code("DEMO-TEAM")
                ```

            Retain deprecated client compatibility metadata:
                ```python
                session = await authn.redeem_invite_code(
                    "DEMO-TEAM",
                    client_id="browser-1",
                )
                ```

        Args:
            code: Exact invite code to claim.
            client_id: Optional deprecated session-only client compatibility metadata.

        Returns:
            AuthSession: New session after one provider-authoritative successful claim.

        Notes:
            Failed claims create no session and never retry through another store.
        """
        _invite, grant = await self.store.claim_invite(_identity("code", code))
        return self._create_session(grant=grant, client_id=client_id)

    async def list_invite_codes(self) -> list[InviteCode]:
        """List provider-authoritative invites through the store's bounded scan.

        Intro:
            The result is detached from provider records and requires no synthetic
            index or process-local invite cache.

        Examples:
            List invites:
                ```python
                invites = await authn.list_invite_codes()
                ```

            Select active invites:
                ```python
                active = [item for item in await authn.list_invite_codes() if item.active]
                ```

        Args:
            None: The service is already bound to one owner-scoped store.

        Returns:
            list[InviteCode]: Bounded key-ordered provider invite values.

        Notes:
            Provider expiry visibility applies before results are returned.
        """
        return await self.store.list_invites()

    async def deactivate_invite_code(self, code: str) -> InviteCode:
        """Deactivate one existing invite through canonical replacement.

        Intro:
            The current provider value is required before one revisioned replacement
            preserves every field except active status.

        Examples:
            Deactivate an invite:
                ```python
                invite = await authn.deactivate_invite_code("DEMO-TEAM")
                ```

            Confirm the result:
                ```python
                assert not (await authn.deactivate_invite_code("DEMO-OLD")).active
                ```

        Args:
            code: Exact invite code.

        Returns:
            InviteCode: Provider-committed inactive invite.

        Notes:
            Missing invites raise `ValueError`; no cache value is substituted.
        """
        invite = await self.store.get_invite(_identity("code", code))
        if invite is None:
            raise ValueError(f"Invite code not found: {code}")
        return await self.store.save_invite(invite.model_copy(update={"active": False}))

    async def delete_invite_code(self, code: str) -> None:
        """Delete one provider-authoritative invite idempotently.

        Intro:
            The canonical store performs exact revision-CAS deletion and no companion
            index maintenance is necessary.

        Examples:
            Delete an invite:
                ```python
                await authn.delete_invite_code("DEMO-TEAM")
                ```

            Repeat deletion safely:
                ```python
                await authn.delete_invite_code("DEMO-TEAM")
                ```

        Args:
            code: Exact invite code.

        Returns:
            None: The provider invite is absent after the call.

        Notes:
            Missing invites are an idempotent success.
        """
        await self.store.delete_invite(_identity("code", code))

    async def update_invite_code(
        self,
        code: str,
        updates: Mapping[str, Any],
    ) -> InviteCode:
        """Replace allowed mutable fields on one provider invite.

        Intro:
            Exact field validation precedes one canonical revisioned replacement of
            the current provider value.

        Examples:
            Change an invite limit:
                ```python
                invite = await authn.update_invite_code(
                    "DEMO-TEAM", {"max_uses": 10}
                )
                ```

            Deactivate through the general update API:
                ```python
                invite = await authn.update_invite_code(
                    "DEMO-TEAM", {"active": False}
                )
                ```

        Args:
            code: Exact invite code.
            updates: Non-empty mapping of allowed mutable Invite fields.

        Returns:
            InviteCode: Provider-committed updated invite.

        Notes:
            Invite identity, grant identity, and usage count cannot be rewritten by
            this compatibility method.
        """
        invite = await self.store.get_invite(_identity("code", code))
        if invite is None:
            raise ValueError(f"Invite code not found: {code}")
        normalized = _updates(updates, allowed={"max_uses", "expires_at", "active"})
        if "max_uses" in normalized and normalized["max_uses"] is not None:
            _positive_int("max_uses", normalized["max_uses"])
        updated = InviteCode.model_validate({**invite.model_dump(), **normalized})
        return await self.store.save_invite(updated)

    async def list_grants(self) -> list[DemoGrant]:
        """List provider-authoritative grants through the store's bounded scan.

        Intro:
            Administrative enumeration returns detached records without establishing
            an Authn-service grant cache.

        Examples:
            List grants:
                ```python
                grants = await authn.list_grants()
                ```

            Select revoked grants:
                ```python
                revoked = [item for item in await authn.list_grants() if item.revoked]
                ```

        Args:
            None: The service is already bound to one owner-scoped store.

        Returns:
            list[DemoGrant]: Bounded key-ordered provider grant values.

        Notes:
            Revoked grants remain visible to administration but not to authentication.
        """
        return await self.store.list_grants()

    async def revoke_grant(self, grant_id: str) -> DemoGrant:
        """Revoke one existing provider grant through canonical replacement.

        Intro:
            The next request for any session referencing the committed grant fails
            closed and deletes that exact process-local session.

        Examples:
            Revoke a grant:
                ```python
                grant = await authn.revoke_grant("grant-1")
                ```

            Confirm revocation:
                ```python
                assert (await authn.revoke_grant("grant-2")).revoked
                ```

        Args:
            grant_id: Exact stable grant identity.

        Returns:
            DemoGrant: Provider-committed revoked grant.

        Notes:
            Session cleanup is lazy and exact on each session's next resolution.
        """
        grant = await self.store.get_grant(_identity("grant_id", grant_id))
        if grant is None:
            raise ValueError(f"Grant not found: {grant_id}")
        return await self.store.save_grant(grant.model_copy(update={"revoked": True}))

    async def delete_grant(self, grant_id: str) -> None:
        """Delete one provider-authoritative grant idempotently.

        Intro:
            Exact revision-CAS deletion makes every referencing session invalid on its
            next provider-backed resolution.

        Examples:
            Delete a grant:
                ```python
                await authn.delete_grant("grant-1")
                ```

            Repeat deletion safely:
                ```python
                await authn.delete_grant("grant-1")
                ```

        Args:
            grant_id: Exact stable grant identity.

        Returns:
            None: The provider grant is absent after the call.

        Notes:
            Invites are not silently cascaded; later claims fail against the missing
            grant through canonical validation.
        """
        await self.store.delete_grant(_identity("grant_id", grant_id))

    async def update_grant(
        self,
        grant_id: str,
        updates: Mapping[str, Any],
    ) -> DemoGrant:
        """Replace allowed mutable fields on one provider grant.

        Intro:
            Exact field validation precedes one canonical revisioned replacement of
            the current provider value.

        Examples:
            Change a client label:
                ```python
                grant = await authn.update_grant(
                    "grant-1", {"client_label": "Research"}
                )
                ```

            Change an Agent allowlist:
                ```python
                grant = await authn.update_grant(
                    "grant-1", {"allowed_agents": ["agent-1"]}
                )
                ```

        Args:
            grant_id: Exact stable grant identity.
            updates: Non-empty mapping of allowed mutable Grant fields.

        Returns:
            DemoGrant: Provider-committed updated grant.

        Notes:
            Grant identity and organization identity cannot be rewritten. App
            allowlists remain explicitly deprecated compatibility metadata in storage.
        """
        grant = await self.store.get_grant(_identity("grant_id", grant_id))
        if grant is None:
            raise ValueError(f"Grant not found: {grant_id}")
        normalized = _updates(
            updates,
            allowed={
                "allowed_apps",
                "allowed_agents",
                "client_label",
                "revoked",
                "read_only",
                "expires_at",
            },
        )
        updated = DemoGrant.model_validate({**grant.model_dump(), **normalized})
        if updated.expires_at is not None and updated.expires_at <= self._now():
            raise ValueError("Grant expiry must be in the future")
        return await self.store.save_grant(updated)

    async def resolve(
        self,
        *,
        deploy_mode: str,
        session_id: str | None,
        client_id: str | None,
        x_user_id: str | None,
        x_org_id: str | None,
        roles: list[str] | None = None,
        x_mode: str | None = None,
    ) -> ResolvedAuth:
        """Resolve one request with provider-authoritative demo grant validation.

        Intro:
            An unexpired presented demo session resolves its grant from canonical
            storage on every request before any identity is authored.

        Examples:
            Resolve a valid demo session:
                ```python
                resolved = await authn.resolve(
                    deploy_mode="demo",
                    session_id=session.session_id,
                    client_id=None,
                    x_user_id=None,
                    x_org_id=None,
                )
                ```

            Resolve trusted cloud headers without a session:
                ```python
                resolved = await authn.resolve(
                    deploy_mode="cloud",
                    session_id=None,
                    client_id=None,
                    x_user_id="user-1",
                    x_org_id="org-1",
                )
                ```

        Args:
            deploy_mode: Exact configured deployment mode.
            session_id: Optional presented demo-session cookie identity.
            client_id: Optional deprecated client compatibility identity.
            x_user_id: Optional trusted cloud-proxy user identity.
            x_org_id: Optional trusted cloud-proxy organization identity.
            roles: Optional trusted cloud-proxy roles.
            x_mode: Optional explicit request mode compatibility value.

        Returns:
            ResolvedAuth: Exact resolved request authentication state.

        Notes:
            Missing, expired, or revoked session grants delete the exact session and
            raise `AuthenticationRejected`. The same request cannot fall through to
            cloud headers, public demo, or local mode.
        """
        effective_roles = list(roles or [])
        session = self.get_session(session_id)
        if session is not None:
            grant = await self.get_grant(session.grant_id)
            if grant is None:
                self.delete_session(session.session_id)
                raise AuthenticationRejected("Demo session grant is no longer valid")
            return ResolvedAuth(
                mode="demo_guest",
                auth_source="demo_guest_session",
                session=session,
                client_id=client_id or session.client_id,
                grant=grant,
                roles=list(session.roles),
                user_id=session.user_id,
                org_id=session.org_id,
            )
        if x_user_id or x_org_id:
            return ResolvedAuth(
                mode="cloud_proxy",
                auth_source="cloud_proxy_headers",
                client_id=client_id,
                roles=effective_roles,
                user_id=x_user_id,
                org_id=x_org_id,
            )
        if (
            client_id
            and self.public_demo_fallback_enabled
            and (deploy_mode == "demo" or x_mode == "demo")
        ):
            return ResolvedAuth(
                mode="demo_guest",
                auth_source="public_demo_client_id",
                client_id=client_id,
                roles=["demo"],
                user_id=f"demo:{client_id}",
                org_id="demo",
            )
        return ResolvedAuth(
            mode="local",
            auth_source="local_default",
            client_id=client_id,
            roles=["dev"],
            user_id="local",
            org_id="local",
        )

    def _create_session(self, *, grant: DemoGrant, client_id: str | None) -> AuthSession:
        now = self._now()
        guest_id = _identity("guest_id", self._guest_id_factory())
        session_id = _identity("session_id", self._session_id_factory())
        if session_id in self._sessions:
            raise ValueError(f"Session identity already exists: {session_id}")
        session = AuthSession(
            session_id=session_id,
            mode="demo_guest",
            subject_id=f"demo_guest:{grant.grant_id}:{guest_id}",
            user_id=f"demo_guest:{grant.grant_id}:{guest_id}",
            org_id=grant.org_id,
            roles=["demo"],
            grant_id=grant.grant_id,
            client_id=client_id,
            expires_at=now + timedelta(seconds=self.session_ttl_seconds),
        )
        self._sessions[session.session_id] = session
        return session

    def _require_valid_grant(self, grant: DemoGrant) -> None:
        if grant.revoked:
            raise ValueError("Grant must not be revoked")
        if grant.expires_at is not None and grant.expires_at <= self._now():
            raise ValueError("Grant expiry must be in the future")

    def _now(self) -> datetime:
        return _utc(self._clock())


def _random_invite_code() -> str:
    return f"DEMO-{secrets.token_urlsafe(6).upper().rstrip('=')}"


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _identity(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be an exact non-empty string")
    return value


def _updates(updates: Mapping[str, Any], *, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(updates, Mapping) or not updates:
        raise ValueError("updates must be a non-empty mapping")
    unknown = sorted(set(updates).difference(allowed))
    if unknown:
        raise ValueError(f"updates contain unsupported fields: {unknown}")
    return dict(updates)


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("clock must return a timezone-aware UTC datetime")
    return value


__all__ = ["CanonicalAuthnService"]
