from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
import secrets
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aethergraph.storage.kv.sqlite_kv_sync import SQLiteKVSync

log = logging.getLogger(__name__)

AuthSessionMode = Literal["local", "demo_guest", "cloud_proxy"]


class DemoGrant(BaseModel):
    grant_id: str
    org_id: str
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_agents: list[str] = Field(default_factory=list)
    client_label: str | None = None
    revoked: bool = False
    read_only: bool = False
    expires_at: datetime | None = None


class InviteCode(BaseModel):
    code: str
    grant_id: str
    max_uses: int | None = None  # None = unlimited
    uses: int = 0
    expires_at: datetime | None = None
    active: bool = True


class AuthSession(BaseModel):
    session_id: str
    mode: AuthSessionMode
    subject_id: str
    user_id: str | None = None
    org_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    grant_id: str | None = None
    client_id: str | None = None
    expires_at: datetime | None = None


class ResolvedAuth(BaseModel):
    mode: AuthSessionMode
    auth_source: str
    session: AuthSession | None = None
    client_id: str | None = None
    grant: DemoGrant | None = None
    roles: list[str] = Field(default_factory=list)
    user_id: str | None = None
    org_id: str | None = None


class AuthnService:
    """Session and demo-grant resolver for HTTP and WebSocket requests."""

    def __init__(
        self,
        *,
        secret: str,
        cookie_name: str = "ag_auth_session",
        cookie_secure: bool = False,
        cookie_samesite: Literal["lax", "strict", "none"] = "lax",
        session_ttl_seconds: int = 24 * 3600,
        grant_ttl_seconds: int = 7 * 24 * 3600,
        public_demo_fallback_enabled: bool = True,
        grant_store: SQLiteKVSync | None = None,
        invite_store: SQLiteKVSync | None = None,
    ) -> None:
        self.secret = secret
        self.cookie_name = cookie_name
        self.cookie_secure = cookie_secure
        self.cookie_samesite = cookie_samesite
        self.session_ttl_seconds = session_ttl_seconds
        self.grant_ttl_seconds = grant_ttl_seconds
        self.public_demo_fallback_enabled = public_demo_fallback_enabled
        self._grant_store = grant_store
        self._invite_store = invite_store
        self._sessions: dict[str, AuthSession] = {}
        self._grants: dict[str, DemoGrant] = {}
        self._invite_codes: dict[str, InviteCode] = {}

    def create_demo_session(self, *, grant: DemoGrant, client_id: str | None = None) -> AuthSession:
        now = datetime.now(UTC)
        guest_id = secrets.token_urlsafe(10)
        session = AuthSession(
            session_id=secrets.token_urlsafe(24),
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
        self._persist_grant(grant)
        return session

    # ---- persistence helpers ------------------------------------------------

    def _persist_grant(self, grant: DemoGrant) -> None:
        self._grants[grant.grant_id] = grant
        if self._grant_store is not None:
            ttl_s = (
                int((grant.expires_at - datetime.now(UTC)).total_seconds())
                if grant.expires_at
                else None
            )
            self._grant_store.set(
                grant.grant_id,
                grant.model_dump(mode="json"),
                ttl_s=ttl_s if ttl_s and ttl_s > 0 else None,
            )

    def _persist_invite(self, invite: InviteCode) -> None:
        self._invite_codes[invite.code] = invite
        if self._invite_store is not None:
            ttl_s = (
                int((invite.expires_at - datetime.now(UTC)).total_seconds())
                if invite.expires_at
                else None
            )
            self._invite_store.set(
                invite.code,
                invite.model_dump(mode="json"),
                ttl_s=ttl_s if ttl_s and ttl_s > 0 else None,
            )
            # maintain index of all codes
            index: list[str] = self._invite_store.get("_index", default=[]) or []
            if invite.code not in index:
                index.append(invite.code)
                self._invite_store.set("_index", index)

    def load_persisted(self) -> None:
        """Hydrate in-memory caches from persistent stores on startup."""
        if self._invite_store is None:
            return
        index: list[str] = self._invite_store.get("_index", default=[]) or []
        loaded_invites = 0
        for code in index:
            raw = self._invite_store.get(code)
            if raw:
                invite = InviteCode.model_validate(raw)
                self._invite_codes[invite.code] = invite
                loaded_invites += 1
        if self._grant_store is not None:
            grant_ids = {inv.grant_id for inv in self._invite_codes.values()}
            for gid in grant_ids:
                raw = self._grant_store.get(gid)
                if raw:
                    grant = DemoGrant.model_validate(raw)
                    self._grants[grant.grant_id] = grant
        if loaded_invites:
            log.info(
                "Loaded %d persisted invite code(s) and %d grant(s)",
                loaded_invites,
                len(self._grants),
            )

    # ---- sessions (in-memory only) -----------------------------------------

    def get_session(self, session_id: str | None) -> AuthSession | None:
        if not session_id:
            return None
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        if sess.expires_at and sess.expires_at <= datetime.now(UTC):
            self._sessions.pop(session_id, None)
            return None
        return sess

    def delete_session(self, session_id: str | None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)

    def get_grant(self, grant_id: str | None) -> DemoGrant | None:
        if not grant_id:
            return None
        grant = self._grants.get(grant_id)
        if grant is None and self._grant_store is not None:
            raw = self._grant_store.get(grant_id)
            if raw:
                grant = DemoGrant.model_validate(raw)
                self._grants[grant_id] = grant
        if grant is None:
            return None
        if grant.expires_at and grant.expires_at <= datetime.now(UTC):
            self._grants.pop(grant_id, None)
            if self._grant_store is not None:
                self._grant_store.delete(grant_id)
            return None
        if grant.revoked:
            return None
        return grant

    def create_invite_code(
        self,
        grant: DemoGrant,
        *,
        max_uses: int | None = None,
        expires_in_seconds: int | None = None,
        code: str | None = None,
    ) -> InviteCode:
        if grant.expires_at is None:
            grant = grant.model_copy(
                update={"expires_at": datetime.now(UTC) + timedelta(seconds=self.grant_ttl_seconds)}
            )
        self._persist_grant(grant)
        if code is not None:
            if code in self._invite_codes:
                raise ValueError(f"Invite code already exists: {code}")
        else:
            code_suffix = secrets.token_urlsafe(6).upper().rstrip("=")
            code = f"DEMO-{code_suffix}"
        invite = InviteCode(
            code=code,
            grant_id=grant.grant_id,
            max_uses=max_uses,
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=expires_in_seconds)
                if expires_in_seconds
                else grant.expires_at
            ),
        )
        self._persist_invite(invite)
        return invite

    def redeem_invite_code(self, code: str, *, client_id: str | None = None) -> AuthSession:
        invite = self._invite_codes.get(code)
        if invite is None and self._invite_store is not None:
            raw = self._invite_store.get(code)
            if raw:
                invite = InviteCode.model_validate(raw)
                self._invite_codes[code] = invite
        if invite is None:
            raise ValueError("Invalid invite code")
        if not invite.active:
            raise ValueError("Invite code is deactivated")
        if invite.expires_at and invite.expires_at <= datetime.now(UTC):
            raise ValueError("Invite code expired")
        if invite.max_uses is not None and invite.uses >= invite.max_uses:
            raise ValueError("Invite code has reached its usage limit")
        grant = self.get_grant(invite.grant_id)
        if grant is None:
            raise ValueError("Invite code grant is no longer valid")
        invite.uses += 1
        self._persist_invite(invite)
        return self.create_demo_session(grant=grant, client_id=client_id)

    def list_invite_codes(self) -> list[InviteCode]:
        return list(self._invite_codes.values())

    def resolve(
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
        roles = roles or []
        sess = self.get_session(session_id)
        if sess is not None:
            grant = self.get_grant(sess.grant_id)
            return ResolvedAuth(
                mode="demo_guest",
                auth_source="demo_guest_session",
                session=sess,
                client_id=client_id or sess.client_id,
                grant=grant,
                roles=sess.roles,
                user_id=sess.user_id,
                org_id=sess.org_id,
            )

        if x_user_id or x_org_id:
            return ResolvedAuth(
                mode="cloud_proxy",
                auth_source="cloud_proxy_headers",
                client_id=client_id,
                roles=roles,
                user_id=x_user_id,
                org_id=x_org_id,
            )

        if (
            client_id
            and self.public_demo_fallback_enabled
            and (deploy_mode == "demo" or x_mode == "demo")
        ):
            user_id = f"demo:{client_id}"
            return ResolvedAuth(
                mode="demo_guest",
                auth_source="public_demo_client_id",
                client_id=client_id,
                roles=["demo"],
                user_id=user_id,
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
