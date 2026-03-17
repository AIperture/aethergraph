from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import secrets
from typing import Literal

from pydantic import BaseModel, Field

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
        demo_token_ttl_seconds: int = 7 * 24 * 3600,
        public_demo_fallback_enabled: bool = True,
    ) -> None:
        self.secret = secret.encode("utf-8")
        self.cookie_name = cookie_name
        self.cookie_secure = cookie_secure
        self.cookie_samesite = cookie_samesite
        self.session_ttl_seconds = session_ttl_seconds
        self.demo_token_ttl_seconds = demo_token_ttl_seconds
        self.public_demo_fallback_enabled = public_demo_fallback_enabled
        self._sessions: dict[str, AuthSession] = {}
        self._grants: dict[str, DemoGrant] = {}

    def issue_demo_token(self, grant: DemoGrant) -> str:
        if grant.expires_at is None:
            grant = grant.model_copy(
                update={
                    "expires_at": datetime.now(UTC) + timedelta(seconds=self.demo_token_ttl_seconds)
                }
            )
        self._grants[grant.grant_id] = grant
        payload = base64.urlsafe_b64encode(
            json.dumps(grant.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")
        ).decode("utf-8")
        sig = hmac.new(self.secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def parse_demo_token(self, token: str) -> DemoGrant:
        try:
            payload, sig = token.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError("Malformed demo token") from exc
        expected = hmac.new(self.secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid demo token signature")
        try:
            raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
            grant = DemoGrant.model_validate(json.loads(raw.decode("utf-8")))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid demo token payload") from exc
        if grant.expires_at and grant.expires_at <= datetime.now(UTC):
            raise ValueError("Demo token expired")
        if grant.revoked:
            raise ValueError("Demo token revoked")
        self._grants[grant.grant_id] = grant
        return grant

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
        self._grants[grant.grant_id] = grant
        return session

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
        if grant is None:
            return None
        if grant.expires_at and grant.expires_at <= datetime.now(UTC):
            self._grants.pop(grant_id, None)
            return None
        if grant.revoked:
            return None
        return grant

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
