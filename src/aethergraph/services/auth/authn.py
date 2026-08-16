"""Stable public authentication DTOs used by canonical Authn services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AuthSessionMode = Literal["local", "demo_guest", "cloud_proxy"]


class AuthenticationRejected(ValueError):
    """Reject one presented authentication state without selecting another mode."""


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
    max_uses: int | None = None
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


__all__ = [
    "AuthenticationRejected",
    "AuthSession",
    "AuthSessionMode",
    "DemoGrant",
    "InviteCode",
    "ResolvedAuth",
]
