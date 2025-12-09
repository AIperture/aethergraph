from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from aethergraph.api.v1.deps import RequestIdentity

AuthZServiceScope = Literal["runs", "artifacts", "graphs", "admin", "system"]
AuthZServiceAction = Literal["read", "write", "execute", "delete", "admin"]


class AuthZService(Protocol):
    async def authorize(
        self,
        *,
        identity: RequestIdentity,
        scope: AuthZServiceScope,
        action: AuthZServiceAction,
    ) -> None:
        """Authorize the given identity to perform the action within the scope.

        Raises HTTPException with status 403 if not authorized.
        """
        ...


@dataclass
class AllowAllAuthz(AuthZService):
    """
    Default OSS-safe behavior: everything is allowed.
    Useful for local/demo, and as a safe fallback if authz isn't configured.
    """

    async def authorize(
        self,
        *,
        identity: RequestIdentity,
        scope: AuthZServiceScope,
        action: AuthZServiceAction,
    ) -> None:
        """Always allow."""
        return


@dataclass
class BasicAuthz(AuthZService):
    """
    Minimal policy based on mode and roles:
      - local: allow everything
      - demo: allow normal operations, block admin/system stuff
      - cloud: allow everything except admin unless role "admin" is present
    """

    allow_local_admin: bool = False

    async def authorize(
        self,
        *,
        identity: RequestIdentity,
        scope: AuthZServiceScope,
        action: AuthZServiceAction,
    ) -> None:
        # Local dev: basically god-mode
        if identity.mode == "local":
            if self.allow_local_admin:
                return
            if scope != "admin":
                return

        # Demo mode: no admin / system endpoints
        if identity.mode == "demo":
            if scope in ("admin", "system"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin/system operations are disabled in demo mode.",
                )
            return

        # Cloud mode: basic, role-based admin
        if identity.mode == "cloud":
            if scope in ("admin", "system"):
                if "admin" not in identity.roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin privileges required.",
                    )
                return

            # Non-admin scopes: for now, allow everything.
            # Later you can restrict per-org / per-resource here.
            return

        # Fallback: if someone invents a new mode and forgets to handle it
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unknown auth mode: {identity.mode}",
        )
