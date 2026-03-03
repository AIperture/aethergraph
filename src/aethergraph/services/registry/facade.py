from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.services.scope.scope import Scope

from .registry_key import Key
from .unified_registry import TenantIdentity, UnifiedRegistry


@dataclass
class RegistryFacade:
    """Scope-bound facade over UnifiedRegistry with tenant-aware defaults."""

    registry: UnifiedRegistry
    scope: Scope | None = None

    def tenant(self) -> dict[str, str | None] | None:
        if self.scope is None:
            return None
        tenant = {
            "org_id": self.scope.org_id,
            "user_id": self.scope.user_id,
            "client_id": self.scope.client_id,
        }
        if not any(tenant.values()):
            return None
        return tenant

    def _effective_tenant(self, tenant: TenantIdentity = None) -> TenantIdentity:
        if tenant is not None:
            return tenant
        return self.tenant()

    def register(
        self,
        *,
        nspace: str,
        name: str,
        version: str,
        obj: Any,
        meta: dict[str, Any] | None = None,
        tenant: TenantIdentity = None,
    ) -> None:
        self.registry.register(
            nspace=nspace,
            name=name,
            version=version,
            obj=obj,
            meta=meta,
            tenant=self._effective_tenant(tenant),
        )

    def register_latest(
        self,
        *,
        nspace: str,
        name: str,
        obj: Any,
        version: str = "0.0.0",
        tenant: TenantIdentity = None,
    ) -> None:
        self.registry.register_latest(
            nspace=nspace,
            name=name,
            version=version,
            obj=obj,
            tenant=self._effective_tenant(tenant),
        )

    def alias(
        self,
        *,
        nspace: str,
        name: str,
        tag: str,
        to_version: str,
        tenant: TenantIdentity = None,
    ) -> None:
        self.registry.alias(
            nspace=nspace,
            name=name,
            tag=tag,
            to_version=to_version,
            tenant=self._effective_tenant(tenant),
        )

    def unregister(
        self,
        *,
        nspace: str,
        name: str,
        version: str | None = None,
        tenant: TenantIdentity = None,
    ) -> None:
        self.registry.unregister(
            nspace=nspace,
            name=name,
            version=version,
            tenant=self._effective_tenant(tenant),
        )

    def get(
        self,
        ref: str | Key,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.registry.get(
            ref,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def list(
        self,
        nspace: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, str]:
        return self.registry.list(
            nspace=nspace,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def get_meta(
        self,
        nspace: str,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, Any] | None:
        return self.registry.get_meta(
            nspace=nspace,
            name=name,
            version=version,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def get_tool(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.registry.get_tool(
            name,
            version,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def get_graph(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.registry.get_graph(
            name,
            version,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def get_graphfn(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.registry.get_graphfn(
            name,
            version,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def get_agent(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.registry.get_agent(
            name,
            version,
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def list_agents(
        self,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, str]:
        return self.registry.list_agents(
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )

    def list_apps(
        self,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, str]:
        return self.registry.list_apps(
            tenant=self._effective_tenant(tenant),
            include_global=include_global,
        )
