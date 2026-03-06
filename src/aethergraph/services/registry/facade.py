from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.services.scope.scope import Scope
from aethergraph.services.scope.tenant import normalize_registry_tenant

from .registration_service import (
    DeletionResult,
    RegistrationResult,
    RegistrationService,
    ReplayReport,
    ValidationResult,
)
from .registry_key import Key
from .unified_registry import TenantIdentity, UnifiedRegistry


@dataclass
class RegistryFacade:
    """Scope-bound facade over UnifiedRegistry with tenant-aware defaults."""

    registry: UnifiedRegistry
    scope: Scope | None = None
    registration_service: RegistrationService | None = None

    def tenant(self) -> dict[str, str | None] | None:
        if self.scope is None:
            return None
        return normalize_registry_tenant(
            {
                "org_id": self.scope.org_id,
                "user_id": self.scope.user_id,
            }
        )

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

    async def register_by_file(
        self,
        path: str,
        *,
        app_config: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        tenant: TenantIdentity = None,
        persist: bool = True,
        strict: bool = True,
    ) -> RegistrationResult:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.register_by_file(
            path,
            app_config=app_config,
            agent_config=agent_config,
            tenant=self._effective_tenant(tenant),
            persist=persist,
            strict=strict,
        )

    async def register_by_artifact(
        self,
        artifact_id: str | None = None,
        uri: str | None = None,
        *,
        app_config: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        tenant: TenantIdentity = None,
        persist: bool = True,
        strict: bool = True,
    ) -> RegistrationResult:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.register_by_artifact(
            artifact_id=artifact_id,
            uri=uri,
            app_config=app_config,
            agent_config=agent_config,
            tenant=self._effective_tenant(tenant),
            persist=persist,
            strict=strict,
        )

    async def register_by_folder(
        self,
        folder: str,
        *,
        pattern: str = "*.py",
        recursive: bool = True,
        tenant: TenantIdentity = None,
        persist: bool = True,
        strict: bool = False,
    ) -> list[RegistrationResult]:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.register_by_folder(
            folder,
            pattern=pattern,
            recursive=recursive,
            tenant=self._effective_tenant(tenant),
            persist=persist,
            strict=strict,
        )

    def validate_graphify_source(
        self,
        source: str,
        *,
        filename: str | None = None,
        strict: bool = True,
    ) -> ValidationResult:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return self.registration_service.validate_graphify_source(
            source,
            filename=filename,
            strict=strict,
        )

    async def replay_registered_sources(
        self,
        *,
        tenant: TenantIdentity = None,
        strict: bool = False,
    ) -> ReplayReport:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.replay_registered_sources(
            tenant=self._effective_tenant(tenant),
            strict=strict,
        )

    async def delete_registered_app(
        self,
        *,
        app_id: str,
        tenant: TenantIdentity = None,
        keep_runtime_registration: bool = False,
    ) -> DeletionResult:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.delete_registered_app(
            app_id=app_id,
            tenant=self._effective_tenant(tenant),
            keep_runtime_registration=keep_runtime_registration,
        )

    async def delete_registered_agent(
        self,
        *,
        agent_id: str,
        tenant: TenantIdentity = None,
        keep_runtime_registration: bool = False,
    ) -> DeletionResult:
        if self.registration_service is None:
            raise RuntimeError("RegistryFacade.registration_service is not configured")
        return await self.registration_service.delete_registered_agent(
            agent_id=agent_id,
            tenant=self._effective_tenant(tenant),
            keep_runtime_registration=keep_runtime_registration,
        )
