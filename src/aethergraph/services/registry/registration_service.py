from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha1, sha256
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.core.graph.graphify_validation import (
    ValidationResult,
    validate_graph_source,
)
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.services.registry.unified_registry import TenantIdentity, UnifiedRegistry
from aethergraph.services.scope.tenant import normalize_registry_tenant
from aethergraph.storage.registry.registration_docstore import RegistrationManifestStore


@dataclass
class RegistrationResult:
    success: bool
    source_kind: str
    source_ref: str
    filename: str | None = None
    sha256: str | None = None
    graph_name: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    version: str | None = None
    entry_id: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class ReplayReport:
    total: int
    loaded: int
    failed: int
    results: list[RegistrationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DeletionResult:
    success: bool
    subject: str
    subject_id: str
    tenant: dict[str, str | None] | None = None
    unregistered: bool = False
    removed_entries: int = 0
    errors: list[str] = field(default_factory=list)


def _normalize_tenant(tenant: TenantIdentity) -> dict[str, str | None] | None:
    return normalize_registry_tenant(tenant)


class RegistrationService:
    """
    Source-based graph hotload + registration + replay persistence.
    """

    def __init__(
        self,
        *,
        registry: UnifiedRegistry,
        manifest_store: RegistrationManifestStore | None = None,
        artifact_store: AsyncArtifactStore | None = None,
        artifact_index: AsyncArtifactIndex | None = None,
    ) -> None:
        self.registry = registry
        self.manifest_store = manifest_store
        self.artifact_store = artifact_store
        self.artifact_index = artifact_index

    def validate_graphify_source(
        self,
        source: str,
        *,
        filename: str | None = None,
        strict: bool = True,
    ) -> ValidationResult:
        return validate_graph_source(source, filename=filename, strict=strict)

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
        source_path = Path(path).resolve()
        source = source_path.read_text(encoding="utf-8")
        return await self._register_source(
            source=source,
            source_kind="file",
            source_ref=str(source_path),
            filename=source_path.name,
            app_config=app_config,
            agent_config=agent_config,
            tenant=tenant,
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
        if not artifact_id and not uri:
            return RegistrationResult(
                success=False,
                source_kind="artifact",
                source_ref="",
                errors=["artifact_id or uri is required"],
            )

        try:
            source_uri = await self._resolve_artifact_uri(artifact_id=artifact_id, uri=uri)
            source = await self.artifact_store.load_text(source_uri)
        except Exception as e:
            if strict:
                raise
            return RegistrationResult(
                success=False,
                source_kind="artifact",
                source_ref=artifact_id or (uri or ""),
                errors=[f"artifact_load_failed: {e!r}"],
            )
        source_ref = f"artifact_id:{artifact_id}" if artifact_id else f"uri:{uri}"
        return await self._register_source(
            source=source,
            source_kind="artifact",
            source_ref=source_ref,
            filename=(Path(uri).name if uri else None),
            app_config=app_config,
            agent_config=agent_config,
            tenant=tenant,
            persist=persist,
            strict=strict,
        )

    async def _resolve_artifact_uri(
        self,
        *,
        artifact_id: str | None,
        uri: str | None,
    ) -> str:
        if uri:
            if self.artifact_store is None:
                raise RuntimeError("Artifact store is not configured for registration service")
            return uri
        if not artifact_id:
            raise ValueError("artifact_id or uri is required")
        if self.artifact_store is None or self.artifact_index is None:
            raise RuntimeError(
                "Artifact store/index are not configured for artifact-based registration"
            )
        art = await self.artifact_index.get(artifact_id)
        if art is None or not art.uri:
            raise FileNotFoundError(f"Artifact {artifact_id} not found or missing uri")
        return art.uri

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
        root = Path(folder).resolve()
        files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
        out: list[RegistrationResult] = []
        for p in files:
            if not p.is_file():
                continue
            result = await self.register_by_file(
                str(p),
                tenant=tenant,
                persist=persist,
                strict=False,
            )
            out.append(result)
            if strict and not result.success:
                raise RuntimeError(f"register_by_folder failed for {p}: {result.errors}")
        return out

    async def replay_registered_sources(
        self,
        *,
        tenant: TenantIdentity = None,
        strict: bool = False,
    ) -> ReplayReport:
        if self.manifest_store is None:
            return ReplayReport(
                total=0,
                loaded=0,
                failed=0,
                errors=["Manifest store is not configured"],
            )

        entries = await self.manifest_store.list_entries(
            tenant=tenant, include_global=True, active_only=True
        )
        results: list[RegistrationResult] = []
        errors: list[str] = []
        loaded = 0
        failed = 0

        for entry in entries:
            source_kind = str(entry.get("source_kind") or "")
            source_ref = str(entry.get("source_ref") or "")
            entry_tenant = entry.get("tenant") if isinstance(entry.get("tenant"), Mapping) else None
            replay_app_config = self._replay_app_config(entry)
            replay_agent_config = self._replay_agent_config(entry)
            try:
                if source_kind == "file":
                    rr = await self.register_by_file(
                        source_ref,
                        app_config=replay_app_config,
                        agent_config=replay_agent_config,
                        tenant=entry_tenant,
                        persist=False,
                        strict=False,
                    )
                elif source_kind == "artifact":
                    replay_artifact_id: str | None = None
                    replay_uri: str | None = None
                    if source_ref.startswith("artifact_id:"):
                        replay_artifact_id = source_ref.split(":", 1)[1] or None
                    elif source_ref.startswith("uri:"):
                        replay_uri = source_ref.split(":", 1)[1] or None
                    else:
                        replay_artifact_id = source_ref or None
                    rr = await self.register_by_artifact(
                        artifact_id=replay_artifact_id,
                        uri=replay_uri,
                        app_config=replay_app_config,
                        agent_config=replay_agent_config,
                        tenant=entry_tenant,
                        persist=False,
                        strict=False,
                    )
                else:
                    rr = RegistrationResult(
                        success=False,
                        source_kind=source_kind or "unknown",
                        source_ref=source_ref,
                        errors=[f"Unsupported manifest source_kind: {source_kind!r}"],
                    )
            except Exception as e:
                rr = RegistrationResult(
                    success=False,
                    source_kind=source_kind or "unknown",
                    source_ref=source_ref,
                    errors=[repr(e)],
                )

            rr.entry_id = str(entry.get("entry_id") or "")
            results.append(rr)
            if rr.success:
                loaded += 1
                if rr.entry_id:
                    await self.manifest_store.set_last_error(entry_id=rr.entry_id, last_error=None)
            else:
                failed += 1
                msg = f"{source_kind}:{source_ref}: {rr.errors}"
                errors.append(msg)
                if rr.entry_id:
                    await self.manifest_store.set_last_error(
                        entry_id=rr.entry_id,
                        last_error="; ".join(rr.errors),
                    )
                if strict:
                    raise RuntimeError(msg)

        return ReplayReport(
            total=len(entries),
            loaded=loaded,
            failed=failed,
            results=results,
            errors=errors,
        )

    @staticmethod
    def _replay_app_config(entry: Mapping[str, Any]) -> dict[str, Any] | None:
        cfg = entry.get("app_config")
        if isinstance(cfg, Mapping):
            return dict(cfg)
        app_id = entry.get("app_id")
        if not app_id:
            return None
        graph_name = entry.get("graph_name")
        version = entry.get("version") or "0.1.0"
        return {
            "id": str(app_id),
            "version": str(version),
            "flow_id": str(graph_name or app_id),
            "graph_name": str(graph_name or app_id),
        }

    @staticmethod
    def _replay_agent_config(entry: Mapping[str, Any]) -> dict[str, Any] | None:
        cfg = entry.get("agent_config")
        if isinstance(cfg, Mapping):
            return dict(cfg)
        agent_id = entry.get("agent_id")
        if not agent_id:
            return None
        graph_name = entry.get("graph_name")
        version = entry.get("version") or "0.1.0"
        return {
            "id": str(agent_id),
            "version": str(version),
            "graph_name": str(graph_name or agent_id),
        }

    async def _register_source(
        self,
        *,
        source: str,
        source_kind: str,
        source_ref: str,
        filename: str | None,
        app_config: dict[str, Any] | None,
        agent_config: dict[str, Any] | None,
        tenant: TenantIdentity,
        persist: bool,
        strict: bool,
    ) -> RegistrationResult:
        vr = self.validate_graphify_source(source, filename=filename, strict=strict)
        if not vr.ok:
            errors = [f"{i.code}: {i.message}" for i in vr.issues]
            if strict:
                raise RuntimeError(f"Validation failed for {source_ref}: {errors}")
            return RegistrationResult(
                success=False,
                source_kind=source_kind,
                source_ref=source_ref,
                filename=filename,
                errors=errors,
            )

        source_sha = sha256(source.encode("utf-8")).hexdigest()
        module_key = sha1(f"{source_kind}:{source_ref}:{source_sha}".encode()).hexdigest()[:12]
        module_name = f"_ag_hotload_{module_key}_{sha1(str(id(self.registry)).encode('utf-8')).hexdigest()[:8]}"
        module = types.ModuleType(module_name)
        module.__file__ = f"<hotload:{module_name}>"
        module.__dict__["__aethergraph_source__"] = source
        module.__dict__["__aethergraph_source_name__"] = filename or source_ref or module_name
        sys.modules[module_name] = module
        # Force decorators in source (e.g. @graphify/@graph_fn) to bind to this service registry.
        with use_services(SimpleNamespace(registry=self.registry)):
            exec(compile(source, filename or source_ref or module_name, "exec"), module.__dict__)

        tenant_norm = _normalize_tenant(tenant)
        graph_name = None
        graph_obj: Any = None
        persisted_app_config: dict[str, Any] | None = None
        persisted_agent_config: dict[str, Any] | None = None
        for candidate in vr.graph_names:
            try:
                graph_obj = self.registry.get_graph(
                    candidate,
                    version=None,
                    tenant=tenant_norm,
                    include_global=True,
                )
                graph_name = candidate
                break
            except Exception:
                continue
        if graph_obj is None:
            for candidate in vr.graphfn_names:
                try:
                    graph_obj = self.registry.get_graphfn(
                        candidate,
                        version=None,
                        tenant=tenant_norm,
                        include_global=True,
                    )
                    graph_name = candidate
                    break
                except Exception:
                    continue

        if graph_obj is None:
            msg = "Source executed but no registered graph/graphfn could be resolved"
            if strict:
                raise RuntimeError(msg)
            return RegistrationResult(
                success=False,
                source_kind=source_kind,
                source_ref=source_ref,
                filename=filename,
                sha256=source_sha,
                errors=[msg],
            )

        version = "0.1.0"
        app_id: str | None = None
        if app_config:
            merged = dict(app_config)
            app_id = str(merged.get("id") or graph_name)
            merged.setdefault("id", app_id)
            merged.setdefault("flow_id", graph_name)
            merged.setdefault("graph_name", graph_name)
            version = str(merged.get("version") or version)
            merged.setdefault("graph_id", graph_name)
            merged.setdefault(
                "backing",
                {
                    "type": "graphfn",
                    "name": graph_name,
                    "version": version,
                },
            )
            persisted_app_config = dict(merged)
            self.registry.register(
                nspace="app",
                name=app_id,
                version=version,
                obj=merged,
                meta=merged,
                tenant=tenant_norm,
            )

        agent_id: str | None = None
        if agent_config:
            merged_agent = dict(agent_config)
            agent_id = str(merged_agent.get("id") or graph_name)
            merged_agent.setdefault("id", agent_id)
            merged_agent.setdefault("graph_name", graph_name)
            agent_ver = str(merged_agent.get("version") or version)
            persisted_agent_config = dict(merged_agent)
            self.registry.register(
                nspace="agent",
                name=agent_id,
                version=agent_ver,
                obj=graph_obj,
                meta=merged_agent,
                tenant=tenant_norm,
            )

        entry_id: str | None = None
        if persist and self.manifest_store is not None:
            tenant_key = _normalize_tenant(tenant_norm)
            stable = sha1(
                f"{source_kind}|{source_ref}|{tenant_key or 'global'}".encode()
            ).hexdigest()
            row = await self.manifest_store.upsert_entry(
                {
                    "entry_id": stable,
                    "source_kind": source_kind,
                    "source_ref": source_ref,
                    "filename": filename,
                    "sha256": source_sha,
                    "graph_name": graph_name,
                    "app_id": app_id,
                    "agent_id": agent_id,
                    "version": version,
                    "tenant": tenant_key,
                    "app_config": persisted_app_config,
                    "agent_config": persisted_agent_config,
                    "last_error": None,
                }
            )
            entry_id = str(row["entry_id"])

        return RegistrationResult(
            success=True,
            source_kind=source_kind,
            source_ref=source_ref,
            filename=filename,
            sha256=source_sha,
            graph_name=graph_name,
            app_id=app_id,
            agent_id=agent_id,
            version=version,
            entry_id=entry_id,
        )

    async def delete_registered_app(
        self,
        *,
        app_id: str,
        tenant: TenantIdentity = None,
        keep_runtime_registration: bool = False,
    ) -> DeletionResult:
        return await self._delete_registered_subject(
            subject="app",
            subject_id=app_id,
            tenant=tenant,
            keep_runtime_registration=keep_runtime_registration,
        )

    async def delete_registered_agent(
        self,
        *,
        agent_id: str,
        tenant: TenantIdentity = None,
        keep_runtime_registration: bool = False,
    ) -> DeletionResult:
        return await self._delete_registered_subject(
            subject="agent",
            subject_id=agent_id,
            tenant=tenant,
            keep_runtime_registration=keep_runtime_registration,
        )

    async def _delete_registered_subject(
        self,
        *,
        subject: str,
        subject_id: str,
        tenant: TenantIdentity,
        keep_runtime_registration: bool,
    ) -> DeletionResult:
        tenant_norm = _normalize_tenant(tenant)
        removed_entries = 0
        errors: list[str] = []
        unregistered = False
        nspace = "app" if subject == "app" else "agent"

        if not keep_runtime_registration:
            try:
                self.registry.unregister(nspace=nspace, name=subject_id, tenant=tenant_norm)
                unregistered = True
            except Exception as e:
                errors.append(f"unregister_failed: {e!r}")

        if self.manifest_store is not None:
            try:
                if subject == "app":
                    removed_entries = await self.manifest_store.delete_entries_for_app(
                        app_id=subject_id,
                        tenant=tenant_norm,
                        include_global=False,
                    )
                else:
                    removed_entries = await self.manifest_store.delete_entries_for_agent(
                        agent_id=subject_id,
                        tenant=tenant_norm,
                        include_global=False,
                    )
            except Exception as e:
                errors.append(f"manifest_delete_failed: {e!r}")

        return DeletionResult(
            success=(len(errors) == 0),
            subject=subject,
            subject_id=subject_id,
            tenant=tenant_norm,
            unregistered=unregistered,
            removed_entries=removed_entries,
            errors=errors,
        )

    @staticmethod
    def to_dict(
        result: RegistrationResult | ValidationResult | ReplayReport | DeletionResult,
    ) -> dict[str, Any]:
        if isinstance(result, ValidationResult):
            return {
                "ok": result.ok,
                "issues": [asdict(i) for i in result.issues],
                "graph_names": list(result.graph_names),
                "graphfn_names": list(result.graphfn_names),
            }
        if isinstance(result, ReplayReport):
            return {
                "total": result.total,
                "loaded": result.loaded,
                "failed": result.failed,
                "errors": list(result.errors),
                "results": [asdict(r) for r in result.results],
            }
        return asdict(result)
