from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import re
import threading
from typing import Any

from aethergraph.services.scope.tenant import normalize_registry_tenant

try:
    # Prefer packaging for correct PEP 440 / pre-release ordering
    from packaging.version import Version  # type: ignore

    _has_packaging = True
except Exception:
    _has_packaging = False

from .key_parsing import parse_ref
from .registry_key import NS, Key

# allow storing either the object, or a factory that returns the object on first use
RegistryObject = Any
RegistryFactory = Callable[[], Any]
RegistryValue = RegistryObject | RegistryFactory
TenantIdentity = Mapping[str, str | None] | None
_GLOBAL_TENANT_KEY = "__global__"


class UnifiedRegistry:
    """
    Runtime-only registry: (nspace, name, version) -> object (or lazy factory).
    Maintains a 'latest' pointer per (nspace, name).

    Thread-safe for concurrent get/register operations.
    """

    def __init__(self, *, allow_overwrite: bool = True):
        # (ns,name,tenant_key) -> version -> object
        self._store: dict[tuple[str, str, str], dict[str, RegistryValue]] = {}
        self._latest: dict[tuple[str, str, str], str] = {}
        # (ns,name,tenant_key) -> alias -> version
        self._aliases: dict[tuple[str, str, str], dict[str, str]] = {}
        self._lock = threading.RLock()
        self._allow_overwrite = allow_overwrite

        # per-version metadata
        self._meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    @staticmethod
    def _normalize_tenant(tenant: TenantIdentity) -> dict[str, str | None] | None:
        return normalize_registry_tenant(tenant)

    @staticmethod
    def _tenant_key(tenant: dict[str, str | None] | None) -> str:
        if tenant is None:
            return _GLOBAL_TENANT_KEY
        return f"org:{tenant.get('org_id') or ''}|user:{tenant.get('user_id') or ''}"

    def _candidate_tenant_keys(
        self, tenant: TenantIdentity, include_global: bool
    ) -> tuple[str, ...]:
        norm = self._normalize_tenant(tenant)
        if norm is None:
            return (_GLOBAL_TENANT_KEY,)
        key = self._tenant_key(norm)
        if include_global and key != _GLOBAL_TENANT_KEY:
            return (key, _GLOBAL_TENANT_KEY)
        return (key,)

    def _bucket_key(
        self, *, nspace: str, name: str, tenant: TenantIdentity = None
    ) -> tuple[str, str, str]:
        return (nspace, name, self._tenant_key(self._normalize_tenant(tenant)))

    def _resolve_version(self, bucket: tuple[str, str, str], version: str | None) -> str | None:
        versions = self._store.get(bucket)
        if not versions:
            return None
        if version is None:
            return self._latest.get(bucket)
        return self._aliases.get(bucket, {}).get(version, version)

    # ---------- registration ----------

    def register(
        self,
        *,
        nspace: str,
        name: str,
        version: str,
        obj: RegistryValue,
        meta: dict[str, Any] | None = None,
        tenant: TenantIdentity = None,
    ) -> None:
        if nspace not in NS:
            raise ValueError(f"Unknown namespace: {nspace}")
        tenant_norm = self._normalize_tenant(tenant)
        key = self._bucket_key(nspace=nspace, name=name, tenant=tenant_norm)
        with self._lock:
            versions = self._store.setdefault(key, {})
            if (version in versions) and not self._allow_overwrite:
                raise ValueError(
                    f"{nspace}:{name}@{version} already registered and overwrite disabled"
                )
            versions[version] = obj
            self._latest[key] = self._pick_latest(versions.keys())

            # Store metadata
            if meta is not None or tenant_norm is not None:
                stored_meta = dict(meta or {})
                if tenant_norm is not None:
                    stored_meta.setdefault("tenant", dict(tenant_norm))
                self._meta[(nspace, name, key[2], version)] = stored_meta

    def register_latest(
        self,
        *,
        nspace: str,
        name: str,
        obj: RegistryValue,
        version: str = "0.0.0",
        tenant: TenantIdentity = None,
    ) -> None:
        # Explicit version anyway; also marks latest via _pick_latest
        self.register(nspace=nspace, name=name, version=version, obj=obj, tenant=tenant)

    def alias(
        self,
        *,
        nspace: str,
        name: str,
        tag: str,
        to_version: str,
        tenant: TenantIdentity = None,
    ) -> None:
        """Define tag aliases like 'stable', 'canary' mapping to a concrete version."""
        key = self._bucket_key(nspace=nspace, name=name, tenant=tenant)
        with self._lock:
            if key not in self._store or to_version not in self._store[key]:
                raise KeyError(f"Cannot alias to missing version: {nspace}:{name}@{to_version}")
            m = self._aliases.setdefault(key, {})
            m[tag] = to_version

    # ---------- resolve ----------

    def get(
        self,
        ref: str | Key,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        key = parse_ref(ref) if isinstance(ref, str) else ref
        with self._lock:
            for tenant_key in self._candidate_tenant_keys(tenant, include_global):
                bucket = (key.nspace, key.name, tenant_key)
                versions = self._store.get(bucket)
                if not versions:
                    continue

                ver = self._resolve_version(bucket, key.version)
                if ver is None or ver not in versions:
                    continue

                val = versions[ver]

                # Materialize if factory -> we handle it when executing the graphs. Here it can
                # cause graph_fn to return a coroutine inside GraphFunction object.
                # if callable(val):
                #     obj = val()
                #     versions[ver] = obj
                #     return obj
                return val
            raise KeyError(f"Not found: {key.canonical()}")

    # ---------- listing / admin ----------

    def list(
        self,
        nspace: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, str]:
        """Return { 'ns:name': '<latest_version>' } optionally filtered."""
        out: dict[str, str] = {}
        seen: set[str] = set()
        with self._lock:
            for tenant_key in self._candidate_tenant_keys(tenant, include_global):
                for ns, name, tk in self._store.keys():  # noqa: SIM118
                    if tk != tenant_key:
                        continue
                    if nspace and ns != nspace:
                        continue
                    ref = f"{ns}:{name}"
                    if ref in seen:
                        continue
                    out[ref] = self._latest.get((ns, name, tk), "unknown")
                    seen.add(ref)
        return out

    def list_versions(
        self,
        *,
        nspace: str,
        name: str,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Iterable[str]:
        with self._lock:
            for tenant_key in self._candidate_tenant_keys(tenant, include_global):
                bucket = (nspace, name, tenant_key)
                if bucket in self._store:
                    return tuple(
                        sorted(self._store.get(bucket, {}).keys(), key=self._semver_sort_key)
                    )
            return tuple()

    def get_aliases(
        self,
        *,
        nspace: str,
        name: str,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Mapping[str, str]:
        with self._lock:
            for tenant_key in self._candidate_tenant_keys(tenant, include_global):
                bucket = (nspace, name, tenant_key)
                aliases = self._aliases.get(bucket)
                if aliases:
                    return dict(aliases)
            return {}

    def unregister(
        self,
        *,
        nspace: str,
        name: str,
        version: str | None = None,
        tenant: TenantIdentity = None,
    ) -> None:
        with self._lock:
            if tenant is None:
                tenant_keys = [
                    tk
                    for ns, nm, tk in self._store.keys()  # noqa: SIM118
                    if ns == nspace and nm == name
                ]
            else:
                tenant_keys = [self._bucket_key(nspace=nspace, name=name, tenant=tenant)[2]]

            for tenant_key in tenant_keys:
                bucket = (nspace, name, tenant_key)
                if bucket not in self._store:
                    continue
                if version is None:
                    # remove all versions and aliases
                    self._store.pop(bucket, None)
                    self._latest.pop(bucket, None)
                    self._aliases.pop(bucket, None)
                    # drop all meta for this (ns,name,tenant)
                    for mk in list(self._meta.keys()):
                        if mk[0] == nspace and mk[1] == name and mk[2] == tenant_key:
                            self._meta.pop(mk, None)
                    continue

                vers = self._store[bucket]
                vers.pop(version, None)
                # drop aliases pointing to this version
                if bucket in self._aliases:
                    for tag, v in list(self._aliases[bucket].items()):
                        if v == version:
                            self._aliases[bucket].pop(tag, None)
                # drop meta for this version
                self._meta.pop((nspace, name, tenant_key, version), None)
                # recompute latest
                if vers:
                    self._latest[bucket] = self._pick_latest(vers.keys())
                else:
                    self._store.pop(bucket, None)
                    self._latest.pop(bucket, None)
                    self._aliases.pop(bucket, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._latest.clear()
            self._aliases.clear()
            self._meta.clear()

    # ---------- typed getters ----------

    @staticmethod
    def _materialize_if_builder(val: Any) -> Any:
        """
        Materialize values explicitly registered as lazy graph builders.
        """
        if getattr(val, "__ag_builder__", False) and callable(val):
            return val()
        return val

    def get_tool(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self.get(
            Key(nspace="tool", name=name, version=version),
            tenant=tenant,
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
        return self._materialize_if_builder(
            self.get(
                Key(nspace="graph", name=name, version=version),
                tenant=tenant,
                include_global=include_global,
            )
        )

    def get_graphfn(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self._materialize_if_builder(
            self.get(
                Key(nspace="graphfn", name=name, version=version),
                tenant=tenant,
                include_global=include_global,
            )
        )

    def get_agent(
        self,
        name: str,
        version: str | None = None,
        *,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> Any:
        return self._materialize_if_builder(
            self.get(
                Key(nspace="agent", name=name, version=version),
                tenant=tenant,
                include_global=include_global,
            )
        )

    def get_meta(
        self,
        nspace: str,
        name: str,
        version: str | None = None,
        tenant: TenantIdentity = None,
        include_global: bool = True,
    ) -> dict[str, Any] | None:
        """
        Return metadata for a given registered object, or None if not set.
        Follows the same version resolution as `get()`: explicit -> alias -> latest.
        """
        if nspace not in NS:
            raise ValueError(f"Unknown namespace: {nspace}")
        with self._lock:
            for tenant_key in self._candidate_tenant_keys(tenant, include_global):
                bucket = (nspace, name, tenant_key)
                versions = self._store.get(bucket)
                if not versions:
                    continue

                ver = self._resolve_version(bucket, version)
                if ver is None:
                    continue

                meta = self._meta.get((nspace, name, tenant_key, ver))
                if meta is not None:
                    return meta
            return None

    # ---------- list typed ----------
    def list_tools(
        self, *, tenant: TenantIdentity = None, include_global: bool = True
    ) -> dict[str, str]:
        return self.list(nspace="tool", tenant=tenant, include_global=include_global)

    def list_graphs(
        self, *, tenant: TenantIdentity = None, include_global: bool = True
    ) -> dict[str, str]:
        return self.list(nspace="graph", tenant=tenant, include_global=include_global)

    def list_graphfns(
        self, *, tenant: TenantIdentity = None, include_global: bool = True
    ) -> dict[str, str]:
        return self.list(nspace="graphfn", tenant=tenant, include_global=include_global)

    def list_agents(
        self, *, tenant: TenantIdentity = None, include_global: bool = True
    ) -> dict[str, str]:
        # Return {'agent:<id>': '<latest_version>'}
        return self.list(nspace="agent", tenant=tenant, include_global=include_global)

    def list_apps(
        self, *, tenant: TenantIdentity = None, include_global: bool = True
    ) -> dict[str, str]:
        # Return {'app:<id>': '<latest_version>'}
        return self.list(nspace="app", tenant=tenant, include_global=include_global)

    # ---------- helpers ----------

    @staticmethod
    def _semver_sort_key(v: str):
        if _has_packaging:
            try:
                return Version(v)
            except Exception:
                # Fall back to naive
                pass
        # naive: split on dots and dashes, integers first
        parts = []
        for token in re.split(r"[.\-+]", v):
            try:
                parts.append((0, int(token)))
            except ValueError:
                parts.append((1, token))
        return tuple(parts)

    def _pick_latest(self, versions: Iterable[str]) -> str:
        vs = list(versions)
        if not vs:
            return "0.0.0"
        return sorted(vs, key=self._semver_sort_key)[-1]
