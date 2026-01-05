from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import re
import threading
from typing import Any

try:
    # Prefer packaging for correct PEP 440 / pre-release ordering
    from packaging.version import Version

    _has_packaging = True
except Exception:
    _has_packaging = False

from .key_parsing import parse_ref
from .registry_key import NS, Key

# allow storing either the object, or a factory that returns the object on first use
RegistryObject = Any
RegistryFactory = Callable[[], Any]
RegistryValue = RegistryObject | RegistryFactory


class UnifiedRegistry:
    """
    Runtime-only registry: (nspace, name, version) -> object (or lazy factory).
    Maintains a 'latest' pointer per (nspace, name).

    Thread-safe for concurrent get/register operations.
    """

    def __init__(self, *, allow_overwrite: bool = True):
        self._store: dict[tuple[str, str], dict[str, RegistryValue]] = {}
        self._latest: dict[tuple[str, str], str] = {}
        self._aliases: dict[tuple[str, str], dict[str, str]] = {}  # (ns,name) -> alias -> version
        self._lock = threading.RLock()
        self._allow_overwrite = allow_overwrite

        # per-version metadata
        self._meta: dict[tuple[str, str, str], dict[str, Any]] = {}

    # ---------- registration ----------

    def register(
        self,
        *,
        nspace: str,
        name: str,
        version: str,
        obj: RegistryValue,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if nspace not in NS:
            raise ValueError(f"Unknown namespace: {nspace}")
        key = (nspace, name)
        with self._lock:
            versions = self._store.setdefault(key, {})
            if (version in versions) and not self._allow_overwrite:
                raise ValueError(
                    f"{nspace}:{name}@{version} already registered and overwrite disabled"
                )
            versions[version] = obj
            self._latest[key] = self._pick_latest(versions.keys())

            # Store metadata
            if meta is not None:
                self._meta[(nspace, name, version)] = meta

    def register_latest(
        self, *, nspace: str, name: str, obj: RegistryValue, version: str = "0.0.0"
    ) -> None:
        # Explicit version anyway; also marks latest via _pick_latest
        self.register(nspace=nspace, name=name, version=version, obj=obj)

    def alias(self, *, nspace: str, name: str, tag: str, to_version: str) -> None:
        """Define tag aliases like 'stable', 'canary' mapping to a concrete version."""
        key = (nspace, name)
        with self._lock:
            if key not in self._store or to_version not in self._store[key]:
                raise KeyError(f"Cannot alias to missing version: {nspace}:{name}@{to_version}")
            m = self._aliases.setdefault(key, {})
            m[tag] = to_version

    # ---------- resolve ----------

    def get(self, ref: str | Key) -> Any:
        key = parse_ref(ref) if isinstance(ref, str) else ref
        k = (key.nspace, key.name)
        with self._lock:
            versions = self._store.get(k)
            if not versions:
                raise KeyError(f"Not found: {key.canonical()}")

            # resolve version: explicit → alias → latest
            ver = key.version
            ver = self._aliases.get(k, {}).get(ver, ver) if ver else self._latest.get(k)

            if ver not in versions:
                raise KeyError(f"Version not found: {key.nspace}:{key.name}@{ver}")

            val = versions[ver]

            ## Materialize if factory -> we handle it when executing the graphs. Here it can cause
            # the graph_fn returns a coroutine inside the GraphFunction object, not the expected function.
            # if callable(val):
            #     obj = val()
            #     versions[ver] = obj
            #     return obj
            return val

    # ---------- listing / admin ----------

    def list(self, nspace: str | None = None) -> dict[str, str]:
        """Return { 'ns:name': '<latest_version>' } optionally filtered."""
        out: dict[str, str] = {}
        with self._lock:
            for (ns, name), _ in self._store.items():
                if nspace and ns != nspace:
                    continue
                out[f"{ns}:{name}"] = self._latest.get((ns, name), "unknown")
        return out

    def list_versions(self, *, nspace: str, name: str) -> Iterable[str]:
        k = (nspace, name)
        with self._lock:
            return tuple(sorted(self._store.get(k, {}).keys(), key=self._semver_sort_key))

    def get_aliases(self, *, nspace: str, name: str) -> Mapping[str, str]:
        with self._lock:
            return dict(self._aliases.get((nspace, name), {}))

    def unregister(self, *, nspace: str, name: str, version: str | None = None) -> None:
        with self._lock:
            k = (nspace, name)
            if k not in self._store:
                return
            if version is None:
                # remove all versions and aliases
                self._store.pop(k, None)
                self._latest.pop(k, None)
                self._aliases.pop(k, None)
                # NEW: drop all meta for this (ns,name)
                for key in list(self._meta.keys()):
                    if key[0] == nspace and key[1] == name:
                        self._meta.pop(key, None)
                return
            vers = self._store[k]
            vers.pop(version, None)
            # drop aliases pointing to this version
            if k in self._aliases:
                for tag, v in list(self._aliases[k].items()):
                    if v == version:
                        self._aliases[k].pop(tag, None)
            # drop meta for this version
            self._meta.pop((nspace, name, version), None)
            # recompute latest
            if vers:
                self._latest[k] = self._pick_latest(vers.keys())
            else:
                self._store.pop(k, None)
                self._latest.pop(k, None)
                self._aliases.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._latest.clear()
            self._aliases.clear()
            self._meta.clear()

    # ---------- typed getters ----------

    def get_tool(self, name: str, version: str | None = None) -> Any:
        return self.get(Key(nspace="tool", name=name, version=version))

    def get_graph(self, name: str, version: str | None = None) -> Any:
        return self.get(Key(nspace="graph", name=name, version=version))

    def get_graphfn(self, name: str, version: str | None = None) -> Any:
        return self.get(Key(nspace="graphfn", name=name, version=version))

    def get_agent(self, name: str, version: str | None = None) -> Any:
        return self.get(Key(nspace="agent", name=name, version=version))

    def get_meta(
        self,
        nspace: str,
        name: str,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Return metadata for a given registered object, or None if not set.
        Follows the same version resolution as `get()`: explicit → alias → latest.
        """
        if nspace not in NS:
            raise ValueError(f"Unknown namespace: {nspace}")
        key = (nspace, name)
        with self._lock:
            versions = self._store.get(key)
            if not versions:
                return None

            ver = version
            # resolve aliases or default to latest
            ver = self._aliases.get(key, {}).get(ver, ver) if ver else self._latest.get(key)
            if ver is None:
                return None

            return self._meta.get((nspace, name, ver))

    # ---------- list typed ----------
    def list_tools(self) -> dict[str, str]:
        return self.list(nspace="tool")

    def list_graphs(self) -> dict[str, str]:
        return self.list(nspace="graph")

    def list_graphfns(self) -> dict[str, str]:
        return self.list(nspace="graphfn")

    def list_agents(self) -> dict[str, str]:
        # Return {'agent:<id>': '<latest_version>'}
        return self.list(nspace="agent")

    def list_apps(self) -> dict[str, str]:
        # Return {'app:<id>': '<latest_version>'}
        return self.list(nspace="app")

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
