from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping, Iterable, Callable, Union
import threading
import re

try:
    # Prefer packaging for correct PEP 440 / pre-release ordering
    from packaging.version import Version
    _has_packaging = True
except Exception:
    _has_packaging = False

from .registry_key import Key, NS
from .key_parsing import parse_ref


# allow storing either the object, or a factory that returns the object on first use
RegistryObject = Any
RegistryFactory = Callable[[], Any]
RegistryValue = Union[RegistryObject, RegistryFactory]


class UnifiedRegistry:
    """
    Runtime-only registry: (nspace, name, version) -> object (or lazy factory).
    Maintains a 'latest' pointer per (nspace, name).

    Thread-safe for concurrent get/register operations.
    """
    def __init__(self, *, allow_overwrite: bool = True):
        self._store: Dict[Tuple[str, str], Dict[str, RegistryValue]] = {}
        self._latest: Dict[Tuple[str, str], str] = {}
        self._aliases: Dict[Tuple[str, str], Dict[str, str]] = {}  # (ns,name) -> alias -> version
        self._lock = threading.RLock()
        self._allow_overwrite = allow_overwrite

    # ---------- registration ----------

    def register(self, *, nspace: str, name: str, version: str, obj: RegistryValue) -> None:
        if nspace not in NS:
            raise ValueError(f"Unknown namespace: {nspace}")
        key = (nspace, name)
        with self._lock:
            versions = self._store.setdefault(key, {})
            if (version in versions) and not self._allow_overwrite:
                raise ValueError(f"{nspace}:{name}@{version} already registered and overwrite disabled")
            versions[version] = obj
            self._latest[key] = self._pick_latest(versions.keys())

    def register_latest(self, *, nspace: str, name: str, obj: RegistryValue, version: str = "0.0.0") -> None:
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
            if ver:
                # try alias first
                ver = self._aliases.get(k, {}).get(ver, ver)
            else:
                ver = self._latest.get(k)

            if ver not in versions:
                raise KeyError(f"Version not found: {key.nspace}:{key.name}@{ver}")

            val = versions[ver]
            # Materialize factories lazily (and cache)
            if callable(val):
                obj = val()
                versions[ver] = obj
                return obj
            return val

    # ---------- listing / admin ----------

    def list(self, nspace: Optional[str] = None) -> Dict[str, str]:
        """Return { 'ns:name': '<latest_version>' } optionally filtered."""
        out: Dict[str, str] = {}
        with self._lock:
            for (ns, name), versions in self._store.items():
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

    def unregister(self, *, nspace: str, name: str, version: Optional[str] = None) -> None:
        with self._lock:
            k = (nspace, name)
            if k not in self._store:
                return
            if version is None:
                # remove all versions and aliases
                self._store.pop(k, None)
                self._latest.pop(k, None)
                self._aliases.pop(k, None)
                return
            vers = self._store[k]
            vers.pop(version, None)
            # drop aliases pointing to this version
            if k in self._aliases:
                for tag, v in list(self._aliases[k].items()):
                    if v == version:
                        self._aliases[k].pop(tag, None)
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

    # ---------- typed getters ----------

    def get_tool(self, name: str, version: Optional[str] = None) -> Any:
        return self.get(Key(nspace="tool", name=name, version=version))

    def get_graph(self, name: str, version: Optional[str] = None) -> Any:
        return self.get(Key(nspace="graph", name=name, version=version))

    def get_graphfn(self, name: str, version: Optional[str] = None) -> Any:
        return self.get(Key(nspace="graphfn", name=name, version=version))

    def get_agent(self, name: str, version: Optional[str] = None) -> Any:
        return self.get(Key(nspace="agent", name=name, version=version))


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
