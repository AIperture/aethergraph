from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.contracts import StorageScope

RESOURCE_KINDS = {
    "artifact",
    "artifact_uri",
    "upload",
    "url",
    "file_path",
    "directory_path",
}
RESOURCE_STATUSES = {"candidate", "materialized", "failed"}

_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_ARTIFACT_CONTENT_PATH_RE = re.compile(r"^/api/v1/artifacts/([^/]+)/content/?$")
_ARTIFACT_URI_RE = re.compile(r"artifact://[^\s<>()\"']+")
_ARTIFACT_CONTENT_TEXT_RE = re.compile(r"(?<!\S)(/api/v1/artifacts/([^/\s]+)/content/?)(?!\S)")
_WINDOWS_PATH_RE = re.compile(r"(?<![\w:/\\])([A-Za-z]:\\[^\s\"'<>|]+)")
_POSIX_PATH_RE = re.compile(r"(?<![\w:])(/[^\s\"'<>|]+)")
_FILE_URI_RE = re.compile(r"file://[^\s\"'<>]+")
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _dict_or_empty(value: Any, *, field_name: str, diagnostics: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    diagnostics.append(f"{field_name} must be an object")
    return {}


def _artifact_id_from_content_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    path = parsed.path or value
    match = _ARTIFACT_CONTENT_PATH_RE.match(path)
    return unquote(match.group(1)) if match else None


def _looks_like_artifact_uri(value: str | None) -> bool:
    return bool(value and _ARTIFACT_URI_RE.match(value))


def _path_kind(path: str) -> str:
    try:
        p = Path(path)
        if p.exists() and p.is_dir():
            return "directory_path"
    except OSError:
        pass
    return "file_path"


@dataclass
class InputResource:
    kind: str
    source: str
    status: str = "candidate"
    id: str | None = None
    name: str | None = None
    mime: str | None = None
    size: int | None = None
    artifact_id: str | None = None
    uri: str | None = None
    url: str | None = None
    path: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in RESOURCE_KINDS:
            self.diagnostics.append(f"unsupported resource kind: {self.kind}")
        if self.status not in RESOURCE_STATUSES:
            self.diagnostics.append(f"unsupported resource status: {self.status}")
            self.status = "candidate"

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        source: str = "unknown",
        trust_policy: ResourceTrustPolicy | None = None,
    ) -> InputResource:
        return InputResourceNormalizer(trust_policy=trust_policy).from_dict(raw, source=source)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "labels": dict(self.labels),
            "meta": dict(self.meta),
        }
        for key in ("id", "name", "size", "artifact_id", "uri", "url", "path"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.mime:
            out["mime"] = self.mime
            out["mimetype"] = self.mime
        if self.diagnostics:
            out["diagnostics"] = list(self.diagnostics)
        return out

    def to_display_file(self) -> dict[str, Any]:
        display_id = self.id or self.artifact_id or self.uri or self.url or self.path or self.name
        out: dict[str, Any] = {
            "id": display_id,
            "name": self.name or self.id or self.artifact_id or self.path or self.url,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "labels": dict(self.labels),
            "meta": dict(self.meta),
        }
        for key in ("artifact_id", "uri", "url", "path", "size"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.mime:
            out["mime"] = self.mime
            out["mimetype"] = self.mime
        return {k: v for k, v in out.items() if v is not None}


def _hydrate_public_artifact(resource: InputResource, artifact: Artifact) -> InputResource:
    if not isinstance(artifact, Artifact):
        raise TypeError("artifact must be the frozen public Artifact DTO")
    if resource.artifact_id is not None and resource.artifact_id != artifact.artifact_id:
        raise ValueError("artifact identity does not match the resource")
    labels = dict(artifact.labels or {})
    resource.artifact_id = artifact.artifact_id
    resource.name = resource.name or labels.get("filename") or artifact.artifact_id
    resource.mime = resource.mime or artifact.mime
    if resource.size is None:
        resource.size = artifact.bytes
    resource.uri = resource.uri or artifact.uri
    resource.url = resource.url or f"/api/v1/artifacts/{artifact.artifact_id}/content"
    if not resource.labels:
        resource.labels = labels
    return resource


class ResourceSet:
    def __init__(self, resources: list[InputResource] | None = None) -> None:
        self.resources: list[InputResource] = []
        if resources:
            self.extend(resources)

    def __iter__(self):
        return iter(self.resources)

    def __len__(self) -> int:
        return len(self.resources)

    def add(self, resource: InputResource | None) -> None:
        if resource is not None:
            self.resources.append(resource)

    def extend(self, resources: list[InputResource] | ResourceSet) -> None:
        iterable = resources.resources if isinstance(resources, ResourceSet) else resources
        for resource in iterable:
            self.add(resource)

    def dedupe(self) -> ResourceSet:
        deduped: list[InputResource] = []
        key_to_index: dict[tuple[str, str], int] = {}

        for resource in self.resources:
            keys = _resource_keys(resource)
            existing_index = next(
                (key_to_index[key] for key in keys if key in key_to_index),
                None,
            )
            if existing_index is None:
                index = len(deduped)
                deduped.append(resource)
                for key in keys:
                    key_to_index[key] = index
                continue

            existing = deduped[existing_index]
            if _resource_rank(resource) > _resource_rank(existing):
                deduped[existing_index] = resource
                for key in _resource_keys(existing):
                    key_to_index.pop(key, None)
                for key in keys:
                    key_to_index[key] = existing_index

        self.resources = deduped
        return self

    def to_dicts(self) -> list[dict[str, Any]]:
        return [resource.to_dict() for resource in self.dedupe().resources]

    def to_attachment_dicts(self) -> list[dict[str, Any]]:
        return self.to_dicts()

    def to_display_files(self) -> list[dict[str, Any]]:
        return [resource.to_display_file() for resource in self.dedupe().resources]


def _resource_keys(resource: InputResource) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if resource.artifact_id:
        keys.append(("artifact_id", resource.artifact_id))
    if resource.uri:
        keys.append(("uri", resource.uri))
    if resource.url:
        artifact_id = _artifact_id_from_content_url(resource.url)
        if artifact_id:
            keys.append(("artifact_id", artifact_id))
        keys.append(("url", resource.url))
    if resource.path:
        keys.append(("path", str(Path(resource.path))))
    if resource.id:
        keys.append(("transport_id", f"{resource.source}:{resource.id}"))
    if not keys:
        keys.append(("object", repr(resource.to_dict())))
    return keys


def _resource_rank(resource: InputResource) -> tuple[int, int, int]:
    status_score = {"materialized": 3, "candidate": 2, "failed": 1}.get(resource.status, 0)
    kind_score = {
        "artifact": 6,
        "upload": 5,
        "artifact_uri": 4,
        "file_path": 3,
        "directory_path": 3,
        "url": 2,
    }.get(resource.kind, 0)
    detail_score = sum(
        1
        for value in (
            resource.artifact_id,
            resource.uri,
            resource.url,
            resource.path,
            resource.name,
            resource.mime,
            resource.size,
        )
        if value is not None
    )
    return status_score, kind_score, detail_score


@dataclass(frozen=True)
class ResourceTrustPolicy:
    trusted_sources: frozenset[str] = frozenset({"webui", "console", "local_api"})
    untrusted_sources: frozenset[str] = frozenset(
        {"slack", "telegram", "external_api", "public_webhook"}
    )

    def allows_local_materialization(self, source: str | None) -> bool:
        return (source or "").strip() in self.trusted_sources


class InputResourceNormalizer:
    def __init__(
        self,
        *,
        trust_policy: ResourceTrustPolicy | None = None,
        retain_untrusted_paths: bool = True,
    ) -> None:
        self.trust_policy = trust_policy or ResourceTrustPolicy()
        self.retain_untrusted_paths = retain_untrusted_paths

    def from_dict(self, raw: dict[str, Any], *, source: str = "unknown") -> InputResource:
        diagnostics: list[str] = []
        eff_source = _clean_str(raw.get("source")) or source
        mime = _clean_str(raw.get("mime")) or _clean_str(raw.get("mimetype"))
        labels = _dict_or_empty(raw.get("labels"), field_name="labels", diagnostics=diagnostics)
        meta = _dict_or_empty(raw.get("meta"), field_name="meta", diagnostics=diagnostics)

        artifact_id = _clean_str(raw.get("artifact_id"))
        url = _clean_str(raw.get("url"))
        uri = _clean_str(raw.get("uri"))
        path = (
            _clean_str(raw.get("path"))
            or _clean_str(raw.get("file_path"))
            or _clean_str(raw.get("directory_path"))
        )
        content_artifact_id = _artifact_id_from_content_url(url)
        if content_artifact_id:
            artifact_id = artifact_id or content_artifact_id

        kind = _clean_str(raw.get("kind"))
        if artifact_id:
            kind = "artifact"
        elif path:
            kind = kind if kind in {"file_path", "directory_path"} else _path_kind(path)
        elif uri and _looks_like_artifact_uri(uri):
            kind = "artifact_uri"
        elif url:
            kind = "url"
        elif kind not in RESOURCE_KINDS:
            diagnostics.append("resource kind could not be inferred")
            kind = kind or "url"

        status = _clean_str(raw.get("status"))
        if not status:
            status = (
                "materialized" if artifact_id or kind in {"artifact_uri", "upload"} else "candidate"
            )

        if path:
            materialization_allowed = self.trust_policy.allows_local_materialization(eff_source)
            meta.setdefault("materialization_allowed", materialization_allowed)
            if not materialization_allowed:
                status = "candidate"

        return InputResource(
            kind=kind,
            source=eff_source,
            status=status,
            id=_clean_str(raw.get("id")),
            name=_clean_str(raw.get("name") or raw.get("filename")),
            mime=mime,
            size=_clean_int(raw.get("size") if "size" in raw else raw.get("bytes")),
            artifact_id=artifact_id,
            uri=uri,
            url=url,
            path=path,
            labels=labels,
            meta=meta,
            diagnostics=diagnostics,
        )

    def from_text(self, text: str | None, *, source: str = "unknown") -> ResourceSet:
        resources = ResourceSet()
        if not text:
            return resources

        seen_ranges: list[tuple[int, int]] = []
        for match in _ARTIFACT_CONTENT_TEXT_RE.finditer(text):
            url = match.group(1)
            artifact_id = unquote(match.group(2))
            seen_ranges.append((match.start(), match.end()))
            resources.add(
                InputResource(
                    kind="artifact",
                    source=source,
                    status="materialized",
                    artifact_id=artifact_id,
                    url=url,
                )
            )

        for match in _URL_RE.finditer(text):
            url = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
            seen_ranges.append((match.start(), match.start() + len(url)))
            artifact_id = _artifact_id_from_content_url(url)
            if artifact_id:
                resources.add(
                    InputResource(
                        kind="artifact",
                        source=source,
                        status="materialized",
                        artifact_id=artifact_id,
                        url=url,
                    )
                )
            else:
                resources.add(InputResource(kind="url", source=source, status="candidate", url=url))

        for match in _ARTIFACT_URI_RE.finditer(text):
            uri = match.group(0)
            seen_ranges.append((match.start(), match.end()))
            resources.add(
                InputResource(kind="artifact_uri", source=source, status="materialized", uri=uri)
            )

        for path in self._extract_local_paths(text, source=source, seen_ranges=seen_ranges):
            resources.add(path)

        return resources.dedupe()

    def from_incoming_file(self, incoming_file: Any, *, source: str) -> InputResource:
        raw = {
            "id": getattr(incoming_file, "id", None),
            "name": getattr(incoming_file, "name", None),
            "mimetype": getattr(incoming_file, "mimetype", None),
            "size": getattr(incoming_file, "size", None),
            "url": getattr(incoming_file, "url", None),
            "uri": getattr(incoming_file, "uri", None),
            "artifact_id": getattr(incoming_file, "artifact_id", None),
            "source": source,
            "meta": getattr(incoming_file, "extra", None) or {},
        }
        resource = self.from_dict(raw, source=source)
        if resource.artifact_id or resource.uri:
            resource.status = "materialized"
            if not resource.kind or resource.kind == "url":
                resource.kind = "artifact" if resource.artifact_id else "artifact_uri"
        elif resource.url:
            resource.kind = "url"
            resource.status = "candidate"
        return resource

    def from_artifact(self, artifact: Artifact, *, source: str = "artifact") -> InputResource:
        resource = InputResource(
            kind="artifact",
            source=source,
            status="materialized",
        )
        return _hydrate_public_artifact(resource, artifact)

    def _extract_local_paths(
        self,
        text: str,
        *,
        source: str,
        seen_ranges: list[tuple[int, int]],
    ) -> list[InputResource]:
        resources: list[InputResource] = []
        for regex in (_FILE_URI_RE, _WINDOWS_PATH_RE, _POSIX_PATH_RE):
            for match in regex.finditer(text):
                if any(
                    match.start() >= start and match.start() < end for start, end in seen_ranges
                ):
                    continue
                raw_path = match.group(0)
                path = _path_from_file_uri(raw_path) if raw_path.startswith("file://") else raw_path
                path = path.rstrip(".,;:")
                allowed = self.trust_policy.allows_local_materialization(source)
                if not allowed and not self.retain_untrusted_paths:
                    continue
                resources.append(
                    InputResource(
                        kind=_path_kind(path),
                        source=source,
                        status="candidate",
                        path=path,
                        meta={"materialization_allowed": allowed},
                    )
                )
        return resources


def _path_from_file_uri(uri: str) -> str:
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if parsed.netloc and not path.startswith("/"):
        path = f"//{parsed.netloc}/{path}"
    if re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    return path


class ResourceEnricher:
    def __init__(self, *, container: Any) -> None:
        self.container = container

    async def enrich(self, resources: ResourceSet | list[InputResource]) -> ResourceSet:
        resource_set = resources if isinstance(resources, ResourceSet) else ResourceSet(resources)
        artifacts = getattr(self.container, "artifact_service", None)
        get_artifact = getattr(artifacts, "get_by_id", None) if artifacts is not None else None

        for resource in resource_set.resources:
            if resource.kind != "artifact" or not resource.artifact_id:
                continue
            resource.url = resource.url or f"/api/v1/artifacts/{resource.artifact_id}/content"
            if get_artifact is None:
                continue
            try:
                artifact = await get_artifact(resource.artifact_id)
            except Exception as exc:
                resource.diagnostics.append(f"artifact enrichment failed: {exc}")
                continue
            if artifact is None:
                continue
            if not isinstance(artifact, Artifact):
                resource.diagnostics.append("artifact enrichment returned an invalid public DTO")
                continue
            if artifact.artifact_id != resource.artifact_id:
                resource.diagnostics.append("artifact enrichment returned a different identity")
                continue
            _hydrate_public_artifact(resource, artifact)
        return resource_set.dedupe()


@dataclass(frozen=True)
class ArtifactIngressScope:
    source: str
    session_id: str | None = None
    run_id: str | None = None
    channel_key: str | None = None
    conversation_id: str | None = None
    graph_id: str = "channel"
    node_id: str = "resource_ingress"
    tool_name: str = "channel.resource_ingress"
    tool_version: str = "1.0.0"

    def scope_id(self) -> str:
        if self.session_id:
            return f"session:{self.session_id}"
        if self.run_id:
            return f"run:{self.run_id}"
        if self.channel_key:
            return f"channel:{self.channel_key}"
        if self.conversation_id:
            return f"conversation:{self.conversation_id}"
        return "channel:unknown"


class ResourceStager:
    def __init__(
        self,
        *,
        container: Any,
        identity: Any | None = None,
        storage_scope: StorageScope | None = None,
    ) -> None:
        if identity is not None and storage_scope is not None:
            raise ValueError("ResourceStager accepts identity or storage_scope, not both")
        self.container = container
        self.identity = identity
        self.storage_scope = storage_scope

    async def stage_bytes(
        self,
        data: bytes,
        *,
        name: str,
        mime: str | None,
        file_id: str | None = None,
        scope: ArtifactIngressScope,
        labels: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        suggested_uri: str | None = None,
    ) -> InputResource:
        facade = self._facade(scope)
        suffix = Path(name).suffix if name else ""
        tmp_path = await facade.stage_path(f"_{uuid4().hex[:8]}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(data)

        eff_labels = self._labels(scope, name=name, file_id=file_id, labels=labels)
        artifact = await facade.save_file(
            path=tmp_path,
            kind="upload",
            mime=mime,
            labels=eff_labels,
            suggested_uri=suggested_uri or f"./uploads/{name}",
            name=name,
        )
        resource = InputResource(
            kind="upload",
            source=scope.source,
            status="materialized",
            id=file_id,
            name=name,
            mime=mime,
            size=len(data),
            labels=eff_labels,
            meta=dict(meta or {}),
        )
        return _hydrate_public_artifact(resource, artifact)

    def _facade(self, scope: ArtifactIngressScope) -> CanonicalPublicArtifactFacade:
        if self.storage_scope is not None:
            values = self.storage_scope.as_filter()
            for name, value in (
                ("session_id", scope.session_id),
                ("run_id", scope.run_id),
                ("graph_id", scope.graph_id),
                ("node_id", scope.node_id),
            ):
                if value is None:
                    continue
                if name in values and values[name] != value:
                    raise ValueError(f"resource scope conflicts with canonical {name}")
                values[name] = value
            return self.container.artifact_factory.for_public_execution(
                StorageScope(**values),
                tool_name=scope.tool_name,
                tool_version=scope.tool_version,
            )
        scope_obj = self._scope(scope)
        storage_scope = StorageScope(
            org_id=scope_obj.org_id,
            user_id=scope_obj.user_id,
            session_id=scope_obj.session_id,
            run_id=scope_obj.run_id,
            graph_id=scope_obj.graph_id,
            node_id=scope_obj.node_id,
            agent_id=scope_obj.agent_id,
            scope_key=(
                None if scope_obj.session_id or scope_obj.run_id else scope_obj.memory_scope_id()
            ),
        )
        return self.container.artifact_factory.for_public_execution(
            storage_scope,
            tool_name=scope.tool_name,
            tool_version=scope.tool_version,
        )

    def _scope(self, scope: ArtifactIngressScope) -> Scope:
        factory = getattr(self.container, "scope_factory", None)
        if factory is not None:
            scope_obj = factory.for_node(
                identity=self.identity,
                run_id=scope.run_id,
                graph_id=scope.graph_id,
                node_id=scope.node_id,
                session_id=scope.session_id,
                tool_name=scope.tool_name,
                tool_version=scope.tool_version,
            )
        else:
            scope_obj = Scope(
                session_id=scope.session_id,
                run_id=scope.run_id,
                graph_id=scope.graph_id,
                node_id=scope.node_id,
                tool_name=scope.tool_name,
                tool_version=scope.tool_version,
            )

        if scope.session_id:
            return replace(scope_obj, memory_level="session")
        if scope.run_id:
            return replace(scope_obj, memory_level="run")
        return scope_obj.with_memory_scope(scope.scope_id(), memory_level="scope")

    @staticmethod
    def _labels(
        scope: ArtifactIngressScope,
        *,
        name: str,
        file_id: str | None,
        labels: dict[str, Any] | None,
    ) -> dict[str, Any]:
        out = dict(labels or {})
        out.update(
            {
                "source": scope.source,
                "scope_id": scope.scope_id(),
                "name": name,
            }
        )
        if file_id:
            out["inbound_file_id"] = file_id
        if scope.session_id:
            out["session_id"] = scope.session_id
        if scope.run_id:
            out["run_id"] = scope.run_id
        if scope.channel_key:
            out["channel_key"] = scope.channel_key
        if scope.conversation_id:
            out["conversation_id"] = scope.conversation_id
        return out
