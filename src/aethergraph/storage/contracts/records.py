"""Immutable canonical records shared by storage-provider implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import math
from types import MappingProxyType
from typing import TypeAlias

from .scope import StorageScope

JsonScalar: TypeAlias = None | bool | int | float | str
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_nonempty(name: str, value: str | None) -> None:
    if value is not None:
        _nonempty(name, value)


def _optional_text(name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")


def _positive_version(value: int) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError("schema_version must be a positive integer")


def _utc(name: str, value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")


def _freeze_json(value: object, *, path: str = "value") -> FrozenJson:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _freeze_mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, FrozenJson]:
    frozen = _freeze_json(value, path=path)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return frozen


def _freeze_metrics(value: Mapping[str, float], *, path: str) -> Mapping[str, float]:
    frozen: dict[str, float] = {}
    for key, metric in value.items():
        _nonempty(f"{path} key", key)
        if isinstance(metric, bool) or not isinstance(metric, int | float):
            raise TypeError(f"{path}.{key} must be numeric")
        number = float(metric)
        if not math.isfinite(number):
            raise ValueError(f"{path}.{key} must be finite")
        frozen[key] = number
    return MappingProxyType(frozen)


def _validate_tags(tags: tuple[str, ...]) -> None:
    if not isinstance(tags, tuple):
        raise TypeError("tags must be an immutable tuple")
    if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("tags must contain non-empty strings")
    if len(set(tags)) != len(tags):
        raise ValueError("tags must not contain duplicates")


@dataclass(frozen=True, slots=True, kw_only=True)
class EventDraft:
    """Provider-independent event content before an ordered cursor is assigned."""

    event_id: str
    occurred_at: datetime
    scope: StorageScope
    kind: str
    stage: str | None = None
    topic: str | None = None
    text: str | None = None
    tags: tuple[str, ...] = ()
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    severity: int | None = None
    signal: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("event_id", self.event_id)
        _utc("occurred_at", self.occurred_at)
        _nonempty("kind", self.kind)
        _optional_nonempty("stage", self.stage)
        _optional_nonempty("topic", self.topic)
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string when supplied")
        _validate_tags(self.tags)
        if self.severity is not None and (
            isinstance(self.severity, bool) or not 0 <= self.severity <= 100
        ):
            raise ValueError("severity must be between 0 and 100 when supplied")
        if self.signal is not None and (
            isinstance(self.signal, bool)
            or not isinstance(self.signal, int | float)
            or not math.isfinite(float(self.signal))
        ):
            raise ValueError("signal must be a finite number when supplied")
        _positive_version(self.schema_version)
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics, path="metrics"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EventRecord(EventDraft):
    """Committed canonical event with a provider-owned monotonic cursor."""

    cursor: str

    def __post_init__(self) -> None:
        super(EventRecord, self).__post_init__()
        _nonempty("cursor", self.cursor)


@dataclass(frozen=True, slots=True, kw_only=True)
class StateRecord:
    """Current canonical state value at one exact optimistic revision."""

    namespace: str
    key: str
    value: FrozenJson
    revision: int
    scope: StorageScope
    updated_at: datetime
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("namespace", self.namespace)
        _nonempty("key", self.key)
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("updated_at", self.updated_at)
        _positive_version(self.schema_version)
        object.__setattr__(self, "value", _freeze_json(self.value))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


class ArtifactAction(StrEnum):
    """Canonical action represented by one artifact occurrence."""

    PRODUCED = "produced"
    CONSUMED = "consumed"
    ATTACHED = "attached"
    PUBLISHED = "published"


class ArtifactRelationKind(StrEnum):
    """Canonical directed lineage relation between two artifact records."""

    DERIVED_FROM = "derived_from"
    TRANSFORMED_FROM = "transformed_from"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRecord:
    """Immutable artifact content metadata and provider-neutral blob identity."""

    artifact_id: str
    content_hash: str
    hash_algorithm: str
    size_bytes: int
    media_type: str
    kind: str
    blob_locator: str
    owner_scope: StorageScope
    created_at: datetime
    preview_locator: str | None = None
    original_filename: str | None = None
    provider_version: str | None = None
    labels: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "content_hash",
            "hash_algorithm",
            "media_type",
            "kind",
            "blob_locator",
        ):
            _nonempty(name, getattr(self, name))
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        _utc("created_at", self.created_at)
        for name in ("preview_locator", "original_filename", "provider_version"):
            _optional_nonempty(name, getattr(self, name))
        _positive_version(self.schema_version)
        object.__setattr__(self, "labels", _freeze_mapping(self.labels, path="labels"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRetentionRecord:
    """Revisioned mutable retention intent for immutable artifact content."""

    artifact_id: str
    scope: StorageScope
    pinned: bool
    revision: int
    updated_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("artifact_id", self.artifact_id)
        if not isinstance(self.pinned, bool):
            raise TypeError("pinned must be a boolean")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 1:
            raise ValueError("revision must be positive")
        _utc("updated_at", self.updated_at)
        _positive_version(self.schema_version)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactOccurrence:
    """One production or use of immutable artifact content in execution scope."""

    occurrence_id: str
    artifact_id: str
    scope: StorageScope
    action: ArtifactAction
    occurred_at: datetime
    tool_name: str | None = None
    tool_version: str | None = None
    labels: Mapping[str, FrozenJson] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("occurrence_id", self.occurrence_id)
        _nonempty("artifact_id", self.artifact_id)
        _utc("occurred_at", self.occurred_at)
        _optional_nonempty("tool_name", self.tool_name)
        _optional_nonempty("tool_version", self.tool_version)
        _positive_version(self.schema_version)
        object.__setattr__(self, "labels", _freeze_mapping(self.labels, path="labels"))
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics, path="metrics"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRelation:
    """Directed typed lineage edge between canonical artifact records."""

    relation_id: str
    source_artifact_id: str
    target_artifact_id: str
    kind: ArtifactRelationKind
    scope: StorageScope
    created_at: datetime
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("relation_id", self.relation_id)
        _nonempty("source_artifact_id", self.source_artifact_id)
        _nonempty("target_artifact_id", self.target_artifact_id)
        if self.source_artifact_id == self.target_artifact_id:
            raise ValueError("artifact lineage must not create a self-edge")
        _utc("created_at", self.created_at)
        _positive_version(self.schema_version)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


class SearchMode(StrEnum):
    """Exact search capability requested without fallback semantics."""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchDocument:
    """Canonical searchable projection keyed by stable corpus and item identity."""

    corpus: str
    item_id: str
    text: str
    scope: StorageScope
    occurred_at: datetime
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("corpus", self.corpus)
        _nonempty("item_id", self.item_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _utc("occurred_at", self.occurred_at)
        _validate_tags(self.tags)
        object.__setattr__(self, "tags", tuple(sorted(self.tags)))
        _positive_version(self.schema_version)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchQuery:
    """Bounded exact-mode search request with optional freshness requirement."""

    corpus: str
    mode: SearchMode
    scope: StorageScope
    query: str = ""
    top_k: int = 10
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    occurred_at_min: datetime | None = None
    occurred_at_max: datetime | None = None
    require_indexed_cursor: str | None = None

    def __post_init__(self) -> None:
        _nonempty("corpus", self.corpus)
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if self.mode is not SearchMode.STRUCTURAL and not self.query.strip():
            raise ValueError("semantic, lexical, and hybrid search require a query")
        if isinstance(self.top_k, bool) or not 1 <= self.top_k <= 1_000:
            raise ValueError("top_k must be between 1 and 1000")
        _validate_tags(self.tags)
        object.__setattr__(self, "tags", tuple(sorted(self.tags)))
        for name in ("occurred_at_min", "occurred_at_max"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if (
            self.occurred_at_min is not None
            and self.occurred_at_max is not None
            and self.occurred_at_min > self.occurred_at_max
        ):
            raise ValueError("occurred_at_min must not be after occurred_at_max")
        _optional_nonempty("require_indexed_cursor", self.require_indexed_cursor)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """Stable search hit with explicit mode score and immutable metadata."""

    corpus: str
    item_id: str
    score: float
    mode: SearchMode
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty("corpus", self.corpus)
        _nonempty("item_id", self.item_id)
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BlobRange:
    """Half-open byte range requested from canonical artifact content."""

    start: int
    end: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if self.end is not None and (isinstance(self.end, bool) or self.end <= self.start):
            raise ValueError("end must be greater than start when supplied")


@dataclass(frozen=True, slots=True, kw_only=True)
class BlobWriteResult:
    """Provider-neutral result of an integrity-checked streaming blob write."""

    blob_locator: str
    content_hash: str
    hash_algorithm: str
    size_bytes: int
    provider_version: str | None = None

    def __post_init__(self) -> None:
        _nonempty("blob_locator", self.blob_locator)
        _nonempty("content_hash", self.content_hash)
        _nonempty("hash_algorithm", self.hash_algorithm)
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        _optional_nonempty("provider_version", self.provider_version)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlobHead:
    """Provider-neutral metadata returned without reading blob content."""

    blob_locator: str
    size_bytes: int
    content_hash: str
    hash_algorithm: str
    provider_version: str | None = None

    def __post_init__(self) -> None:
        _nonempty("blob_locator", self.blob_locator)
        _nonempty("content_hash", self.content_hash)
        _nonempty("hash_algorithm", self.hash_algorithm)
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        _optional_nonempty("provider_version", self.provider_version)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactOrphanCleanupResult:
    """Bounded result of reference-safe artifact blob maintenance."""

    examined: int
    deleted_scoped_blobs: int
    deleted_physical_blobs: int
    freed_bytes: int
    has_more: bool

    def __post_init__(self) -> None:
        for name in (
            "examined",
            "deleted_scoped_blobs",
            "deleted_physical_blobs",
            "freed_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.has_more, bool):
            raise TypeError("has_more must be a boolean")
