"""Stable public Artifact behavior over one canonical artifact facade."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
import json
import mimetypes
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.storage.contracts import SearchMode

from .canonical_facade import (
    ArtifactCommitReceipt,
    CanonicalArtifactFacade,
    CanonicalArtifactWriter,
    PublicArtifactSearchHit,
)

_CONTENT_PREFIX = "/api/v1/artifacts/"
_CONTENT_SUFFIX = "/content"


class CanonicalPublicArtifactFacade:
    """Project stable NodeContext Artifact behavior onto canonical repositories."""

    def __init__(
        self,
        *,
        canonical: CanonicalArtifactFacade,
        deprecated_app_id: str | None = None,
    ) -> None:
        """Bind public Artifact behavior to one canonical execution facade.

        Construction retains the already-bound facade and performs no provider I/O,
        selection, lifecycle operation, or physical-path resolution.

        Examples:
            Bind runtime Artifacts:
                ```python
                artifacts = CanonicalPublicArtifactFacade(canonical=canonical)
                ```

            Retain deprecated response metadata:
                ```python
                artifacts = CanonicalPublicArtifactFacade(
                    canonical=canonical,
                    deprecated_app_id="app-1",
                )
                ```

        Args:
            canonical: Exact owner- and execution-bound canonical Artifact facade.
            deprecated_app_id: Optional explicitly deprecated response-only App metadata.

        Returns:
            None: The public projection is ready without persistence I/O.

        Notes:
            Deprecated App identity never affects provider scope, search, authorization,
            artifact identity, occurrence identity, or blob addressing.
        """
        if deprecated_app_id is not None and (
            not isinstance(deprecated_app_id, str)
            or not deprecated_app_id.strip()
            or deprecated_app_id != deprecated_app_id.strip()
        ):
            raise ValueError("deprecated_app_id must be a non-empty exact string when supplied")
        self.canonical = canonical
        self.scope = canonical.execution_scope
        self.run_id = canonical.execution_scope.run_id
        self.graph_id = canonical.execution_scope.graph_id
        self.node_id = canonical.execution_scope.node_id
        self.tool_name = canonical.tool_name
        self.tool_version = canonical.tool_version
        self._deprecated_app_id = deprecated_app_id
        self.last_artifact: Artifact | None = None

    async def stage_path(self, ext: str = "") -> str:
        """Allocate one transient local staging file.

        The canonical service owns creation and validates the suffix before returning
        an absolute producer-only path.

        Examples:
            Stage text output:
                ```python
                path = await artifacts.stage_path(".txt")
                ```

            Stage extension-free output:
                ```python
                path = await artifacts.stage_path()
                ```

        Args:
            ext: Optional exact filename suffix.

        Returns:
            str: Absolute path to an existing empty transient file.

        Notes:
            The path is never persisted as a provider locator or public Artifact URI.
        """
        return await self.canonical.stage_path(ext)

    async def stage_dir(self, suffix: str = "") -> str:
        """Allocate one transient local staging directory.

        The canonical service owns creation and validates the optional suffix.

        Examples:
            Stage a directory:
                ```python
                path = await artifacts.stage_dir()
                ```

            Stage a named directory shape:
                ```python
                path = await artifacts.stage_dir("-frames")
                ```

        Args:
            suffix: Optional exact directory-name suffix.

        Returns:
            str: Absolute path to an existing transient directory.

        Notes:
            Unused staging directories remain caller-owned transient state.
        """
        return await self.canonical.stage_dir(suffix)

    @asynccontextmanager
    async def writer(
        self,
        *,
        kind: str,
        planned_ext: str | None = None,
        mime: str | None = None,
        pin: bool = False,
    ) -> AsyncIterator[CanonicalArtifactWriter]:
        """Stream one public Artifact through canonical persistence.

        Successful context exit projects the canonical commit receipt to the frozen
        public DTO and updates `last_artifact`; failures propagate without another write.

        Examples:
            Stream binary content:
                ```python
                async with artifacts.writer(kind="binary") as writer:
                    await writer.write(b"payload")
                ```

            Stream labeled text:
                ```python
                async with artifacts.writer(
                    kind="report", planned_ext=".txt", mime="text/plain"
                ) as writer:
                    writer.add_labels({"category": "evidence"})
                    await writer.write(b"report")
                ```

        Args:
            kind: Exact canonical Artifact kind.
            planned_ext: Optional transient staging suffix.
            mime: Optional exact media type; inferred from suffix when absent.
            pin: Whether to create explicit pinned retention intent.

        Returns:
            AsyncIterator[CanonicalArtifactWriter]: Asynchronous writer context whose
            `write` method must be awaited.

        Notes:
            The writer is canonical and has no synchronous-write or legacy-store path.
        """
        self.last_artifact = None
        media_type = mime or _media_type(planned_ext)
        async with self.canonical.writer(
            kind=kind,
            media_type=media_type,
            planned_ext=planned_ext,
            pinned=pin,
        ) as stream:
            yield stream
        if stream.receipt is None:
            raise RuntimeError("canonical Artifact writer completed without a receipt")
        self._project(stream.receipt)

    async def save_file(
        self,
        path: str,
        *,
        kind: str,
        tags: list[str] | None = None,
        mime: str | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        suggested_uri: str | None = None,
        name: str | None = None,
        pin: bool = False,
        cleanup: bool = True,
    ) -> Artifact:
        """Persist one local source file and return its public Artifact DTO.

        Public labels and tags become canonical immutable content metadata; metrics
        remain occurrence data. A suggested URI contributes only a descriptive filename.

        Examples:
            Save generated text:
                ```python
                artifact = await artifacts.save_file(path, kind="report")
                ```

            Save and retain a named dataset:
                ```python
                artifact = await artifacts.save_file(
                    path,
                    kind="dataset",
                    name="data.csv",
                    tags=["verified"],
                    pin=True,
                )
                ```

        Args:
            path: Existing local producer file consumed by canonical persistence.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            mime: Optional exact media type; inferred from filename when absent.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive filename.
            pin: Whether to create explicit pinned retention intent.
            cleanup: Delete only the exact source file after complete success.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed occurrence.

        Notes:
            Provider locators and deprecated App/client labels are never accepted from
            `suggested_uri` or persisted into public metadata.
        """
        filename = _filename(name=name, suggested_uri=suggested_uri)
        content_labels = _content_labels(tags=tags, labels=labels, filename=filename)
        receipt = await self.canonical.save_file(
            path,
            kind=kind,
            media_type=mime or _media_type(filename or path),
            original_filename=filename,
            content_labels=content_labels,
            metrics=metrics,
            pinned=pin,
            cleanup=cleanup,
        )
        return self._project(receipt)

    async def save_text(
        self,
        payload: str,
        *,
        suggested_uri: str | None = None,
        name: str | None = None,
        kind: str = "text",
        tags: list[str] | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        pin: bool = False,
    ) -> Artifact:
        """Persist UTF-8 text and return its public Artifact DTO.

        Text streams directly to the canonical provider and its exact searchable
        projection; no caller-visible temporary file or alternate store is used.

        Examples:
            Save plain text:
                ```python
                artifact = await artifacts.save_text("hello")
                ```

            Save a named report:
                ```python
                artifact = await artifacts.save_text(
                    report,
                    name="report.txt",
                    tags=["final"],
                    pin=True,
                )
                ```

        Args:
            payload: Exact text encoded as UTF-8.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive filename.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            pin: Whether to create explicit pinned retention intent.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed occurrence.

        Notes:
            Deprecated App identity is added only to the returned DTO projection.
        """
        filename = _filename(name=name, suggested_uri=suggested_uri)
        receipt = await self.canonical.save_text(
            payload,
            kind=kind,
            original_filename=filename,
            content_labels=_content_labels(tags=tags, labels=labels, filename=filename),
            metrics=metrics,
            pinned=pin,
        )
        return self._project(receipt)

    async def save_json(
        self,
        payload: Mapping[str, Any],
        *,
        suggested_uri: str | None = None,
        name: str | None = None,
        kind: str = "json",
        tags: list[str] | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        pin: bool = False,
    ) -> Artifact:
        """Persist deterministic JSON and return its public Artifact DTO.

        The canonical service performs strict finite JSON encoding and one coherent
        content, occurrence, search, retention, and counter commit sequence.

        Examples:
            Save a JSON result:
                ```python
                artifact = await artifacts.save_json({"status": "ok"})
                ```

            Save a named manifest:
                ```python
                artifact = await artifacts.save_json(
                    manifest,
                    name="manifest.json",
                    kind="manifest",
                    pin=True,
                )
                ```

        Args:
            payload: JSON-compatible top-level mapping.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive filename.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            pin: Whether to create explicit pinned retention intent.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed occurrence.

        Notes:
            Encoding or provider failures propagate without a second serialization path.
        """
        filename = _filename(name=name, suggested_uri=suggested_uri)
        receipt = await self.canonical.save_json(
            payload,
            kind=kind,
            original_filename=filename,
            content_labels=_content_labels(tags=tags, labels=labels, filename=filename),
            metrics=metrics,
            pinned=pin,
        )
        return self._project(receipt)

    async def get_by_id(self, artifact_id: str) -> Artifact | None:
        """Read one authorized public Artifact by stable identity.

        The canonical occurrence query applies exact owner and execution scope before
        projecting immutable content and retention state.

        Examples:
            Read an Artifact:
                ```python
                artifact = await artifacts.get_by_id("artifact-1")
                ```

            Detect absence:
                ```python
                assert await artifacts.get_by_id("missing") is None
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.

        Returns:
            Artifact | None: Newest authorized public occurrence or `None` when absent.

        Notes:
            A miss never consults search metadata, a provider locator, or legacy index.
        """
        return await self.canonical.get_public(
            artifact_id,
            deprecated_app_id=self._deprecated_app_id,
        )

    async def load_bytes_by_id(self, artifact_id: str) -> bytes:
        """Load exact owner-authorized Artifact bytes by identity.

        Public occurrence authorization is checked before canonical content streaming.

        Examples:
            Load bytes:
                ```python
                payload = await artifacts.load_bytes_by_id("artifact-1")
                ```

            Handle absence:
                ```python
                try:
                    await artifacts.load_bytes_by_id("missing")
                except FileNotFoundError:
                    pass
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.

        Returns:
            bytes: Complete immutable Artifact content.

        Notes:
            This bounded convenience method performs no filesystem or remote-URI fallback.
        """
        if await self.get_by_id(artifact_id) is None:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        return await self.canonical.load_bytes(artifact_id)

    async def load_text_by_id(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """Load and decode exact owner-authorized Artifact text by identity.

        Bytes come only from canonical content and are decoded once with caller policy.

        Examples:
            Load UTF-8 text:
                ```python
                text = await artifacts.load_text_by_id("artifact-1")
                ```

            Select an encoding:
                ```python
                text = await artifacts.load_text_by_id(
                    "artifact-1", encoding="utf-16"
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            encoding: Text codec used for one decode.
            errors: Exact decode error policy.

        Returns:
            str: Decoded immutable Artifact content.

        Notes:
            Decode errors propagate and never select another codec.
        """
        return (await self.load_bytes_by_id(artifact_id)).decode(encoding, errors=errors)

    async def load_json_by_id(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        """Load and parse exact owner-authorized Artifact JSON by identity.

        Canonical bytes are decoded once and parsed once through strict JSON loading.

        Examples:
            Load JSON:
                ```python
                value = await artifacts.load_json_by_id("artifact-1")
                ```

            Select an encoding:
                ```python
                value = await artifacts.load_json_by_id(
                    "artifact-1", encoding="utf-16"
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            encoding: Text codec used for one decode.
            errors: Exact decode error policy.

        Returns:
            Any: Parsed JSON value.

        Notes:
            Decode and JSON errors propagate without a text or bytes fallback.
        """
        return json.loads(await self.load_text_by_id(artifact_id, encoding=encoding, errors=errors))

    async def load_bytes(self, uri: str) -> bytes:
        """Load canonical Artifact bytes from an AG public URI.

        The URI is decoded to one public Artifact identity, then the same authorized
        identity loader performs canonical content access.

        Examples:
            Load an API content URI:
                ```python
                payload = await artifacts.load_bytes(
                    "/api/v1/artifacts/artifact-1/content"
                )
                ```

            Load an Artifact resource URI:
                ```python
                payload = await artifacts.load_bytes("artifact://artifact-1")
                ```

        Args:
            uri: Exact AG content-route or `artifact://` public identity URI.

        Returns:
            bytes: Complete immutable Artifact content.

        Notes:
            Filesystem, HTTP fetch, S3, and provider blob locators are rejected rather
            than treated as alternate content backends.
        """
        return await self.load_bytes_by_id(_artifact_id(uri))

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """Load canonical Artifact text from an AG public URI.

        The URI selects one public identity and bytes are decoded exactly once.

        Examples:
            Load API-routed text:
                ```python
                text = await artifacts.load_text(
                    "/api/v1/artifacts/artifact-1/content"
                )
                ```

            Load resource-routed text:
                ```python
                text = await artifacts.load_text("artifact://artifact-1")
                ```

        Args:
            uri: Exact AG content-route or `artifact://` public identity URI.
            encoding: Text codec used for one decode.
            errors: Exact decode error policy.

        Returns:
            str: Decoded immutable Artifact content.

        Notes:
            Invalid public URIs and decode failures propagate without path fallback.
        """
        return await self.load_text_by_id(
            _artifact_id(uri),
            encoding=encoding,
            errors=errors,
        )

    async def load_json(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        """Load canonical Artifact JSON from an AG public URI.

        The URI selects one public identity before strict byte decoding and JSON parsing.

        Examples:
            Load API-routed JSON:
                ```python
                value = await artifacts.load_json(
                    "/api/v1/artifacts/artifact-1/content"
                )
                ```

            Load resource-routed JSON:
                ```python
                value = await artifacts.load_json("artifact://artifact-1")
                ```

        Args:
            uri: Exact AG content-route or `artifact://` public identity URI.
            encoding: Text codec used for one decode.
            errors: Exact decode error policy.

        Returns:
            Any: Parsed JSON value.

        Notes:
            Invalid public URIs, decode errors, and JSON errors propagate directly.
        """
        return json.loads(await self.load_text(uri, encoding=encoding, errors=errors))

    async def search_public_artifacts(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int = 10,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        require_indexed_cursor: str | None = None,
    ) -> tuple[PublicArtifactSearchHit, ...]:
        """Search and hydrate ranked public Artifacts in one exact mode.

        Provider search applies owner scope and filters before bounded occurrence,
        immutable-content, and retention hydration in exact provider rank order.

        Examples:
            Search lexically:
                ```python
                hits = await artifacts.search_public_artifacts(
                    query="migration",
                    mode=SearchMode.LEXICAL,
                )
                ```

            Require tags and freshness:
                ```python
                hits = await artifacts.search_public_artifacts(
                    query="approved",
                    mode=SearchMode.SEMANTIC,
                    tags=["verified"],
                    require_indexed_cursor=cursor,
                )
                ```

        Args:
            query: Exact search text; structural mode may use an empty value.
            mode: Required canonical search mode with no inference or fallback.
            top_k: Positive hydrated result bound.
            tags: Optional tags every indexed Artifact must contain.
            metadata: Optional exact canonical search metadata filters.
            require_indexed_cursor: Optional opaque covering search cursor requirement.

        Returns:
            tuple[PublicArtifactSearchHit, ...]: Provider-ranked hydrated public results.

        Notes:
            Deprecated App metadata is response-only; stale search projections fail
            through the canonical integrity contract.
        """
        return await self.canonical.search_public_artifacts(
            query=query,
            mode=mode,
            top_k=top_k,
            tags=_tags(tags),
            metadata=metadata,
            require_indexed_cursor=require_indexed_cursor,
            deprecated_app_id=self._deprecated_app_id,
        )

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        """Set canonical retention intent for one owner-authorized Artifact.

        The provider performs revision-CAS retention updates and returns only after the
        requested current intent is authoritative.

        Examples:
            Pin content:
                ```python
                await artifacts.pin("artifact-1")
                ```

            Unpin content:
                ```python
                await artifacts.pin("artifact-1", pinned=False)
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            pinned: Requested retention intent.

        Returns:
            None: Complete when canonical retention state reflects the request.

        Notes:
            The method does not mutate immutable content or a legacy Artifact index.
        """
        await self.canonical.pin(artifact_id, pinned)

    def _project(self, receipt: ArtifactCommitReceipt) -> Artifact:
        artifact = self.canonical.project_commit(
            receipt,
            deprecated_app_id=self._deprecated_app_id,
        )
        self.last_artifact = artifact
        return artifact


def _content_labels(
    *,
    tags: Sequence[str] | None,
    labels: Mapping[str, Any] | None,
    filename: str | None,
) -> dict[str, Any]:
    result = dict(labels or {})
    normalized_tags = _tags(tags)
    if normalized_tags:
        existing = result.get("tags")
        if existing is not None and existing != list(normalized_tags):
            raise ValueError("labels.tags conflicts with explicit tags")
        result["tags"] = list(normalized_tags)
    if filename is not None:
        existing_filename = result.get("filename")
        if existing_filename is not None and existing_filename != filename:
            raise ValueError("labels.filename conflicts with explicit name")
        result["filename"] = filename
    return result


def _tags(values: Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("tags must be a sequence of exact strings, not a string")
    result = tuple(dict.fromkeys(values or ()))
    if any(not isinstance(tag, str) or not tag.strip() or tag != tag.strip() for tag in result):
        raise ValueError("tags must contain exact non-empty strings")
    return result


def _filename(*, name: str | None, suggested_uri: str | None) -> str | None:
    if name is not None:
        _exact_filename(name)
        return name
    if suggested_uri is None:
        return None
    if not isinstance(suggested_uri, str) or not suggested_uri.strip():
        raise ValueError("suggested_uri must be a non-empty string when supplied")
    parsed = urlparse(suggested_uri)
    candidate = PurePosixPath(parsed.path.replace("\\", "/")).name
    if not candidate:
        raise ValueError("suggested_uri must contain a descriptive filename")
    _exact_filename(candidate)
    return candidate


def _exact_filename(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or PurePosixPath(value.replace("\\", "/")).name != value
    ):
        raise ValueError("name must be an exact filename without path components")


def _media_type(value: str | None) -> str:
    guessed = mimetypes.guess_type(value or "")[0]
    return guessed or "application/octet-stream"


def _artifact_id(uri: str) -> str:
    if not isinstance(uri, str) or not uri.strip() or uri != uri.strip():
        raise ValueError("Artifact URI must be a non-empty exact string")
    parsed = urlparse(uri)
    if parsed.scheme == "artifact":
        if parsed.query or parsed.fragment or parsed.username is not None:
            raise ValueError("Artifact URI must not contain query, fragment, or user info")
        identity = unquote(parsed.netloc + parsed.path)
    elif not parsed.scheme and parsed.query == "" and parsed.fragment == "":
        path = parsed.path
        if not path.startswith(_CONTENT_PREFIX) or not path.endswith(_CONTENT_SUFFIX):
            raise ValueError("Artifact URI must use the AG public content route")
        identity = unquote(path[len(_CONTENT_PREFIX) : -len(_CONTENT_SUFFIX)])
    else:
        raise ValueError("Artifact URI must use an AG public Artifact identity")
    if not identity or identity != identity.strip():
        raise ValueError("Artifact URI must contain an exact Artifact identity")
    return identity
