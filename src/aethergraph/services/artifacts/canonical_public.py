"""Stable public Artifact behavior over one canonical artifact facade."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
import ipaddress
import json
import mimetypes
from pathlib import Path, PurePosixPath
import socket
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import warnings

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.storage.contracts import (
    ArtifactMetricOrder,
    Page,
    PageRequest,
    SearchMode,
    StorageScope,
)

from .canonical_facade import (
    ArtifactCommitReceipt,
    CanonicalArtifactFacade,
    CanonicalArtifactWriter,
    PublicArtifactSearchHit,
)
from .types import ArtifactContent

_CONTENT_PREFIX = "/api/v1/artifacts/"
_CONTENT_SUFFIX = "/content"
_DEFAULT_READ_LIMIT = 64 * 1024 * 1024
_MAX_PUBLIC_PAGE_SIZE = 500


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote artifact URL must be absolute HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("remote artifact URL must not contain credentials")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except OSError as exc:
        raise ValueError("remote artifact host could not be resolved") from exc
    if not addresses:
        raise ValueError("remote artifact host resolved to no addresses")
    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        if not address.is_global:
            raise ValueError("remote artifact URL resolves to a non-public address")
    return url


def _download_remote_file(
    url: str,
    *,
    timeout_s: float,
    max_bytes: int,
) -> tuple[bytes, str, str, str]:
    _validate_remote_url(url)
    opener = build_opener(_ValidatedRedirectHandler())
    request = Request(url, headers={"User-Agent": "AetherGraph-Artifact/1"})
    try:
        with opener.open(request, timeout=timeout_s) as response:
            final_url = response.geturl()
            _validate_remote_url(final_url)
            status = getattr(response, "status", 200)
            if status < 200 or status >= 300:
                raise ValueError(f"remote artifact download returned HTTP {status}")
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except ValueError as exc:
                    raise ValueError("remote artifact Content-Length is invalid") from exc
                if content_length < 0 or content_length > max_bytes:
                    raise ValueError("remote artifact exceeds max_bytes")
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ValueError("remote artifact exceeds max_bytes")
            media_type = response.headers.get_content_type() or "application/octet-stream"
            filename = response.headers.get_filename()
    except HTTPError as exc:
        raise ValueError(f"remote artifact download returned HTTP {exc.code}") from exc

    parsed_final = urlparse(final_url)
    if not filename:
        filename = PurePosixPath(unquote(parsed_final.path)).name or "download.bin"
    filename = Path(filename).name
    if not filename:
        filename = "download.bin"
    source = urlunparse((parsed_final.scheme, parsed_final.netloc, parsed_final.path, "", "", ""))
    return payload, media_type, filename, source


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

    async def save_directory(
        self,
        path: str,
        *,
        kind: str = "directory",
        tags: list[str] | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        suggested_uri: str | None = None,
        name: str | None = None,
        pin: bool = False,
        cleanup: bool = False,
        max_entries: int = 10_000,
        max_total_bytes: int = 1024 * 1024 * 1024,
    ) -> Artifact:
        """Persist one directory as a bounded deterministic canonical archive.

        The canonical service validates the source tree, rejects links and special
        files, normalizes archive metadata, and commits one immutable Artifact.

        Examples:
            Save a generated directory:
                ```python
                artifact = await artifacts.save_directory("build", kind="bundle")
                ```

            Consume an explicitly staged directory:
                ```python
                artifact = await artifacts.save_directory(
                    staged,
                    tags=["final"],
                    cleanup=True,
                    pin=True,
                )
                ```

        Args:
            path: Existing non-linked local source directory.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive archive filename.
            pin: Whether to create explicit pinned retention intent.
            cleanup: Delete only the exact source directory after complete success.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum total regular-file source bytes.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed directory occurrence.

        Notes:
            Directory content uses canonical `tar.v1`; provider paths and source
            filesystem metadata never enter public identity.
        """
        filename = _filename(name=name, suggested_uri=suggested_uri)
        receipt = await self.canonical.save_directory(
            path,
            kind=kind,
            original_filename=filename,
            content_labels=_content_labels(tags=tags, labels=labels, filename=filename),
            metrics=metrics,
            pinned=pin,
            cleanup=cleanup,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
        )
        return self._project(receipt)

    async def ingest_file(
        self,
        staged_path: str,
        *,
        kind: str,
        tags: list[str] | None = None,
        mime: str | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        suggested_uri: str | None = None,
        name: str | None = None,
        pin: bool = False,
    ) -> Artifact:
        """Consume one staged file through canonical file persistence.

        This compatibility name delegates exactly once to `save_file` with successful
        source cleanup and never selects a legacy Artifact store.

        Examples:
            Ingest staged output:
                ```python
                artifact = await artifacts.ingest_file(staged, kind="report")
                ```

            Ingest labeled output:
                ```python
                artifact = await artifacts.ingest_file(
                    staged,
                    kind="dataset",
                    tags=["verified"],
                )
                ```

        Args:
            staged_path: Existing local producer file consumed after success.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            mime: Optional exact media type; inferred from filename when absent.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive filename.
            pin: Whether to create explicit pinned retention intent.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed occurrence.

        Notes:
            `ingest_file` is deprecated compatibility vocabulary; use `save_file`
            with `cleanup=True`. The alias has no independent persistence behavior.
        """
        warnings.warn(
            "CanonicalPublicArtifactFacade.ingest_file() is deprecated; "
            "use save_file(..., cleanup=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.save_file(
            staged_path,
            kind=kind,
            tags=tags,
            mime=mime,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            name=name,
            pin=pin,
            cleanup=True,
        )

    async def ingest_dir(
        self,
        staged_dir: str,
        *,
        kind: str = "directory",
        tags: list[str] | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        suggested_uri: str | None = None,
        name: str | None = None,
        pin: bool = False,
        max_entries: int = 10_000,
        max_total_bytes: int = 1024 * 1024 * 1024,
    ) -> Artifact:
        """Consume one staged directory through canonical directory persistence.

        This compatibility name delegates exactly once to `save_directory` with
        successful source cleanup and deterministic archive construction.

        Examples:
            Ingest staged output:
                ```python
                artifact = await artifacts.ingest_dir(staged)
                ```

            Ingest a bounded dataset:
                ```python
                artifact = await artifacts.ingest_dir(
                    staged,
                    kind="dataset",
                    max_entries=500,
                )
                ```

        Args:
            staged_dir: Existing non-linked local source directory consumed after success.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            suggested_uri: Optional legacy descriptive name source, never a locator.
            name: Optional exact descriptive archive filename.
            pin: Whether to create explicit pinned retention intent.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum total regular-file source bytes.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed directory occurrence.

        Notes:
            `ingest_dir` is deprecated compatibility vocabulary; use
            `save_directory(..., cleanup=True)`.
        """
        warnings.warn(
            "CanonicalPublicArtifactFacade.ingest_dir() is deprecated; "
            "use save_directory(..., cleanup=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.save_directory(
            staged_dir,
            kind=kind,
            tags=tags,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            name=name,
            pin=pin,
            cleanup=True,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
        )

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

    async def save_url(
        self,
        url: str,
        *,
        kind: str,
        tags: list[str] | None = None,
        mime: str | None = None,
        labels: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        name: str | None = None,
        pin: bool = False,
        timeout_s: float = 30.0,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> Artifact:
        """Download one bounded public HTTP(S) resource into canonical storage.

        Examples:
            Save a remote report:
                ```python
                artifact = await artifacts.save_url(
                    "https://example.test/report.csv",
                    kind="dataset",
                    name="report.csv",
                )
                ```

            Bound a generated-image copy:
                ```python
                artifact = await artifacts.save_url(
                    provider_url,
                    kind="image",
                    max_bytes=16 * 1024 * 1024,
                )
                ```

        Args:
            url: Absolute public HTTP(S) source without embedded credentials.
            kind: Exact canonical Artifact kind.
            tags: Optional unique public content tags.
            mime: Optional safe media-type hint when the response is generic.
            labels: Optional immutable public content labels.
            metrics: Optional finite occurrence metrics.
            name: Optional exact descriptive filename.
            pin: Whether to create explicit pinned retention intent.
            timeout_s: Positive total socket timeout for the bounded request.
            max_bytes: Positive maximum response body size.

        Returns:
            Artifact: Frozen public Artifact DTO for the committed response body.

        Notes:
            Redirect targets are revalidated. Loopback, private, link-local, multicast,
            reserved, and credential-bearing targets fail before content persistence.
            Query strings and fragments never enter public Artifact metadata.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        payload, response_mime, response_name, source = await asyncio.to_thread(
            _download_remote_file,
            url,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
        )
        filename = _filename(name=name, suggested_uri=response_name)
        media_type = response_mime
        if media_type == "application/octet-stream" and mime:
            media_type = mime
        content_labels = _content_labels(
            tags=tags,
            labels={**(labels or {}), "source_url": source},
            filename=filename,
        )
        async with self.writer(
            kind=kind,
            planned_ext=Path(filename or "").suffix or None,
            mime=media_type,
            pin=pin,
        ) as writer:
            writer.add_labels(content_labels)
            writer.add_metrics(metrics or {})
            await writer.write(payload)
        if self.last_artifact is None:
            raise RuntimeError("canonical Artifact writer completed without a public projection")
        return self.last_artifact

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

    async def get_by_id(
        self,
        artifact_id: str,
        *,
        occurrence_scope: StorageScope | None = None,
    ) -> Artifact | None:
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

            Authorize through a root-run attachment occurrence:
                ```python
                artifact = await artifacts.get_by_id(
                    "artifact-1",
                    occurrence_scope=StorageScope(run_id="run-1"),
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            occurrence_scope: Optional canonical occurrence filter. When omitted,
                the facade's exact execution scope is required.

        Returns:
            Artifact | None: Newest authorized public occurrence or `None` when absent.

        Notes:
            A miss never consults search metadata, a provider locator, or legacy index.
        """
        return await self.canonical.get_public(
            artifact_id,
            scope=occurrence_scope,
            deprecated_app_id=self._deprecated_app_id,
        )

    async def load_bytes_by_id(
        self,
        artifact_id: str,
        *,
        occurrence_scope: StorageScope | None = None,
    ) -> bytes:
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

            Load a run-adopted input from a downstream node:
                ```python
                payload = await artifacts.load_bytes_by_id(
                    "artifact-1",
                    occurrence_scope=StorageScope(run_id="run-1"),
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            occurrence_scope: Optional canonical occurrence filter. When omitted,
                the facade's exact execution scope is required.

        Returns:
            bytes: Complete immutable Artifact content.

        Notes:
            This bounded convenience method performs no filesystem or remote-URI fallback.
        """
        if (
            await self.get_by_id(
                artifact_id,
                occurrence_scope=occurrence_scope,
            )
            is None
        ):
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

    async def load_content(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        max_bytes: int = _DEFAULT_READ_LIMIT,
    ) -> ArtifactContent:
        """Load one Artifact through an explicit canonical content-mode policy.

        JSON and text classification uses only immutable kind and media-type metadata;
        every other Artifact remains bytes. Canonical hydration enforces the byte bound.

        Examples:
            Load structured content:
                ```python
                content = await artifacts.load_content("artifact-1")
                assert content.mode == "json"
                ```

            Apply a smaller hydration bound:
                ```python
                content = await artifacts.load_content(
                    "artifact-2",
                    max_bytes=1024,
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            encoding: Text codec used for JSON or text decoding.
            errors: Exact decode error policy.
            max_bytes: Positive maximum complete content bytes permitted in memory.

        Returns:
            ArtifactContent: Typed JSON, text, or complete bytes with its public DTO.

        Notes:
            Unknown kinds and media types are bytes by explicit policy. Content is
            never truncated, reclassified by filename, or retried through another backend.
        """
        artifact = await self.get_by_id(artifact_id)
        if artifact is None:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        payload = await self.canonical.load_bytes(artifact_id, max_bytes=max_bytes)
        mode = _content_mode(artifact)
        if mode == "json":
            return ArtifactContent(
                artifact=artifact,
                mode="json",
                json=json.loads(payload.decode(encoding, errors=errors)),
            )
        if mode == "text":
            return ArtifactContent(
                artifact=artifact,
                mode="text",
                text=payload.decode(encoding, errors=errors),
            )
        return ArtifactContent(artifact=artifact, mode="bytes", data=payload)

    async def as_local_file_by_id(
        self,
        artifact_id: str,
        *,
        max_bytes: int = _DEFAULT_READ_LIMIT,
    ) -> str:
        """Materialize one bounded canonical Artifact into a new staging file.

        The public identity is authorized before canonical bytes are copied into a
        service-owned transient file; no provider locator becomes caller-visible.

        Examples:
            Materialize a report:
                ```python
                path = await artifacts.as_local_file_by_id("artifact-1")
                ```

            Apply a smaller content bound:
                ```python
                path = await artifacts.as_local_file_by_id(
                    "artifact-2",
                    max_bytes=1_000_000,
                )
                ```

        Args:
            artifact_id: Exact stable canonical Artifact identity.
            max_bytes: Positive maximum complete content bytes permitted locally.

        Returns:
            str: Absolute path to a new complete transient file.

        Notes:
            The caller owns the returned transient file. URI/path-based localizers and
            direct provider paths are intentionally not part of this facade.
        """
        artifact = await self.get_by_id(artifact_id)
        if artifact is None:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        payload = await self.canonical.load_bytes(artifact_id, max_bytes=max_bytes)
        suffix = _artifact_suffix(artifact)
        staged = Path(await self.stage_path(suffix))
        completed = False
        try:
            await asyncio.to_thread(staged.write_bytes, payload)
            completed = True
        finally:
            if not completed:
                await asyncio.to_thread(staged.unlink, missing_ok=True)
        return str(staged.resolve(strict=True))

    async def materialize_directory(
        self,
        artifact_id: str,
        destination: str,
        *,
        max_entries: int = 10_000,
        max_total_bytes: int = 1024 * 1024 * 1024,
        max_archive_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> str:
        """Safely materialize one canonical directory into an explicit destination.

        Public occurrence authorization precedes bounded canonical archive extraction.
        The destination must be new and its existing parent remains caller-controlled.

        Examples:
            Materialize a directory:
                ```python
                path = await artifacts.materialize_directory(
                    "artifact-1", "output"
                )
                ```

            Apply tighter extraction bounds:
                ```python
                path = await artifacts.materialize_directory(
                    "artifact-1",
                    "output",
                    max_entries=100,
                    max_total_bytes=10_000_000,
                )
                ```

        Args:
            artifact_id: Exact stable canonical directory Artifact identity.
            destination: New local directory path beneath an existing parent.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum extracted regular-file bytes.
            max_archive_bytes: Positive maximum transient archive bytes.

        Returns:
            str: Absolute path to the completely materialized new directory.

        Notes:
            Existing destinations are never merged or overwritten. Extraction failure
            removes only the new destination created by this call.
        """
        if await self.get_by_id(artifact_id) is None:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        return await self.canonical.materialize_directory(
            artifact_id,
            destination,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
            max_archive_bytes=max_archive_bytes,
        )

    async def as_local_dir_by_id(
        self,
        artifact_id: str,
        *,
        max_entries: int = 10_000,
        max_total_bytes: int = 1024 * 1024 * 1024,
        max_archive_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> str:
        """Materialize one canonical directory into new transient staging.

        A service-owned parent is allocated and canonical extraction creates one new
        child destination without exposing archive or provider locations.

        Examples:
            Materialize directory content:
                ```python
                path = await artifacts.as_local_dir_by_id("artifact-1")
                ```

            Apply tighter extraction bounds:
                ```python
                path = await artifacts.as_local_dir_by_id(
                    "artifact-1",
                    max_entries=100,
                )
                ```

        Args:
            artifact_id: Exact stable canonical directory Artifact identity.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum extracted regular-file bytes.
            max_archive_bytes: Positive maximum transient archive bytes.

        Returns:
            str: Absolute path to a completely materialized transient directory.

        Notes:
            The caller owns the returned staging parent and content. Arbitrary URI/path
            directory conversion is intentionally retired.
        """
        parent = Path(await self.stage_dir("-materialized"))
        destination = parent / "content"
        completed = False
        try:
            result = await self.materialize_directory(
                artifact_id,
                str(destination),
                max_entries=max_entries,
                max_total_bytes=max_total_bytes,
                max_archive_bytes=max_archive_bytes,
            )
            completed = True
            return result
        finally:
            if not completed:
                await asyncio.to_thread(parent.rmdir)

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

    async def query_public_artifacts(
        self,
        page: PageRequest | None = None,
        *,
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        labels: Mapping[str, Any] | None = None,
        pinned: bool | None = None,
        metric: str | None = None,
        metric_order: ArtifactMetricOrder | None = None,
    ) -> Page[Artifact]:
        """Query one bounded cursor page in the exact bound execution scope.

        The canonical repository applies structured filters before bounded batch
        hydration and returns its opaque continuation cursor unchanged.

        Examples:
            Query recent public Artifacts:
                ```python
                page = await artifacts.query_public_artifacts()
                ```

            Continue a filtered query:
                ```python
                page = await artifacts.query_public_artifacts(
                    PageRequest(limit=25, cursor=cursor),
                    kind="report",
                    tags=["final"],
                )
                ```

        Args:
            page: Optional bounded opaque cursor request, capped at 500 records.
            kind: Optional exact immutable Artifact kind.
            tags: Optional tags every immutable Artifact must contain.
            labels: Optional exact immutable content-label filters.
            pinned: Optional current retention-state filter.
            metric: Optional exact occurrence metric key used for indexed ranking.
            metric_order: Required maximum/minimum ranking direction with `metric`.

        Returns:
            Page[Artifact]: Frozen public DTOs and the exact provider cursor.

        Notes:
            Scope is fixed at construction. Deprecated App metadata is projection-only,
            and no offset pagination or legacy Artifact index is available.
        """
        requested = page or PageRequest()
        if requested.limit > _MAX_PUBLIC_PAGE_SIZE:
            raise ValueError(
                f"public Artifact page limit must be between 1 and {_MAX_PUBLIC_PAGE_SIZE}"
            )
        return await self.canonical.query_public_artifacts(
            requested,
            tags=_tags(tags),
            kind=kind,
            labels=labels,
            pinned=pinned,
            metric=metric,
            metric_order=metric_order,
            deprecated_app_id=self._deprecated_app_id,
        )

    async def list(
        self,
        *,
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        pinned: bool | None = None,
        limit: int = 100,
    ) -> list[Artifact]:
        """List one bounded first page in the exact bound execution scope.

        This convenience projection performs one structured canonical query and returns
        its items; callers needing continuation use `query_public_artifacts` directly.

        Examples:
            List recent Artifacts:
                ```python
                rows = await artifacts.list()
                ```

            List pinned reports:
                ```python
                rows = await artifacts.list(
                    kind="report",
                    tags=["final"],
                    pinned=True,
                    limit=25,
                )
                ```

        Args:
            kind: Optional exact immutable Artifact kind.
            tags: Optional tags every immutable Artifact must contain.
            filters: Optional exact immutable content-label filters.
            pinned: Optional current retention-state filter.
            limit: Positive result bound, at most 500.

        Returns:
            list[Artifact]: Frozen public DTOs from the first provider page.

        Notes:
            Legacy level and node widening are retired because this facade is already
            bound to one exact execution scope; results never cross that boundary.
        """
        page = await self.query_public_artifacts(
            PageRequest(limit=limit),
            kind=kind,
            tags=tags,
            labels=filters,
            pinned=pinned,
        )
        return list(page.items)

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        metric_mode: Literal["max", "min"],
        tags: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        pinned: bool | None = None,
    ) -> Artifact | None:
        """Return the best exact-scope Artifact for one occurrence metric.

        Provider-side metric ranking and structured filtering occur before one-record
        public hydration; ties follow the repository's stable occurrence order.

        Examples:
            Select maximum quality:
                ```python
                artifact = await artifacts.best(
                    kind="model",
                    metric="quality",
                    metric_mode="max",
                )
                ```

            Select minimum loss with tags:
                ```python
                artifact = await artifacts.best(
                    kind="model",
                    metric="loss",
                    metric_mode="min",
                    tags=["validated"],
                )
                ```

        Args:
            kind: Exact immutable Artifact kind.
            metric: Exact occurrence metric key.
            metric_mode: Required `max` or `min` provider ranking direction.
            tags: Optional tags every immutable Artifact must contain.
            filters: Optional exact immutable content-label filters.
            pinned: Optional current retention-state filter.

        Returns:
            Artifact | None: Best matching frozen DTO, or `None` when no row matches.

        Notes:
            Ranking has no client-side scan, numeric coercion, default direction, or
            fallback to creation order.
        """
        if metric_mode == "max":
            order = ArtifactMetricOrder.MAXIMUM
        elif metric_mode == "min":
            order = ArtifactMetricOrder.MINIMUM
        else:
            raise ValueError("metric_mode must be exactly 'max' or 'min'")
        page = await self.query_public_artifacts(
            PageRequest(limit=1),
            kind=kind,
            tags=tags,
            labels=filters,
            pinned=pinned,
            metric=metric,
            metric_order=order,
        )
        return page.items[0] if page.items else None

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


def _content_mode(artifact: Artifact) -> Literal["json", "text", "bytes"]:
    kind = artifact.kind or ""
    media_type = artifact.mime or ""
    if kind == "json" or media_type == "application/json" or media_type.endswith("+json"):
        return "json"
    if kind in {"text", "log", "note"} or media_type.startswith("text/"):
        return "text"
    return "bytes"


def _artifact_suffix(artifact: Artifact) -> str:
    labels = artifact.labels or {}
    filename = labels.get("filename")
    if filename is None:
        return ""
    _exact_filename(filename)
    return PurePosixPath(filename).suffix


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
