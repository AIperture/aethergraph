"""Inactive runtime-output projection onto one canonical provider sink."""

from __future__ import annotations

from dataclasses import replace

from aethergraph.contracts.services.runtime_output import RuntimeOutputFrame
from aethergraph.server.security.redaction import sanitize_text
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    Page,
    RuntimeOutputFrame as StorageRuntimeOutputFrame,
    RuntimeOutputQuery,
    RuntimeOutputRecord,
    RuntimeOutputSink as StorageRuntimeOutputSink,
    RuntimeOutputStream,
    StorageBundle,
    StorageScope,
)

from .runtime_output import _RuntimeOutputBounds

_DEPRECATED_IDENTITY_TAG_PREFIXES = ("app_id:", "application_id:", "client_id:")


class CanonicalRuntimeOutputSink:
    """Project bounded service frames onto one provider-owned runtime-output sink."""

    def __init__(
        self,
        *,
        sink: StorageRuntimeOutputSink,
        owner_scope: StorageScope,
        tags: tuple[str, ...] = (),
        max_line_bytes: int = 16 * 1024,
        max_run_bytes: int = 256 * 1024,
        max_rows_per_run: int = 1_000,
    ) -> None:
        """Bind runtime output to one provider-authoritative owner.

        The projection shares the established line/run limiter, redacts text, builds
        exact run/node scope, and synchronously submits canonical frames to the one
        already-open provider sink.

        Examples:
            Bind an opened bundle sink:
                ```python
                sink = CanonicalRuntimeOutputSink(
                    sink=bundle.runtime_output,
                    owner_scope=owner_scope,
                )
                ```

            Configure tighter capture bounds:
                ```python
                sink = CanonicalRuntimeOutputSink(
                    sink=fake_sink,
                    owner_scope=StorageScope(project_id="project-1"),
                    max_rows_per_run=100,
                )
                ```

        Args:
            sink: Canonical provider-owned runtime-output sink from one bundle.
            owner_scope: Exact trusted runtime ownership scope.
            tags: Unique provider-neutral output classification tags.
            max_line_bytes: Positive UTF-8 byte ceiling for one frame.
            max_run_bytes: Positive UTF-8 byte ceiling for one run.
            max_rows_per_run: Positive persisted-frame ceiling for one run.

        Returns:
            None: The inactive-until-S9 projection is ready without I/O.

        Notes:
            Bundle lifecycle owns sink closure; this service never closes a store.
        """
        validate_storage_owner_scope(owner_scope)
        _validate_bounds(max_line_bytes, max_run_bytes, max_rows_per_run)
        _validate_tags(tags)
        self._sink = sink
        self._owner_scope = owner_scope
        self._tags = ("runtime-console", *tags)
        self._bounds = _RuntimeOutputBounds(
            max_line_bytes=max_line_bytes,
            max_run_bytes=max_run_bytes,
            max_rows_per_run=max_rows_per_run,
        )

    def emit(self, frame: RuntimeOutputFrame) -> None:
        """Bound, redact, normalize, and synchronously admit one output frame.

        Empty frames removed by the shared limiter perform no provider call. Accepted
        frames receive deterministic execution/sequence identity and exact canonical
        scope before the provider's explicit capacity and integrity checks.

        Examples:
            Emit stdout:
                ```python
                sink.emit(stdout_frame)
                ```

            Emit a terminal partial frame:
                ```python
                sink.emit(replace(frame, partial=True, eof=True))
                ```

        Args:
            frame: Existing frozen runtime-capture service frame.

        Returns:
            None: The frame was omitted by bounds or accepted by the provider sink.

        Notes:
            Capacity and integrity failures propagate; no EventLog or file fallback is used.
        """
        checkpoint = self._bounds.checkpoint(frame.run_id)
        bounded = self._bounds.bounded(frame)
        if bounded is None:
            return
        dimensions = {
            "run_id": bounded.run_id,
            "node_id": bounded.node_id,
        }
        if bounded.session_id is not None:
            dimensions["session_id"] = bounded.session_id
        if bounded.graph_id is not None:
            dimensions["graph_id"] = bounded.graph_id
        scope = merge_storage_scope(self._owner_scope, **dimensions)
        try:
            self._sink.emit(
                StorageRuntimeOutputFrame(
                    output_id=f"runtime:{bounded.execution_id}:{bounded.sequence}",
                    execution_id=bounded.execution_id,
                    scope=scope,
                    stream=RuntimeOutputStream(bounded.stream),
                    sequence=bounded.sequence,
                    text=sanitize_text(bounded.text),
                    source=bounded.source,
                    tool_name=bounded.tool_name,
                    partial=bounded.partial,
                    truncated=bounded.truncated,
                    eof=bounded.eof,
                    tags=self._tags,
                )
            )
        except BaseException:
            self._bounds.restore(frame.run_id, checkpoint)
            raise

    async def flush_execution(self, execution_id: str) -> None:
        """Flush accepted frames for one exact execution through the provider.

        The provider barrier covers frames admitted before the call while other
        executions may continue. Persistence failures propagate unchanged.

        Examples:
            Flush a completed Tool:
                ```python
                await sink.flush_execution("execution-1")
                ```

            Flush capture-context teardown:
                ```python
                await sink.flush_execution(frame.execution_id)
                ```

        Args:
            execution_id: Exact stable execution identity.

        Returns:
            None: Previously accepted execution frames are durable.

        Notes:
            This method delegates only to the selected bundle sink.
        """
        await self._sink.flush_execution(execution_id)

    async def flush_run(self, run_id: str) -> None:
        """Flush accepted frames for one exact run through the provider.

        The provider barrier covers every matching execution admitted before the call
        and leaves unrelated run frames available for independent flushing.

        Examples:
            Flush before result publication:
                ```python
                await sink.flush_run("run-1")
                ```

            Flush cancellation output:
                ```python
                await sink.flush_run(canceled_run_id)
                ```

        Args:
            run_id: Exact stable run identity.

        Returns:
            None: Previously accepted run frames are durable.

        Notes:
            Bundle shutdown, not this service, owns the final all-frame close barrier.
        """
        await self._sink.flush_run(run_id)

    async def query(self, query: RuntimeOutputQuery) -> Page[RuntimeOutputRecord]:
        """Read committed runtime output through the canonical owner boundary.

        Intro:
            Merges trusted provider ownership into the requested run scope and
            delegates one bounded read to the selected provider repository.

        Examples:
            Read one run:
                ```python
                page = await output.query(
                    RuntimeOutputQuery(scope=StorageScope(run_id="run-1"))
                )
                ```

            Continue a merged semantic/output stream:
                ```python
                page = await output.query(
                    RuntimeOutputQuery(
                        scope=StorageScope(run_id="run-1"),
                        after_delivery_cursor=last_cursor,
                    )
                )
                ```

        Args:
            query: Exact runtime-output query with execution-level scope dimensions.

        Returns:
            Page[RuntimeOutputRecord]: Committed provider records in delivery order.

        Notes:
            Pending frames and legacy EventLog rows are not visible. Conflicting
            trusted ownership dimensions fail before provider I/O.
        """
        scope = merge_storage_scope(self._owner_scope, **query.scope.as_filter())
        return await self._sink.query(replace(query, scope=scope))


def bind_canonical_runtime_output(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    tags: tuple[str, ...] = (),
) -> CanonicalRuntimeOutputSink:
    """Bind runtime-output capture to the exact canonical bundle field.

    The factory performs no provider selection, filesystem access, capture-proxy
    installation, or I/O. S9 composition may install the returned focused sink into
    the existing capture boundary after the bundle is ready.

    Examples:
        Bind production composition inputs:
            ```python
            sink = bind_canonical_runtime_output(
                bundle=bundle,
                owner_scope=open_request.owner_scope,
            )
            ```

        Add deployment classification:
            ```python
            sink = bind_canonical_runtime_output(
                bundle=fake_bundle,
                owner_scope=StorageScope(project_id="project-1"),
                tags=("test-host",),
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Exact trusted runtime ownership scope.
        tags: Unique provider-neutral output classification tags.

    Returns:
        CanonicalRuntimeOutputSink: Focused bounded projection over `bundle.runtime_output`.

    Notes:
        The active EventLog capture installer remains unchanged until the S9 atomic cut.
    """
    return CanonicalRuntimeOutputSink(
        sink=bundle.runtime_output,
        owner_scope=owner_scope,
        tags=tags,
    )


def _validate_bounds(max_line_bytes: int, max_run_bytes: int, max_rows_per_run: int) -> None:
    for name, value in (
        ("max_line_bytes", max_line_bytes),
        ("max_run_bytes", max_run_bytes),
        ("max_rows_per_run", max_rows_per_run),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _validate_tags(tags: tuple[str, ...]) -> None:
    if not isinstance(tags, tuple):
        raise TypeError("tags must be an immutable tuple")
    if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("tags must contain non-empty strings")
    all_tags = ("runtime-console", *tags)
    if len(set(all_tags)) != len(all_tags):
        raise ValueError("tags must not contain duplicates")
    deprecated = tuple(
        tag for tag in tags if tag.casefold().startswith(_DEPRECATED_IDENTITY_TAG_PREFIXES)
    )
    if deprecated:
        raise ValueError("deprecated App/client identity is not a runtime-output tag")
