from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from .models import (
    LLMObservationRecord,
    ObservationFilter,
    ObservationRecord,
    PurgeResult,
    StorageStats,
    utc_now_iso,
)
from .policy import ObservationPolicy
from .prompt_store import PreparedPromptCapture, PromptStore, canonical_json

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS content_fragments (
    fragment_id TEXT PRIMARY KEY,
    content_kind TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    app_id TEXT,
    session_id TEXT,
    run_id TEXT,
    trace_id TEXT,
    agent_id TEXT,
    graph_id TEXT,
    node_id TEXT,
    turn_id TEXT,
    parent_observation_id TEXT,
    caused_by_observation_id TEXT,
    source_event_id TEXT,
    llm_call_id TEXT,
    summary TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    payload_fragment_id TEXT REFERENCES content_fragments(fragment_id),
    retention_class TEXT NOT NULL,
    expires_at TEXT
);

CREATE INDEX IF NOT EXISTS ix_observations_occurred ON observations(occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_observations_category ON observations(category, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_observations_run ON observations(run_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_observations_session ON observations(session_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_observations_trace ON observations(trace_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_observations_llm_call ON observations(llm_call_id);

CREATE TABLE IF NOT EXISTS observation_resource_links (
    observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    resource_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    revision TEXT,
    content_hash TEXT,
    slot_key TEXT,
    artifact_id TEXT,
    PRIMARY KEY (observation_id, resource_key, relation)
);
CREATE INDEX IF NOT EXISTS ix_observation_resources_key
    ON observation_resource_links(resource_key, relation);

CREATE TABLE IF NOT EXISTS llm_calls (
    llm_call_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL UNIQUE REFERENCES observations(observation_id) ON DELETE CASCADE,
    call_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    profile_name TEXT,
    call_name TEXT,
    reasoning_effort TEXT,
    max_output_tokens INTEGER,
    output_format TEXT,
    json_schema_json TEXT,
    schema_name TEXT,
    strict_schema INTEGER,
    validate_json INTEGER,
    extra_params_json TEXT NOT NULL,
    request_args_json TEXT NOT NULL,
    provider_request_args_json TEXT NOT NULL,
    compatibility_notes_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    latency_ms INTEGER,
    error_type TEXT,
    error_message TEXT,
    capture_mode TEXT NOT NULL,
    prompt_manifest_id TEXT,
    response_fragment_id TEXT REFERENCES content_fragments(fragment_id)
);

CREATE TABLE IF NOT EXISTS llm_call_attempts (
    llm_call_id TEXT NOT NULL REFERENCES llm_calls(llm_call_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    retryable INTEGER NOT NULL,
    status_code INTEGER,
    error_code TEXT,
    request_id TEXT,
    provider_delay_ms INTEGER,
    scheduled_delay_ms INTEGER,
    rate_limits_json TEXT NOT NULL,
    PRIMARY KEY (llm_call_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS ix_llm_call_attempts_call
    ON llm_call_attempts(llm_call_id, attempt_number);

CREATE TABLE IF NOT EXISTS prompt_manifests (
    manifest_id TEXT PRIMARY KEY,
    llm_call_id TEXT NOT NULL UNIQUE REFERENCES llm_calls(llm_call_id) ON DELETE CASCADE,
    renderer_version TEXT NOT NULL,
    capture_mode TEXT NOT NULL,
    assembled_request_hash TEXT NOT NULL,
    total_chars INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL,
    roles_json TEXT NOT NULL,
    omission_reason TEXT
);

CREATE TABLE IF NOT EXISTS prompt_manifest_parts (
    manifest_id TEXT NOT NULL REFERENCES prompt_manifests(manifest_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    semantic_kind TEXT NOT NULL,
    role TEXT,
    fragment_id TEXT NOT NULL REFERENCES content_fragments(fragment_id),
    source_event_id TEXT,
    PRIMARY KEY (manifest_id, ordinal)
);
CREATE INDEX IF NOT EXISTS ix_prompt_parts_fragment ON prompt_manifest_parts(fragment_id);

CREATE TABLE IF NOT EXISTS trace_management (
    scope_key TEXT PRIMARY KEY,
    tenant_id TEXT,
    project_id TEXT,
    app_id TEXT,
    session_id TEXT,
    run_id TEXT,
    trace_id TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    label TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    retention_class TEXT NOT NULL DEFAULT 'standard',
    expires_at TEXT,
    hidden INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
"""

_SCOPE_COLUMNS = (
    "tenant_id",
    "project_id",
    "org_id",
    "user_id",
    "app_id",
    "session_id",
    "run_id",
    "trace_id",
    "agent_id",
    "graph_id",
    "node_id",
    "turn_id",
)
_RESOURCE_RELATIONS = {
    "input",
    "output",
    "read",
    "created",
    "updated",
    "derived_from",
    "supersedes",
    "invalidates",
    "mentions",
}


class ObservationStoreError(RuntimeError):
    pass


class SQLiteObservationStore:
    """Persist canonical AG observations in one SQLite database.

    Intro:
        Owns schema, transactions, scoped queries, deletion, and fragment GC.

    Examples:
        Create a writable store:
        ```python
        store = SQLiteObservationStore("events/observability.db")
        ```

        Open retained data without mutation:
        ```python
        store = SQLiteObservationStore("events/observability.db", read_only=True)
        ```

    Args:
        path: SQLite database path.
        read_only: Whether all mutation methods must fail.
        policy: Prompt capture and bounded-metadata policy.

    Returns:
        SQLiteObservationStore: Concrete concurrent observation store.

    Notes:
        Each operation owns a short-lived connection; WAL permits concurrent
        active writes and historical reads without a backend registry.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        policy: ObservationPolicy | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.read_only = read_only
        self.policy = policy or ObservationPolicy()
        self.policy.validate()
        self.prompt_store = PromptStore(self.policy)
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        self._has_llm_call_attempts = self._table_exists("llm_call_attempts")

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            conn = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro",
                uri=True,
                timeout=5.0,
            )
        else:
            conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)

    def _table_exists(self, table_name: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                ).fetchone()
                is not None
            )

    async def close(self) -> None:
        return None

    async def append_observation(
        self,
        record: ObservationRecord,
        *,
        resource_links: Iterable[dict[str, Any]] = (),
    ) -> str:
        return await asyncio.to_thread(self._append_observation, record, tuple(resource_links))

    def _append_observation(
        self,
        record: ObservationRecord,
        resource_links: tuple[dict[str, Any], ...],
    ) -> str:
        self._ensure_writable()
        attributes_json = self._bounded_json(record.attributes)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_observation(conn, record, attributes_json=attributes_json)
            for link in resource_links:
                if link.get("relation") not in _RESOURCE_RELATIONS:
                    raise ValueError(
                        f"Unsupported observation resource relation: {link.get('relation')}"
                    )
                conn.execute(
                    """
                    INSERT INTO observation_resource_links(
                        observation_id, resource_key, relation, revision,
                        content_hash, slot_key, artifact_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.observation_id,
                        link["resource_key"],
                        link["relation"],
                        link.get("revision"),
                        link.get("content_hash"),
                        link.get("slot_key"),
                        link.get("artifact_id"),
                    ),
                )
            conn.commit()
        return record.observation_id

    async def append_llm_call(self, record: LLMObservationRecord) -> str:
        capture = self.prompt_store.prepare(record)
        record.prompt_manifest_id = capture.manifest_id
        await asyncio.to_thread(self._append_llm_call, record, capture)
        return record.llm_call_id

    def _append_llm_call(
        self,
        record: LLMObservationRecord,
        capture: PreparedPromptCapture,
    ) -> None:
        self._ensure_writable()
        observation = ObservationRecord(
            category="llm",
            name=record.call_type,
            summary=f"{record.provider}/{record.model} {record.call_type}",
            occurred_at=record.created_at,
            status="error" if record.error_type else "ok",
            severity="error" if record.error_type else "info",
            scope=record.scope,
            llm_call_id=record.llm_call_id,
            retention_class="forensic" if capture.capture_mode == "full" else "standard",
            expires_at=self._full_capture_expiry(record.created_at)
            if capture.capture_mode == "full"
            else None,
            attributes={
                "capture_mode": capture.capture_mode,
                "prompt_roles": list(capture.roles),
                "prompt_message_count": len(record.messages),
                "prompt_chars": capture.total_chars,
                "prompt_bytes": capture.total_bytes,
                "assembled_request_hash": capture.assembled_request_hash,
                "omission_reason": capture.omission_reason,
                "trace_payload": self._summarize_value(record.trace_payload),
            },
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for fragment in capture.fragments:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO content_fragments(
                        fragment_id, content_kind, canonical_hash, byte_count, body, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fragment.fragment_id,
                        fragment.content_kind,
                        fragment.canonical_hash,
                        fragment.byte_count,
                        fragment.body,
                        utc_now_iso(),
                    ),
                )
            self._insert_observation(
                conn,
                observation,
                attributes_json=self._bounded_json(observation.attributes),
            )
            conn.execute(
                """
                INSERT INTO llm_calls(
                    llm_call_id, observation_id, call_type, provider, model,
                    profile_name, call_name, reasoning_effort, max_output_tokens,
                    output_format, json_schema_json, schema_name, strict_schema,
                    validate_json, extra_params_json, request_args_json,
                    provider_request_args_json, compatibility_notes_json, usage_json,
                    latency_ms, error_type, error_message, capture_mode,
                    prompt_manifest_id, response_fragment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.llm_call_id,
                    observation.observation_id,
                    record.call_type,
                    record.provider,
                    record.model,
                    record.profile_name,
                    record.call_name,
                    record.reasoning_effort,
                    record.max_output_tokens,
                    record.output_format,
                    self._bounded_json(record.json_schema),
                    record.schema_name,
                    self._bool_int(record.strict_schema),
                    self._bool_int(record.validate_json),
                    self._bounded_json(record.extra_params),
                    self._bounded_json(record.request_args),
                    self._bounded_json(record.provider_request_args),
                    canonical_json([str(note)[:500] for note in record.compatibility_notes[:20]]),
                    self._bounded_json(record.usage),
                    record.latency_ms,
                    record.error_type,
                    self._clip(record.error_message, self.policy.max_error_chars),
                    capture.capture_mode,
                    capture.manifest_id,
                    capture.response_fragment.fragment_id if capture.response_fragment else None,
                ),
            )
            for attempt in record.attempts:
                conn.execute(
                    """
                    INSERT INTO llm_call_attempts(
                        llm_call_id, attempt_number, elapsed_ms, outcome,
                        retryable, status_code, error_code, request_id,
                        provider_delay_ms, scheduled_delay_ms, rate_limits_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.llm_call_id,
                        attempt.attempt_number,
                        round(attempt.elapsed_s * 1_000),
                        attempt.outcome,
                        int(attempt.retryable),
                        attempt.status_code,
                        attempt.error_code,
                        self._clip(attempt.request_id, 256),
                        self._optional_milliseconds(attempt.provider_delay_s),
                        self._optional_milliseconds(attempt.scheduled_delay_s),
                        canonical_json([asdict(snapshot) for snapshot in attempt.rate_limits]),
                    ),
                )
            if capture.manifest_id is not None:
                conn.execute(
                    """
                    INSERT INTO prompt_manifests(
                        manifest_id, llm_call_id, renderer_version, capture_mode,
                        assembled_request_hash, total_chars, total_bytes, roles_json,
                        omission_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capture.manifest_id,
                        record.llm_call_id,
                        self.policy.renderer_version,
                        capture.capture_mode,
                        capture.assembled_request_hash,
                        capture.total_chars,
                        capture.total_bytes,
                        canonical_json(capture.roles),
                        capture.omission_reason,
                    ),
                )
                for part in capture.parts:
                    conn.execute(
                        """
                        INSERT INTO prompt_manifest_parts(
                            manifest_id, ordinal, semantic_kind, role, fragment_id, source_event_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            capture.manifest_id,
                            part.ordinal,
                            part.semantic_kind,
                            part.role,
                            part.fragment_id,
                            part.source_event_id,
                        ),
                    )
            conn.commit()

    async def get_observation(self, observation_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_observation, observation_id)

    def _get_observation(self, observation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
            ).fetchone()
            return self._observation_row(row, conn=conn) if row else None

    async def list_observations(
        self,
        filters: ObservationFilter | None = None,
        *,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_observations, filters or ObservationFilter(), offset
        )

    def _list_observations(
        self,
        filters: ObservationFilter,
        offset: int,
    ) -> list[dict[str, Any]]:
        where, params = self._filter_sql(filters)
        sql = f"SELECT * FROM observations o {where} ORDER BY occurred_at DESC"
        if filters.limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend((filters.limit, offset))
        with self._connect() as conn:
            return [
                self._observation_row(row, conn=conn)
                for row in conn.execute(sql, params).fetchall()
            ]

    async def list_resource_observations(
        self, resource_key: str, *, relation: str | None = None
    ) -> list[dict[str, Any]]:
        """List AG observations linked to one canonical resource identity.

        Intro:
            Queries the dedicated resource-link index without parsing payload text.

        Examples:
            `rows = await store.list_resource_observations("artifact:a-1")`
            `outputs = await store.list_resource_observations("artifact:a-1", relation="output")`

        Args:
            resource_key: Exact namespaced resource identity.
            relation: Optional exact canonical relationship.

        Returns:
            list[dict[str, Any]]: Linked observations in reverse chronological order.

        Notes:
            The authoritative resource content remains outside observability.
        """
        return await asyncio.to_thread(self._list_resource_observations, resource_key, relation)

    def _list_resource_observations(
        self, resource_key: str, relation: str | None
    ) -> list[dict[str, Any]]:
        where = ["l.resource_key = ?"]
        params: list[Any] = [resource_key]
        if relation is not None:
            where.append("l.relation = ?")
            params.append(relation)
        sql = f"""
            SELECT DISTINCT o.* FROM observations o
            JOIN observation_resource_links l
              ON l.observation_id = o.observation_id
            WHERE {" AND ".join(where)}
            ORDER BY o.occurred_at DESC
        """
        with self._connect() as conn:
            return [
                self._observation_row(row, conn=conn)
                for row in conn.execute(sql, params).fetchall()
            ]

    async def query_llm_calls(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._query_llm_calls,
            {
                "run_id": run_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "graph_id": graph_id,
                "node_id": node_id,
                "user_id": user_id,
                "org_id": org_id,
            },
            since,
            until,
            limit,
            offset,
        )

    def _query_llm_calls(
        self,
        dimensions: dict[str, str | None],
        since: datetime | None,
        until: datetime | None,
        limit: int | None,
        offset: int,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in dimensions.items():
            if value is not None:
                clauses.append(f"o.{key} = ?")
                params.append(value)
        if since is not None:
            clauses.append("o.occurred_at >= ?")
            params.append(self._iso(since))
        if until is not None:
            clauses.append("o.occurred_at <= ?")
            params.append(self._iso(until))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT c.*, o.occurred_at, o.tenant_id, o.project_id, o.org_id, o.user_id,
                   o.app_id, o.session_id, o.run_id, o.trace_id, o.agent_id,
                   o.graph_id, o.node_id, o.turn_id, o.attributes_json,
                   {self._llm_attempt_aggregate_projection()}
            FROM llm_calls c JOIN observations o ON o.observation_id = c.observation_id
            {where} ORDER BY o.occurred_at DESC
        """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend((limit, offset))
        with self._connect() as conn:
            return [
                self._llm_row(row, include_content=False, conn=conn)
                for row in conn.execute(sql, params)
            ]

    async def get_llm_call(self, llm_call_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_llm_call, llm_call_id)

    def _get_llm_call(self, llm_call_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT c.*, o.occurred_at, o.tenant_id, o.project_id, o.org_id, o.user_id,
                       o.app_id, o.session_id, o.run_id, o.trace_id, o.agent_id,
                       o.graph_id, o.node_id, o.turn_id, o.attributes_json,
                       {self._llm_attempt_aggregate_projection()}
                FROM llm_calls c JOIN observations o ON o.observation_id = c.observation_id
                WHERE c.llm_call_id = ?
                """,
                (llm_call_id,),
            ).fetchone()
            return self._llm_row(row, include_content=True, conn=conn) if row else None

    async def hydrate_prompt_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._hydrate_prompt_manifest, manifest_id)

    def _hydrate_prompt_manifest(
        self,
        manifest_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            manifest = connection.execute(
                "SELECT * FROM prompt_manifests WHERE manifest_id = ?", (manifest_id,)
            ).fetchone()
            if manifest is None:
                return None
            result = dict(manifest)
            result["roles"] = json.loads(result.pop("roles_json"))
            rows = connection.execute(
                """
                SELECT p.*, f.content_kind, f.body, f.byte_count
                FROM prompt_manifest_parts p
                JOIN content_fragments f ON f.fragment_id = p.fragment_id
                WHERE p.manifest_id = ? ORDER BY p.ordinal
                """,
                (manifest_id,),
            ).fetchall()
            result["parts"] = [dict(row) for row in rows]
            if result["capture_mode"] == "full" and rows:
                result["provider_request"] = json.loads(rows[0]["body"])
            elif result["capture_mode"] == "manifest" and rows:
                messages: list[dict[str, Any]] = []
                provider_request_args: dict[str, Any] = {}
                for row in rows:
                    body = json.loads(row["body"])
                    if row["content_kind"] == "provider_request_config":
                        provider_request_args = body
                    else:
                        messages.append(body)
                result["provider_request"] = {
                    "messages": messages,
                    "provider_request_args": provider_request_args,
                }
            else:
                result["provider_request"] = None
            return result
        finally:
            if owns_connection:
                connection.close()

    async def delete_observation(self, observation_id: str) -> PurgeResult:
        return await self.purge_observations(
            ObservationFilter(),
            dry_run=False,
            observation_ids=(observation_id,),
        )

    async def delete_trace(self, trace_id: str, *, dry_run: bool = False) -> PurgeResult:
        result = await self.purge_observations(
            ObservationFilter(trace_id=trace_id), dry_run=dry_run
        )
        if not dry_run:
            await self.update_trace_management(
                f"trace:{trace_id}",
                hidden=True,
                deleted=True,
                scope={"trace_id": trace_id},
            )
        return result

    async def delete_run_observations(self, run_id: str, *, dry_run: bool = False) -> PurgeResult:
        result = await self.purge_observations(ObservationFilter(run_id=run_id), dry_run=dry_run)
        if not dry_run:
            await self.update_trace_management(
                f"run:{run_id}",
                hidden=True,
                deleted=True,
                scope={"run_id": run_id},
            )
        return result

    async def delete_session_observations(
        self, session_id: str, *, dry_run: bool = False
    ) -> PurgeResult:
        result = await self.purge_observations(
            ObservationFilter(session_id=session_id), dry_run=dry_run
        )
        if not dry_run:
            await self.update_trace_management(
                f"session:{session_id}",
                hidden=True,
                deleted=True,
                scope={"session_id": session_id},
            )
        return result

    async def purge_observations(
        self,
        filters: ObservationFilter,
        *,
        dry_run: bool = True,
        observation_ids: tuple[str, ...] | None = None,
    ) -> PurgeResult:
        return await asyncio.to_thread(self._purge_observations, filters, dry_run, observation_ids)

    def _purge_observations(
        self,
        filters: ObservationFilter,
        dry_run: bool,
        observation_ids: tuple[str, ...] | None,
    ) -> PurgeResult:
        if not dry_run:
            self._ensure_writable()
        with self._connect() as conn:
            where, params = self._filter_sql(filters)
            if observation_ids is not None:
                if not observation_ids:
                    return PurgeResult(True, 0, 0, 0, 0, 0, 0)
                placeholders = ",".join("?" for _ in observation_ids)
                where = f"WHERE o.observation_id IN ({placeholders})"
                params = list(observation_ids)
            selection_sql = (
                f"SELECT o.observation_id, o.trace_id FROM observations o {where} "
                "ORDER BY o.occurred_at ASC"
            )
            if filters.limit is not None and observation_ids is None:
                selection_sql += " LIMIT ?"
                params.append(filters.limit)
            selected = conn.execute(selection_sql, params).fetchall()
            if filters.target_reclaimed_bytes is not None and selected:
                selected = self._select_reclaim_target(
                    conn,
                    selected,
                    max(0, filters.target_reclaimed_bytes),
                )
            ids = [row["observation_id"] for row in selected]
            if not ids:
                return PurgeResult(dry_run, 0, 0, 0, 0, 0, 0)
            placeholders = ",".join("?" for _ in ids)
            manifests = conn.execute(
                f"""
                SELECT m.manifest_id FROM prompt_manifests m
                JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
                WHERE c.observation_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            fragment_rows = conn.execute(
                f"""
                SELECT DISTINCT f.* FROM content_fragments f
                WHERE f.fragment_id IN (
                    SELECT p.fragment_id FROM prompt_manifest_parts p
                    JOIN prompt_manifests m ON m.manifest_id = p.manifest_id
                    JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
                    WHERE c.observation_id IN ({placeholders})
                    UNION
                    SELECT response_fragment_id FROM llm_calls
                    WHERE observation_id IN ({placeholders}) AND response_fragment_id IS NOT NULL
                    UNION
                    SELECT payload_fragment_id FROM observations
                    WHERE observation_id IN ({placeholders}) AND payload_fragment_id IS NOT NULL
                )
                """,
                [*ids, *ids, *ids],
            ).fetchall()
            candidate_ids = [row["fragment_id"] for row in fragment_rows]
            row_bytes = self._observation_nonfragment_bytes(conn, ids)
            exclusive = 0
            exclusive_storage = 0
            shared = 0
            for row in fragment_rows:
                if self._fragment_has_external_reference(conn, row["fragment_id"], set(ids)):
                    shared += int(row["byte_count"])
                else:
                    exclusive += int(row["byte_count"])
                    exclusive_storage += self._row_logical_bytes(row)
            preview = PurgeResult(
                dry_run=dry_run,
                matching_traces=len({row["trace_id"] for row in selected if row["trace_id"]}),
                matching_observations=len(ids),
                matching_manifests=len(manifests),
                exclusive_fragment_bytes=exclusive,
                shared_fragment_bytes_retained=shared,
                estimated_reclaimed_bytes=exclusive_storage + row_bytes,
            )
            if dry_run:
                return preview
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(f"DELETE FROM observations WHERE observation_id IN ({placeholders})", ids)
            deleted_fragments = self._garbage_collect_fragments(conn, candidate_ids)
            conn.commit()
            return PurgeResult(
                **{
                    **asdict(preview),
                    "dry_run": False,
                    "deleted_observations": len(ids),
                    "deleted_manifests": len(manifests),
                    "deleted_fragments": deleted_fragments,
                }
            )

    async def garbage_collect_fragments(self) -> int:
        return await asyncio.to_thread(self._collect_all_fragments)

    def _collect_all_fragments(self) -> int:
        self._ensure_writable()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            deleted = self._garbage_collect_fragments(conn)
            conn.commit()
            return deleted

    async def compact_storage(self) -> StorageStats:
        """Return deleted SQLite capacity to the filesystem.

        Intro:
            Checkpoints the WAL and vacuums the observation database explicitly.

        Examples:
            `stats = await store.compact_storage()`
            `reclaimed = before.physical_bytes - stats.physical_bytes`

        Args:
            None.

        Returns:
            StorageStats: Storage accounting after compaction.

        Notes:
            This maintenance operation may require an exclusive SQLite lock and
            is never run automatically on the agent execution path.
        """
        await asyncio.to_thread(self._compact_storage)
        return await self.get_storage_stats()

    def _compact_storage(self) -> None:
        self._ensure_writable()
        with self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

    async def get_storage_stats(self) -> StorageStats:
        return await asyncio.to_thread(self._get_storage_stats)

    async def list_scope_storage(self, scope_column: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_scope_storage, scope_column)

    async def list_suppressed_scopes(self) -> dict[str, set[str]]:
        """List tombstoned observability scope identities.

        Intro:
            Reads session, run, and trace scopes hidden by explicit deletion.

        Examples:
            `scopes = await store.list_suppressed_scopes()`
            `hidden_sessions = scopes["session_id"]`

        Args:
            None.

        Returns:
            dict[str, set[str]]: Suppressed IDs grouped by scope column.

        Notes:
            Canonical runtime records remain untouched by these tombstones.
        """
        return await asyncio.to_thread(self._list_suppressed_scopes)

    def _list_suppressed_scopes(self) -> dict[str, set[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, run_id, trace_id FROM trace_management
                WHERE hidden = 1 OR deleted = 1
                """
            ).fetchall()
        return {
            column: {str(row[column]) for row in rows if row[column]}
            for column in ("session_id", "run_id", "trace_id")
        }

    def _list_scope_storage(self, scope_column: str) -> list[dict[str, Any]]:
        if scope_column not in {"trace_id", "run_id"}:
            raise ValueError(f"Unsupported storage scope: {scope_column}")
        with self._connect() as conn:
            scopes = conn.execute(
                f"""
                SELECT {scope_column} AS scope_id, MAX(occurred_at) AS latest_at
                FROM observations
                WHERE {scope_column} IS NOT NULL
                GROUP BY {scope_column}
                ORDER BY latest_at DESC
                """
            ).fetchall()
            result: list[dict[str, Any]] = []
            for scope in scopes:
                observation_ids = [
                    row[0]
                    for row in conn.execute(
                        f"SELECT observation_id FROM observations WHERE {scope_column} = ?",
                        (scope["scope_id"],),
                    )
                ]
                fragment_bytes = self._observation_fragment_bytes(conn, observation_ids)
                row_bytes = self._observation_nonfragment_bytes(conn, observation_ids)
                pinned = bool(
                    conn.execute(
                        f"SELECT 1 FROM trace_management WHERE {scope_column} = ? AND pinned = 1 LIMIT 1",
                        (scope["scope_id"],),
                    ).fetchone()
                )
                result.append(
                    {
                        "scope_id": scope["scope_id"],
                        "latest_at": scope["latest_at"],
                        "logical_bytes": row_bytes + fragment_bytes,
                        "pinned": pinned,
                    }
                )
            return result

    def _get_storage_stats(self) -> StorageStats:
        with self._connect() as conn:

            def count(table: str) -> int:
                return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

            fragment_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(byte_count), 0) FROM content_fragments"
            ).fetchone()
            page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            logical_bytes = sum(
                self._table_logical_bytes(conn, table)
                for table in (
                    "content_fragments",
                    "observations",
                    "observation_resource_links",
                    "llm_calls",
                    "prompt_manifests",
                    "prompt_manifest_parts",
                    "trace_management",
                )
            )
            database_bytes = page_count * page_size
            wal_path = Path(f"{self.path}-wal")
            wal_bytes = wal_path.stat().st_size if wal_path.is_file() else 0
            shm_path = Path(f"{self.path}-shm")
            shm_bytes = shm_path.stat().st_size if shm_path.is_file() else 0
            return StorageStats(
                observations=count("observations"),
                llm_calls=count("llm_calls"),
                manifests=count("prompt_manifests"),
                fragments=int(fragment_row[0]),
                fragment_bytes=int(fragment_row[1]),
                logical_bytes=logical_bytes,
                database_bytes=database_bytes,
                wal_bytes=wal_bytes,
                shm_bytes=shm_bytes,
                physical_bytes=database_bytes + wal_bytes + shm_bytes,
            )

    async def update_trace_management(
        self,
        scope_key: str,
        *,
        pinned: bool | None = None,
        label: str | None = None,
        tags: list[str] | None = None,
        retention_class: str | None = None,
        expires_at: str | None = None,
        hidden: bool | None = None,
        deleted: bool | None = None,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._update_trace_management,
            scope_key,
            pinned,
            label,
            tags,
            retention_class,
            expires_at,
            hidden,
            deleted,
            scope or {},
        )

    def _update_trace_management(
        self,
        scope_key: str,
        pinned: bool | None,
        label: str | None,
        tags: list[str] | None,
        retention_class: str | None,
        expires_at: str | None,
        hidden: bool | None,
        deleted: bool | None,
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        self._ensure_writable()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM trace_management WHERE scope_key = ?", (scope_key,)
            ).fetchone()
            values = (
                dict(existing)
                if existing
                else {
                    "scope_key": scope_key,
                    "pinned": 0,
                    "label": None,
                    "tags_json": "[]",
                    "retention_class": "standard",
                    "expires_at": None,
                    "hidden": 0,
                    "deleted": 0,
                }
            )
            for key in ("tenant_id", "project_id", "app_id", "session_id", "run_id", "trace_id"):
                values[key] = scope.get(key, values.get(key))
            if pinned is not None:
                values["pinned"] = int(pinned)
            if label is not None:
                values["label"] = label
            if tags is not None:
                values["tags_json"] = canonical_json(tags)
            if retention_class is not None:
                values["retention_class"] = retention_class
            if expires_at is not None:
                values["expires_at"] = expires_at
            if hidden is not None:
                values["hidden"] = int(hidden)
            if deleted is not None:
                values["deleted"] = int(deleted)
            values["updated_at"] = utc_now_iso()
            columns = tuple(values)
            placeholders = ",".join("?" for _ in columns)
            updates = ",".join(
                f"{column}=excluded.{column}" for column in columns if column != "scope_key"
            )
            conn.execute(
                f"INSERT INTO trace_management({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(scope_key) DO UPDATE SET {updates}",
                [values[column] for column in columns],
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM trace_management WHERE scope_key = ?", (scope_key,)
            ).fetchone()
            payload = dict(result)
            payload["tags"] = json.loads(payload.pop("tags_json"))
            payload["pinned"] = bool(payload["pinned"])
            payload["hidden"] = bool(payload["hidden"])
            payload["deleted"] = bool(payload["deleted"])
            return payload

    def _insert_observation(
        self,
        conn: sqlite3.Connection,
        record: ObservationRecord,
        *,
        attributes_json: str,
    ) -> None:
        scope = asdict(record.scope)
        conn.execute(
            """
            INSERT INTO observations(
                observation_id, category, name, occurred_at, status, severity,
                tenant_id, project_id, org_id, user_id, app_id, session_id,
                run_id, trace_id, agent_id, graph_id, node_id, turn_id,
                parent_observation_id, caused_by_observation_id, source_event_id,
                llm_call_id, summary, attributes_json, payload_fragment_id,
                retention_class, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.observation_id,
                record.category,
                self._clip(record.name, 500),
                record.occurred_at,
                record.status,
                record.severity,
                *(scope[column] for column in _SCOPE_COLUMNS),
                record.parent_observation_id,
                record.caused_by_observation_id,
                record.source_event_id,
                record.llm_call_id,
                self._clip(record.summary, self.policy.max_summary_chars),
                attributes_json,
                record.payload_fragment_id,
                record.retention_class,
                record.expires_at,
            ),
        )

    def _llm_row(
        self,
        row: sqlite3.Row,
        *,
        include_content: bool,
        conn: sqlite3.Connection,
    ) -> dict[str, Any]:
        result = dict(row)
        result["call_id"] = result["llm_call_id"]
        result["created_at"] = result.pop("occurred_at")
        for key in (
            "json_schema_json",
            "extra_params_json",
            "request_args_json",
            "provider_request_args_json",
            "compatibility_notes_json",
            "usage_json",
            "attributes_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        result["strict_schema"] = self._int_bool(result["strict_schema"])
        result["validate_json"] = self._int_bool(result["validate_json"])
        result["messages"] = None
        result["raw_text"] = None
        result["trace_payload"] = None
        result["messages_preview"] = self._prompt_preview(result)
        result["raw_text_preview"] = None
        result["trace_payload_preview"] = result["attributes"].get("trace_payload")
        result["attempts"] = (
            self._llm_attempt_rows(result["llm_call_id"], conn=conn) if include_content else []
        )
        manifest_id = result.get("prompt_manifest_id")
        if include_content and manifest_id:
            manifest = self._hydrate_prompt_manifest(manifest_id, conn=conn)
            result["prompt_manifest"] = manifest
            provider_request = manifest.get("provider_request") if manifest else None
            if provider_request:
                result["messages"] = provider_request.get("messages")
            response_id = result.get("response_fragment_id")
            if response_id:
                response = conn.execute(
                    "SELECT body FROM content_fragments WHERE fragment_id = ?", (response_id,)
                ).fetchone()
                if response:
                    result["raw_text"] = json.loads(response["body"]).get("text")
        return result

    def _llm_attempt_rows(
        self,
        llm_call_id: str,
        *,
        conn: sqlite3.Connection,
    ) -> list[dict[str, Any]]:
        if not self._has_llm_call_attempts:
            return []
        rows = conn.execute(
            """
            SELECT attempt_number, elapsed_ms, outcome, retryable, status_code,
                   error_code, request_id, provider_delay_ms,
                   scheduled_delay_ms, rate_limits_json
            FROM llm_call_attempts
            WHERE llm_call_id = ?
            ORDER BY attempt_number
            """,
            (llm_call_id,),
        ).fetchall()
        return [
            {
                **{key: value for key, value in dict(row).items() if key != "rate_limits_json"},
                "retryable": bool(row["retryable"]),
                "rate_limits": json.loads(row["rate_limits_json"]),
            }
            for row in rows
        ]

    def _llm_attempt_aggregate_projection(self) -> str:
        if not self._has_llm_call_attempts:
            return "0 AS attempt_count, 0 AS retry_count, " "0 AS total_retry_wait_ms"
        return """
            (SELECT COUNT(*) FROM llm_call_attempts a
             WHERE a.llm_call_id = c.llm_call_id) AS attempt_count,
            (SELECT COUNT(*) FROM llm_call_attempts a
             WHERE a.llm_call_id = c.llm_call_id
               AND a.scheduled_delay_ms IS NOT NULL) AS retry_count,
            COALESCE((SELECT SUM(a.scheduled_delay_ms) FROM llm_call_attempts a
             WHERE a.llm_call_id = c.llm_call_id), 0) AS total_retry_wait_ms
        """.strip()

    @staticmethod
    def _optional_milliseconds(value: float | None) -> int | None:
        return None if value is None else round(value * 1_000)

    @staticmethod
    def _prompt_preview(result: dict[str, Any]) -> dict[str, Any] | None:
        attributes = result["attributes"]
        if result["capture_mode"] == "off":
            return None
        return {
            "count": attributes.get("prompt_message_count", 0),
            "roles": attributes.get("prompt_roles", []),
            "length": attributes.get("prompt_chars", 0),
            "sha256": attributes.get("assembled_request_hash"),
            "capture_mode": result["capture_mode"],
            "omission_reason": attributes.get("omission_reason"),
        }

    @staticmethod
    def _observation_row(
        row: sqlite3.Row, *, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any]:
        result = dict(row)
        result["attributes"] = json.loads(result.pop("attributes_json"))
        result["resource_links"] = (
            [
                dict(link)
                for link in conn.execute(
                    """
                    SELECT resource_key, relation, revision, content_hash,
                           slot_key, artifact_id
                    FROM observation_resource_links
                    WHERE observation_id = ?
                    ORDER BY resource_key, relation
                    """,
                    (result["observation_id"],),
                ).fetchall()
            ]
            if conn is not None
            else []
        )
        return result

    def _bounded_json(self, value: Any) -> str:
        body = canonical_json(value)
        encoded = body.encode("utf-8")
        if len(encoded) <= self.policy.max_attributes_bytes:
            return body
        summary = self._summarize_value(value)
        summary["omission_reason"] = "attributes_exceed_limit"
        return canonical_json(summary)

    @staticmethod
    def _clip(value: str | None, limit: int) -> str | None:
        if value is None or len(value) <= limit:
            return value
        return value[:limit]

    @staticmethod
    def _summarize_value(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        body = canonical_json(value)
        import hashlib

        return {
            "chars": len(body),
            "bytes": len(body.encode("utf-8")),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

    def _filter_sql(self, filters: ObservationFilter) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column in (
            "tenant_id",
            "project_id",
            "org_id",
            "user_id",
            "app_id",
            "session_id",
            "run_id",
            "trace_id",
            "agent_id",
            "graph_id",
            "node_id",
            "category",
            "retention_class",
        ):
            value = getattr(filters, column)
            if value is not None:
                clauses.append(f"o.{column} = ?")
                params.append(value)
        if filters.capture_mode is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM llm_calls c WHERE c.observation_id = o.observation_id AND c.capture_mode = ?)"
            )
            params.append(filters.capture_mode)
        if filters.created_before is not None:
            clauses.append("o.occurred_at < ?")
            params.append(filters.created_before)
        if filters.expired_before is not None:
            clauses.append("o.expires_at IS NOT NULL AND o.expires_at < ?")
            params.append(filters.expired_before)
        if filters.exclude_severity is not None:
            clauses.append("o.severity != ?")
            params.append(filters.exclude_severity)
        if filters.pinned is not None:
            exists = "EXISTS" if filters.pinned else "NOT EXISTS"
            clauses.append(
                f"{exists} (SELECT 1 FROM trace_management t WHERE t.pinned = 1 "
                "AND ((t.trace_id IS NOT NULL AND t.trace_id = o.trace_id) "
                "OR (t.run_id IS NOT NULL AND t.run_id = o.run_id) "
                "OR (t.session_id IS NOT NULL AND t.session_id = o.session_id)))"
            )
        return (f"WHERE {' AND '.join(clauses)}" if clauses else "", params)

    @staticmethod
    def _fragment_has_external_reference(
        conn: sqlite3.Connection,
        fragment_id: str,
        deleting_observation_ids: set[str],
    ) -> bool:
        rows = conn.execute(
            """
            SELECT c.observation_id FROM prompt_manifest_parts p
            JOIN prompt_manifests m ON m.manifest_id = p.manifest_id
            JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
            WHERE p.fragment_id = ?
            UNION
            SELECT observation_id FROM llm_calls WHERE response_fragment_id = ?
            UNION
            SELECT observation_id FROM observations WHERE payload_fragment_id = ?
            """,
            (fragment_id, fragment_id, fragment_id),
        ).fetchall()
        return any(row["observation_id"] not in deleting_observation_ids for row in rows)

    def _select_reclaim_target(
        self,
        conn: sqlite3.Connection,
        selected: list[sqlite3.Row],
        target_bytes: int,
    ) -> list[sqlite3.Row]:
        if target_bytes == 0:
            return selected[:1]
        retained: list[sqlite3.Row] = []
        logical_bytes = 0
        for row in selected:
            retained.append(row)
            observation_id = row["observation_id"]
            logical_bytes += self._observation_nonfragment_bytes(conn, [observation_id])
            logical_bytes += self._observation_fragment_bytes(conn, [observation_id])
            if logical_bytes >= target_bytes:
                break
        return retained

    @classmethod
    def _observation_nonfragment_bytes(
        cls,
        conn: sqlite3.Connection,
        observation_ids: list[str],
    ) -> int:
        if not observation_ids:
            return 0
        placeholders = ",".join("?" for _ in observation_ids)
        rows = [
            *conn.execute(
                f"SELECT * FROM observations WHERE observation_id IN ({placeholders})",
                observation_ids,
            ).fetchall(),
            *conn.execute(
                f"""
                SELECT r.* FROM observation_resource_links r
                WHERE r.observation_id IN ({placeholders})
                """,
                observation_ids,
            ).fetchall(),
            *conn.execute(
                f"SELECT c.* FROM llm_calls c WHERE c.observation_id IN ({placeholders})",
                observation_ids,
            ).fetchall(),
            *conn.execute(
                f"""
                SELECT m.* FROM prompt_manifests m
                JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
                WHERE c.observation_id IN ({placeholders})
                """,
                observation_ids,
            ).fetchall(),
            *conn.execute(
                f"""
                SELECT p.* FROM prompt_manifest_parts p
                JOIN prompt_manifests m ON m.manifest_id = p.manifest_id
                JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
                WHERE c.observation_id IN ({placeholders})
                """,
                observation_ids,
            ).fetchall(),
        ]
        return sum(cls._row_logical_bytes(row) for row in rows)

    @classmethod
    def _table_logical_bytes(
        cls,
        conn: sqlite3.Connection,
        table: str,
    ) -> int:
        return sum(
            cls._row_logical_bytes(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()
        )

    @staticmethod
    def _row_logical_bytes(row: sqlite3.Row) -> int:
        return sum(len(str(value).encode("utf-8")) for value in row if value is not None)

    @staticmethod
    def _observation_fragment_bytes(
        conn: sqlite3.Connection,
        observation_ids: list[str],
    ) -> int:
        if not observation_ids:
            return 0
        placeholders = ",".join("?" for _ in observation_ids)
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(byte_count), 0) FROM content_fragments
            WHERE fragment_id IN (
                SELECT p.fragment_id FROM prompt_manifest_parts p
                JOIN prompt_manifests m ON m.manifest_id = p.manifest_id
                JOIN llm_calls c ON c.llm_call_id = m.llm_call_id
                WHERE c.observation_id IN ({placeholders})
                UNION
                SELECT response_fragment_id FROM llm_calls
                WHERE observation_id IN ({placeholders}) AND response_fragment_id IS NOT NULL
                UNION
                SELECT payload_fragment_id FROM observations
                WHERE observation_id IN ({placeholders}) AND payload_fragment_id IS NOT NULL
            )
            """,
            [*observation_ids, *observation_ids, *observation_ids],
        ).fetchone()
        return int(row[0] or 0)

    @staticmethod
    def _garbage_collect_fragments(
        conn: sqlite3.Connection,
        candidate_ids: list[str] | None = None,
    ) -> int:
        clause = ""
        params: list[Any] = []
        if candidate_ids is not None:
            if not candidate_ids:
                return 0
            clause = f"AND fragment_id IN ({','.join('?' for _ in candidate_ids)})"
            params.extend(candidate_ids)
        cursor = conn.execute(
            f"""
            DELETE FROM content_fragments
            WHERE NOT EXISTS (
                SELECT 1 FROM prompt_manifest_parts p
                WHERE p.fragment_id = content_fragments.fragment_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM llm_calls c
                WHERE c.response_fragment_id = content_fragments.fragment_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM observations o
                WHERE o.payload_fragment_id = content_fragments.fragment_id
            )
            {clause}
            """,
            params,
        )
        return max(0, cursor.rowcount)

    def _full_capture_expiry(self, created_at: str) -> str:
        from datetime import timedelta

        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (created + timedelta(days=self.policy.full_prompt_ttl_days)).isoformat()

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise ObservationStoreError("Observation store is read-only")

    @staticmethod
    def _bool_int(value: bool | None) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _int_bool(value: int | None) -> bool | None:
        return None if value is None else bool(value)

    @staticmethod
    def _iso(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
