"""Resumable deletion of quiescent sessions inside one canonical local bundle."""

from __future__ import annotations

import hashlib
import json

from aethergraph.services.canonical_storage_scope import validate_storage_owner_scope

from ...contracts import (
    StorageConfigurationError,
    StorageConflictError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .observation_repository import _delete_observation_ids


class LocalSessionPurge:
    """Keep deletion progress beside canonical data, preserving shared resources.

    The caller must fence new execution before calling this maintenance boundary.
    Every role transaction is repeatable; the control receipt retains run and blob
    identities until filesystem reclamation has completed.
    """

    def __init__(self, bundle):
        if bundle.mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Session deletion requires writable storage")
        self.bundle = bundle
        self.control, self.events, self.search = bundle._databases
        self.control.install_component(
            name="session_purge",
            version=1,
            statements=(
                "CREATE TABLE local_session_purges (purge_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0)",
            ),
        )

    async def purge(
        self,
        owner: StorageScope,
        session_ids: tuple[str, ...],
        *,
        stopped_session_ids: tuple[str, ...] = (),
    ) -> None:
        """Delete exact sessions and their evidence; retry resumes the same receipt."""
        validate_storage_owner_scope(owner)
        if not session_ids or any(
            not isinstance(value, str) or not value.strip() for value in session_ids
        ):
            raise StorageConfigurationError("Session deletion requires explicit session identities")
        sessions = sorted(set(session_ids))
        if not set(stopped_session_ids).issubset(sessions):
            raise StorageConfigurationError("Stopped sessions must belong to the deletion scope")
        identity = json.dumps([owner.as_filter(), sessions], sort_keys=True, separators=(",", ":"))
        purge_id = hashlib.sha256(identity.encode()).hexdigest()

        def prepare(connection):
            existing = connection.execute(
                "SELECT manifest_json, completed FROM local_session_purges WHERE purge_id = ?",
                (purge_id,),
            ).fetchone()
            if existing:
                return json.loads(existing[0]), bool(existing[1])
            where, values = self._where(owner, sessions, [])
            active_sessions = {
                row[0]
                for row in connection.execute(
                    f"SELECT session_id FROM local_runs WHERE {where} AND status NOT IN ('succeeded', 'failed', 'canceled')",
                    values,
                )
            }
            if active_sessions - set(stopped_session_ids):
                raise StorageConflictError(
                    "Finish or cancel active runs before deleting session history"
                )
            runs = [
                row[0]
                for row in connection.execute(
                    f"SELECT run_id FROM local_runs WHERE {where}", values
                )
            ]
            manifest = {"sessions": sessions, "runs": runs, "blobs": []}
            where, values = self._where(owner, sessions, runs)
            artifact_where, artifact_values = self._where(owner, sessions, runs, alias="o")
            artifacts = connection.execute(
                f"SELECT DISTINCT a.artifact_id, a.owner_scope_identity, a.blob_locator, a.preview_locator FROM local_artifacts a JOIN local_artifact_occurrences o ON o.artifact_id = a.artifact_id WHERE {artifact_where}",
                artifact_values,
            ).fetchall()
            manifest["artifacts"] = [row[0] for row in artifacts]
            for _, scope, locator, preview in artifacts:
                for candidate in (locator, preview):
                    if candidate:
                        blob = connection.execute(
                            "SELECT content_hash FROM local_blobs WHERE scope_key = ? AND blob_locator = ?",
                            (scope, candidate),
                        ).fetchone()
                        if blob:
                            manifest["blobs"].append([scope, candidate, blob[0]])
            connection.execute(
                "INSERT INTO local_session_purges(purge_id, manifest_json) VALUES (?, ?)",
                (purge_id, json.dumps(manifest)),
            )
            return manifest, False

        manifest, completed = await self.control.transaction(prepare)
        if completed:
            return
        where, values = self._where(owner, manifest["sessions"], manifest["runs"])
        json_where, json_values = self._where(
            owner, manifest["sessions"], manifest["runs"], json_column="scope_identity"
        )

        def purge_events(connection):
            connection.execute(
                f"DELETE FROM local_event_search_intents WHERE (stream, event_id) IN (SELECT stream, event_id FROM local_events WHERE {where})",
                values,
            )
            for table in (
                "local_events",
                "local_inbound_events",
                "local_semantic_events",
                "local_runtime_output",
            ):
                connection.execute(f"DELETE FROM {table} WHERE {where}", values)

        await self.events.transaction(purge_events)
        await self.search.transaction(
            lambda connection: connection.execute(
                f"DELETE FROM local_search_documents WHERE {json_where}", json_values
            ).rowcount
        )

        def purge_control(connection):
            observation_ids = tuple(
                row[0]
                for row in connection.execute(
                    f"SELECT observation_id FROM local_observations WHERE {where}",
                    values,
                )
            )
            _delete_observation_ids(connection, observation_ids)
            connection.execute(
                f"DELETE FROM local_artifact_search_intents WHERE occurrence_id IN (SELECT occurrence_id FROM local_artifact_occurrences WHERE {where})",
                values,
            )
            for table in (
                "local_artifact_occurrences",
                "local_continuation_leases",
                "local_continuations",
                "local_trigger_claims",
                "local_triggers",
                "local_ingress_claims",
                "local_external_session_bindings",
                "local_observation_scope_management",
                "local_runs",
                "local_sessions",
            ):
                connection.execute(f"DELETE FROM {table} WHERE {where}", values)
            for table in (
                "local_document_metadata",
                "local_documents",
                "local_key_values",
                "local_state_current",
                "local_state_history",
                "local_state_outbox",
                "local_artifact_relations",
            ):
                connection.execute(f"DELETE FROM {table} WHERE {json_where}", json_values)
            for artifact_id in manifest["artifacts"]:
                # A sibling occurrence or lineage link makes the artifact shared.
                connection.execute(
                    "DELETE FROM local_artifacts WHERE artifact_id = ? AND NOT EXISTS (SELECT 1 FROM local_artifact_occurrences o WHERE o.artifact_id = local_artifacts.artifact_id) AND NOT EXISTS (SELECT 1 FROM local_artifact_relations r WHERE r.source_artifact_id = local_artifacts.artifact_id OR r.target_artifact_id = local_artifacts.artifact_id) AND NOT EXISTS (SELECT 1 FROM local_artifact_search_intents i WHERE i.artifact_id = local_artifacts.artifact_id)",
                    (artifact_id,),
                )

        await self.control.transaction(purge_control)
        for scope, locator, content_hash in manifest["blobs"]:
            referenced = await self.control.read_transaction(
                lambda connection, scope=scope, locator=locator: connection.execute(
                    "SELECT 1 FROM local_artifacts WHERE owner_scope_identity = ? AND (blob_locator = ? OR preview_locator = ?) LIMIT 1",
                    (scope, locator, locator),
                ).fetchone()
                is not None
            )
            if not referenced:
                await self.bundle.blobs.delete(StorageScope(**json.loads(scope)), locator)
                # delete() may have committed metadata before a process interruption.
                await self.bundle.blobs._drain_gc_tombstones(1, content_hash=content_hash)
        await self.control.transaction(
            lambda connection: connection.execute(
                "UPDATE local_session_purges SET completed = 1 WHERE purge_id = ?",
                (purge_id,),
            ).rowcount
        )

    @staticmethod
    def _where(owner, sessions, runs, *, json_column=None, alias=None):
        def column(name):
            return (
                f"json_extract({json_column}, '$.{name}')"
                if json_column
                else f"{alias}.{name}"
                if alias
                else name
            )

        clauses = [f"{column(name)} = ?" for name in owner.as_filter()]
        values = list(owner.as_filter().values())
        selected = [f"{column('session_id')} IN (SELECT value FROM json_each(?))"]
        values.append(json.dumps(sessions))
        if runs:
            selected.append(f"{column('run_id')} IN (SELECT value FROM json_each(?))")
            values.append(json.dumps(runs))
        return " AND ".join([*clauses, "(" + " OR ".join(selected) + ")"]), values
