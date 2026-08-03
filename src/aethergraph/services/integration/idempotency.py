"""Durable ingress idempotency claims and receipts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Literal, Protocol

from aethergraph.contracts.integration import IngressEnvelope, IngressReceipt


class IngressIdempotencyError(RuntimeError):
    """Structured failure raised for conflicting or invalid idempotency state."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.idempotency_conflict",
            "integration.idempotency_not_claimed",
            "integration.idempotency_already_completed",
            "integration.receipt_deployment_mismatch",
            "integration.receipt_duplicate_invalid",
        ],
        message: str,
    ) -> None:
        """Create one stable idempotency-store failure.

        Examples:
            Reject reuse with different content:
            ```python
            IngressIdempotencyError(
                code="integration.idempotency_conflict",
                message="The key is already bound to another envelope.",
            )
            ```

            Reject a second completion:
            ```python
            IngressIdempotencyError(
                code="integration.idempotency_already_completed",
                message="The ingress claim is already complete.",
            )
            ```

        Args:
            code: Stable machine-readable idempotency failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Conflicts never reuse a receipt from a different canonical envelope.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class IngressClaim:
    """Atomic claim result for one deployment-scoped idempotency key."""

    acquired: bool
    pending: bool
    receipt: IngressReceipt | None


class IngressIdempotencyStore(Protocol):
    """Provider-neutral persistence contract for ingress claims and receipts."""

    async def claim(
        self,
        *,
        deployment_id: str,
        envelope: IngressEnvelope,
    ) -> IngressClaim:
        """Atomically claim one canonical ingress identity.

        Examples:
            Claim a new envelope:
            ```python
            claim = await store.claim(deployment_id="deployment-1", envelope=envelope)
            ```

            Detect a completed duplicate:
            ```python
            if claim.receipt is not None:
                return claim.receipt
            ```

        Args:
            deployment_id: Exact host deployment accepting ingress.
            envelope: Closed canonical ingress envelope.

        Returns:
            IngressClaim: Ownership or existing claim state.

        Notes:
            Implementations must reject key reuse with different envelope content.
        """
        ...

    async def complete(
        self,
        *,
        deployment_id: str,
        envelope: IngressEnvelope,
        receipt: IngressReceipt,
    ) -> None:
        """Assign the terminal receipt for one acquired claim.

        Examples:
            Store an accepted receipt:
            ```python
            await store.complete(
                deployment_id="deployment-1",
                envelope=envelope,
                receipt=receipt,
            )
            ```

            Store a rejected receipt:
            ```python
            await store.complete(
                deployment_id="deployment-1",
                envelope=envelope,
                receipt=rejected,
            )
            ```

        Args:
            deployment_id: Exact host deployment owning the claim.
            envelope: Same envelope used during claim acquisition.
            receipt: Single terminal ingress result.

        Returns:
            None.

        Notes:
            Implementations must not overwrite completed receipts.
        """
        ...


class SQLiteIngressIdempotencyStore:
    """Persist unique ingress claims and terminal receipts in SQLite."""

    def __init__(self, path: str | Path) -> None:
        """Create or open the integration operational database.

        Examples:
            Create a store:
            ```python
            store = SQLiteIngressIdempotencyStore("host/integration.db")
            ```

            Reopen completed receipts after restart:
            ```python
            restored = SQLiteIngressIdempotencyStore("host/integration.db")
            ```

        Args:
            path: SQLite database path owned by the local AG Host workspace.

        Returns:
            None.

        Notes:
            The primary key includes deployment and integration identity so
            provider keys cannot collide across authorities.
        """
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingress_receipts (
                    deployment_id TEXT NOT NULL,
                    integration_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    external_event_id TEXT NOT NULL,
                    envelope_digest TEXT NOT NULL,
                    receipt_json TEXT,
                    PRIMARY KEY(deployment_id, integration_id, idempotency_key)
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ingress_external_event
                ON ingress_receipts(deployment_id, integration_id, external_event_id)
                """
            )

    async def claim(
        self,
        *,
        deployment_id: str,
        envelope: IngressEnvelope,
    ) -> IngressClaim:
        """Atomically acquire or inspect one canonical ingress identity.

        Examples:
            Acquire a new ingress key:
            ```python
            claim = await store.claim(deployment_id="deployment-1", envelope=envelope)
            assert claim.acquired
            ```

            Return a prior terminal result for a duplicate:
            ```python
            duplicate = await store.claim(deployment_id="deployment-1", envelope=envelope)
            assert duplicate.receipt.duplicate
            ```

        Args:
            deployment_id: Exact host deployment accepting the ingress.
            envelope: Closed canonical ingress envelope.

        Returns:
            IngressClaim: New ownership, pending duplicate, or completed duplicate.

        Notes:
            A pending duplicate performs no side effect. The coordinator decides
            how to await or report the in-flight original operation.
        """
        return await asyncio.to_thread(self._claim, deployment_id, envelope)

    def _claim(self, deployment_id: str, envelope: IngressEnvelope) -> IngressClaim:
        digest = _envelope_digest(envelope)
        key = (deployment_id, envelope.integration_id, envelope.idempotency_key)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT idempotency_key, external_event_id, envelope_digest, receipt_json
                FROM ingress_receipts
                WHERE deployment_id = ? AND integration_id = ?
                  AND (idempotency_key = ? OR external_event_id = ?)
                """,
                (*key, envelope.external_event_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO ingress_receipts(
                        deployment_id, integration_id, idempotency_key,
                        external_event_id, envelope_digest, receipt_json
                    ) VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (*key, envelope.external_event_id, digest),
                )
                return IngressClaim(acquired=True, pending=False, receipt=None)

            if (
                row["idempotency_key"] != envelope.idempotency_key
                or row["external_event_id"] != envelope.external_event_id
                or row["envelope_digest"] != digest
            ):
                raise IngressIdempotencyError(
                    code="integration.idempotency_conflict",
                    message="Idempotency key is already bound to a different ingress envelope.",
                )
            if row["receipt_json"] is None:
                return IngressClaim(acquired=False, pending=True, receipt=None)
            receipt = IngressReceipt.model_validate_json(row["receipt_json"])
            return IngressClaim(
                acquired=False,
                pending=False,
                receipt=receipt.model_copy(update={"duplicate": True}),
            )

    async def complete(
        self,
        *,
        deployment_id: str,
        envelope: IngressEnvelope,
        receipt: IngressReceipt,
    ) -> None:
        """Persist the one terminal receipt for an acquired ingress claim.

        Examples:
            Complete an accepted root turn:
            ```python
            await store.complete(
                deployment_id="deployment-1",
                envelope=envelope,
                receipt=receipt,
            )
            ```

            Complete a stable rejection:
            ```python
            await store.complete(
                deployment_id="deployment-1",
                envelope=envelope,
                receipt=rejected_receipt,
            )
            ```

        Args:
            deployment_id: Exact host deployment owning the claim.
            envelope: Same canonical envelope used to acquire the claim.
            receipt: Terminal accepted or rejected ingress receipt.

        Returns:
            None.

        Notes:
            Completion is single-assignment. Retrying completion is an explicit
            error rather than an overwrite or compatibility path.
        """
        await asyncio.to_thread(self._complete, deployment_id, envelope, receipt)

    def _complete(
        self,
        deployment_id: str,
        envelope: IngressEnvelope,
        receipt: IngressReceipt,
    ) -> None:
        if receipt.deployment_id != deployment_id:
            raise IngressIdempotencyError(
                code="integration.receipt_deployment_mismatch",
                message="Ingress receipt deployment does not match the claimed deployment.",
            )
        if receipt.duplicate:
            raise IngressIdempotencyError(
                code="integration.receipt_duplicate_invalid",
                message="The persisted original ingress receipt cannot be marked duplicate.",
            )
        digest = _envelope_digest(envelope)
        key = (deployment_id, envelope.integration_id, envelope.idempotency_key)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT external_event_id, envelope_digest, receipt_json
                FROM ingress_receipts
                WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?
                """,
                key,
            ).fetchone()
            if row is None:
                raise IngressIdempotencyError(
                    code="integration.idempotency_not_claimed",
                    message="Ingress receipt cannot complete before its key is claimed.",
                )
            if (
                row["external_event_id"] != envelope.external_event_id
                or row["envelope_digest"] != digest
            ):
                raise IngressIdempotencyError(
                    code="integration.idempotency_conflict",
                    message="Claimed idempotency key belongs to a different ingress envelope.",
                )
            if row["receipt_json"] is not None:
                raise IngressIdempotencyError(
                    code="integration.idempotency_already_completed",
                    message="Ingress idempotency claim already has a terminal receipt.",
                )
            conn.execute(
                """
                UPDATE ingress_receipts
                SET receipt_json = ?
                WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?
                """,
                (receipt.model_dump_json(), *key),
            )


def _envelope_digest(envelope: IngressEnvelope) -> str:
    payload = envelope.model_dump(mode="json", exclude={"received_at"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
