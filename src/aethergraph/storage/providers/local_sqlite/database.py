"""Central SQLite connection, transaction, and schema-version policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import sqlite3
from typing import Any, TypeVar

from ...contracts import (
    StorageFormatError,
    StorageHealth,
    StorageHealthError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageTimeoutError,
)

LOCAL_DATABASE_SCHEMA_VERSION = 2
_T = TypeVar("_T")
_ROLE_FILENAMES = {
    "control": "control.sqlite3",
    "events": "events.sqlite3",
    "search": "search.sqlite3",
}


class LocalDatabaseRole(StrEnum):
    """Logical local database roles whose filenames remain provider-private."""

    CONTROL = "control"
    EVENTS = "events"
    SEARCH = "search"


@dataclass(frozen=True, slots=True)
class LocalCheckpoint:
    """Bounded WAL checkpoint result without exposing a SQLite row object."""

    busy_pages: int
    log_pages: int
    checkpointed_pages: int


@dataclass(slots=True)
class LocalSQLiteDatabase:
    """One provider-owned SQLite role with serialized asynchronous access."""

    role: LocalDatabaseRole
    path: Path
    mode: StorageOpenMode
    _connection: sqlite3.Connection = field(repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def open(
        cls,
        *,
        workspace_root: Path,
        role: LocalDatabaseRole,
        mode: StorageOpenMode,
        busy_timeout_ms: int = 5_000,
        durability: str = "normal",
    ) -> LocalSQLiteDatabase:
        """Open one exact local database role under the provider-private layout.

        Existing files are inspected read-only for exact role/schema metadata before
        a writable connection can enable WAL. Only a missing file in read-write mode
        is initialized with the current empty-workspace schema marker.

        Examples:
            Open the control database for writes:
                ```python
                database = LocalSQLiteDatabase.open(
                    workspace_root=root,
                    role=LocalDatabaseRole.CONTROL,
                    mode=StorageOpenMode.READ_WRITE,
                )
                ```

            Open an initialized events database read-only:
                ```python
                database = LocalSQLiteDatabase.open(
                    workspace_root=root,
                    role=LocalDatabaseRole.EVENTS,
                    mode=StorageOpenMode.READ_ONLY,
                )
                ```

        Args:
            workspace_root: Authorized manifested workspace root.
            role: Exact logical local database role.
            mode: Explicit read-write or read-only access mode.
            busy_timeout_ms: Bounded SQLite lock wait in milliseconds.
            durability: Exact `normal` or `full` SQLite synchronous policy.

        Returns:
            LocalSQLiteDatabase: Open provider-owned database handle.

        Notes:
            Missing read-only files and pre-existing unversioned files fail directly;
            no legacy schema inference or migration is attempted.
        """
        if not isinstance(role, LocalDatabaseRole):
            raise TypeError("role must be a LocalDatabaseRole")
        if isinstance(busy_timeout_ms, bool) or not 1 <= busy_timeout_ms <= 120_000:
            raise ValueError("busy_timeout_ms must be between 1 and 120000")
        if durability not in {"normal", "full"}:
            raise ValueError("durability must be 'normal' or 'full'")
        root = workspace_root.resolve()
        path = root / "local" / _ROLE_FILENAMES[role.value]
        existed = path.exists()
        if existed:
            if not path.is_file() or path.is_symlink():
                raise StorageFormatError(f"Local {role.value} database must be a regular file")
            _validate_existing_database(path, role)
        elif mode is StorageOpenMode.READ_ONLY:
            raise StorageFormatError(f"Read-only local {role.value} database does not exist")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        try:
            connection = _connect(path, mode, busy_timeout_ms)
            if mode is StorageOpenMode.READ_WRITE:
                connection.execute(f"PRAGMA synchronous={durability.upper()}")
                connection.execute("PRAGMA journal_mode=WAL").fetchone()
                if not existed:
                    _initialize_database(connection, role)
            else:
                connection.execute("PRAGMA query_only=ON")
        except sqlite3.Error as exc:
            raise _classify_sqlite_error(exc, role) from exc
        return cls(role=role, path=path, mode=mode, _connection=connection)

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        """Execute one provider-private write statement asynchronously.

        The statement runs on the centralized worker-thread boundary while the
        database lock prevents interleaving with transactions and maintenance.

        Examples:
            Insert one provider row:
                ```python
                count = await database.execute(
                    "INSERT INTO records(id) VALUES (?)",
                    (record_id,),
                )
                ```

            Update matching rows:
                ```python
                count = await database.execute(
                    "UPDATE records SET status = ? WHERE status = ?",
                    ("done", "running"),
                )
                ```

        Args:
            sql: Provider-owned SQL write statement.
            parameters: Positional SQLite bind values.

        Returns:
            int: SQLite affected-row count.

        Notes:
            Read-only handles raise `StorageReadOnlyError` before executing SQL.
        """
        if self.mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError(f"Local {self.role.value} database is read-only")

        def operation(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(sql, tuple(parameters))
            return cursor.rowcount

        return await self._run(operation)

    def install_component(
        self,
        *,
        name: str,
        version: int,
        statements: Sequence[str],
    ) -> None:
        """Install or validate one exact provider-owned schema component.

        Installation runs synchronously during provider construction before the
        database is published. Existing exact versions are idempotent; mismatches
        fail without applying statements or upgrading legacy data.

        Examples:
            Install a blob metadata table:
                ```python
                database.install_component(
                    name="blobs",
                    version=1,
                    statements=(CREATE_BLOBS_TABLE,),
                )
                ```

            Validate the same component during read-only open:
                ```python
                readonly.install_component(name="blobs", version=1, statements=())
                ```

        Args:
            name: Exact provider-private component identifier.
            version: Positive exact component schema version.
            statements: Ordered SQL statements used only for first installation.

        Returns:
            None: The exact component version exists after successful return.

        Notes:
            Read-only open validates an existing component and never executes schema
            statements. Component upgrades require a future explicit format change.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("component name must be non-empty")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("component version must be a positive integer")
        if self._closed:
            raise StorageHealthError(f"Local {self.role.value} database is closed")
        try:
            row = self._connection.execute(
                "SELECT version FROM ag_storage_components WHERE name = ?",
                (name,),
            ).fetchone()
            if row is not None:
                if row[0] != version:
                    raise StorageFormatError(
                        f"Local schema component {name!r} has unsupported version {row[0]!r}"
                    )
                return
            if self.mode is StorageOpenMode.READ_ONLY:
                raise StorageFormatError(
                    f"Read-only local database is missing schema component {name!r}"
                )
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO ag_storage_components(name, version) VALUES (?, ?)",
                    (name, version),
                )
            except BaseException:
                self._connection.rollback()
                raise
            self._connection.commit()
        except sqlite3.Error as exc:
            raise _classify_sqlite_error(exc, self.role) from exc

    async def fetch_all(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> tuple[sqlite3.Row, ...]:
        """Fetch a bounded provider-private query result asynchronously.

        Rows are materialized inside the serialized worker operation so no cursor
        escapes the database lifecycle boundary.

        Examples:
            Read one exact identity:
                ```python
                rows = await database.fetch_all(
                    "SELECT * FROM records WHERE id = ?",
                    (record_id,),
                )
                ```

            Read a bounded page:
                ```python
                rows = await database.fetch_all(
                    "SELECT * FROM records ORDER BY id LIMIT ?",
                    (50,),
                )
                ```

        Args:
            sql: Provider-owned bounded query.
            parameters: Positional SQLite bind values.

        Returns:
            tuple[sqlite3.Row, ...]: Fully materialized rows in query order.

        Notes:
            Public storage records are constructed by owning repositories; SQLite rows
            never cross the provider boundary.
        """

        def operation(connection: sqlite3.Connection) -> tuple[sqlite3.Row, ...]:
            return tuple(connection.execute(sql, tuple(parameters)).fetchall())

        return await self._run(operation)

    async def transaction(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run one synchronous provider operation in an immediate transaction.

        The callback executes on the database worker while holding both the async
        handle lock and SQLite write reservation. Success commits; any exception
        rolls back before typed error classification.

        Examples:
            Atomically update related rows:
                ```python
                result = await database.transaction(update_related_rows)
                ```

            Return a created provider record:
                ```python
                record = await database.transaction(insert_and_read_record)
                ```

        Args:
            operation: Synchronous provider-private callback receiving the connection.

        Returns:
            _T: Exact callback result after a successful commit.

        Notes:
            Read-only handles reject transactions. Callbacks must not retain the
            connection or perform asynchronous work.
        """
        if self.mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError(f"Local {self.role.value} database is read-only")

        def transactional(connection: sqlite3.Connection) -> _T:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = operation(connection)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
            return result

        return await self._run(transactional)

    async def read_transaction(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run multiple provider reads against one consistent SQLite snapshot.

        The synchronous callback executes on the serialized database worker inside a
        deferred read transaction. The snapshot is always rolled back after the
        callback so this primitive works through read-write and read-only handles.

        Examples:
            Read related rows consistently:
                ```python
                rows = await database.read_transaction(load_related_rows)
                ```

            Compute a dry-run preview:
                ```python
                preview = await database.read_transaction(compute_preview)
                ```

        Args:
            operation: Synchronous provider-private callback receiving the connection.

        Returns:
            _T: Exact callback result from one consistent read snapshot.

        Notes:
            The callback must not write, retain the connection, or perform async work.
        """

        def transactional(connection: sqlite3.Connection) -> _T:
            connection.execute("BEGIN")
            try:
                return operation(connection)
            finally:
                connection.rollback()

        return await self._run(transactional)

    async def health(self) -> StorageHealth:
        """Run SQLite's bounded quick integrity check for this role.

        The check shares the serialized execution boundary and reports readiness
        without opening another database or exposing physical schema details.

        Examples:
            Check an active database:
                ```python
                status = await database.health()
                ```

            Gate provider readiness:
                ```python
                if not (await database.health()).ready:
                    raise StorageHealthError("storage unavailable")
                ```

        Args:
            None.

        Returns:
            StorageHealth: Ready when `PRAGMA quick_check(1)` returns `ok`.

        Notes:
            A closed database raises `StorageHealthError`.
        """
        rows = await self.fetch_all("PRAGMA quick_check(1)")
        detail = str(rows[0][0]) if rows else "no result"
        return StorageHealth(ready=detail == "ok", detail=detail)

    async def checkpoint(self) -> LocalCheckpoint:
        """Checkpoint this role's WAL without changing provider selection.

        A passive checkpoint copies available frames while allowing concurrent readers
        and returns bounded page counts for provider maintenance reporting.

        Examples:
            Checkpoint after a backup barrier:
                ```python
                result = await database.checkpoint()
                ```

            Inspect remaining WAL pages:
                ```python
                assert result.log_pages >= result.checkpointed_pages
                ```

        Args:
            None.

        Returns:
            LocalCheckpoint: Busy, log, and checkpointed page counts.

        Notes:
            Read-only handles raise `StorageReadOnlyError`.
        """
        if self.mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError(f"Local {self.role.value} database is read-only")
        rows = await self.fetch_all("PRAGMA wal_checkpoint(PASSIVE)")
        if len(rows) != 1 or len(rows[0]) != 3:
            raise StorageHealthError("Local SQLite checkpoint returned an invalid result")
        return LocalCheckpoint(*(int(value) for value in rows[0]))

    async def close(self) -> None:
        """Close this database connection exactly once.

        Close is serialized with every query, transaction, health check, and
        checkpoint so no operation can continue on a released connection.

        Examples:
            Close during provider shutdown:
                ```python
                await database.close()
                ```

            Close safely after partial startup:
                ```python
                await database.close()
                await database.close()
                ```

        Args:
            None.

        Returns:
            None: The connection is closed or was already closed.

        Notes:
            The provider bundle, not individual services, owns this operation.
        """
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await asyncio.to_thread(self._connection.close)

    def _close_during_open_failure(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()

    async def _run(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        async with self._lock:
            if self._closed:
                raise StorageHealthError(f"Local {self.role.value} database is closed")
            try:
                return await asyncio.to_thread(operation, self._connection)
            except sqlite3.Error as exc:
                raise _classify_sqlite_error(exc, self.role) from exc


def _connect(path: Path, mode: StorageOpenMode, busy_timeout_ms: int) -> sqlite3.Connection:
    target = str(path)
    uri = False
    if mode is StorageOpenMode.READ_ONLY:
        target = f"{path.as_uri()}?mode=ro"
        uri = True
    connection = sqlite3.connect(
        target,
        uri=uri,
        timeout=busy_timeout_ms / 1000,
        check_same_thread=False,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return connection


def _initialize_database(
    connection: sqlite3.Connection,
    role: LocalDatabaseRole,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            CREATE TABLE ag_storage_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                role TEXT NOT NULL,
                schema_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE ag_storage_components (
                name TEXT PRIMARY KEY,
                version INTEGER NOT NULL CHECK (version > 0)
            )
            """
        )
        connection.execute(
            "INSERT INTO ag_storage_meta(singleton, role, schema_version) VALUES (1, ?, ?)",
            (role.value, LOCAL_DATABASE_SCHEMA_VERSION),
        )
    except BaseException:
        connection.rollback()
        raise
    connection.commit()


def _validate_existing_database(path: Path, role: LocalDatabaseRole) -> None:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=1,
        )
        row = connection.execute(
            "SELECT role, schema_version FROM ag_storage_meta WHERE singleton = 1"
        ).fetchone()
    except sqlite3.Error as exc:
        raise StorageFormatError(
            f"Existing local {role.value} database is not a supported current schema"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if row != (role.value, LOCAL_DATABASE_SCHEMA_VERSION):
        raise StorageFormatError(
            f"Existing local {role.value} database role or schema version does not match"
        )


def _classify_sqlite_error(
    error: sqlite3.Error,
    role: LocalDatabaseRole,
) -> Exception:
    message = str(error).lower()
    prefix = f"Local {role.value} database"
    if "locked" in message or "busy" in message:
        return StorageTimeoutError(f"{prefix} lock wait timed out")
    if "readonly" in message or "read-only" in message:
        return StorageReadOnlyError(f"{prefix} is read-only")
    if "malformed" in message or "not a database" in message:
        return StorageIntegrityError(f"{prefix} failed integrity validation")
    return StorageIntegrityError(f"{prefix} operation failed")
