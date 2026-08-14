from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Literal

LegacyFileKind = Literal[
    "engine_trace_sqlite",
    "engine_trace_wal",
    "engine_trace_shm",
    "llm_observation_jsonl",
]

_LEGACY_FILE_LAYOUT: tuple[tuple[str, LegacyFileKind], ...] = (
    ("trace/trace.sqlite3", "engine_trace_sqlite"),
    ("trace/trace.sqlite3-wal", "engine_trace_wal"),
    ("trace/trace.sqlite3-shm", "engine_trace_shm"),
    ("events/llm/llm_calls.jsonl", "llm_observation_jsonl"),
)
_EVENT_DATABASE_PATH = "events/events.db"
_ARCHIVE_MANIFEST = "legacy_observability_cleanup_manifest.json"


class LegacyObservabilityCleanupError(RuntimeError):
    """Base error for explicit legacy observability maintenance."""


class LegacyObservabilityContainmentError(LegacyObservabilityCleanupError):
    """Reject a cleanup or archive target outside its declared boundary."""


class LegacyObservabilityWorkspaceActiveError(LegacyObservabilityCleanupError):
    """Reject cleanup while the workspace server holds its runtime lock."""


@dataclass(frozen=True)
class LegacyFileCandidate:
    relative_path: str
    kind: LegacyFileKind
    physical_bytes: int


@dataclass(frozen=True)
class LegacyEventRowsCandidate:
    relative_path: str
    row_count: int
    logical_bytes: int


@dataclass(frozen=True)
class LegacyObservabilityReport:
    workspace: str
    generated_at: str
    files: tuple[LegacyFileCandidate, ...]
    event_rows: tuple[LegacyEventRowsCandidate, ...]

    @property
    def file_bytes(self) -> int:
        return sum(item.physical_bytes for item in self.files)

    @property
    def event_row_bytes(self) -> int:
        return sum(item.logical_bytes for item in self.event_rows)

    @property
    def candidate_bytes(self) -> int:
        return self.file_bytes + self.event_row_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for an administrative receipt.

        Intro:
            Produces the stable JSON-compatible report shape used by the CLI.

        Examples:
            `payload = report.to_dict()`

        Args:
            None.

        Returns:
            dict[str, Any]: Candidate details and aggregate byte totals.

        Notes:
            Serialization does not rescan the workspace.
        """
        return {
            "workspace": self.workspace,
            "generated_at": self.generated_at,
            "files": [asdict(item) for item in self.files],
            "event_rows": [asdict(item) for item in self.event_rows],
            "file_bytes": self.file_bytes,
            "event_row_bytes": self.event_row_bytes,
            "candidate_bytes": self.candidate_bytes,
        }


@dataclass(frozen=True)
class LegacyObservabilityCleanupResult:
    dry_run: bool
    report: LegacyObservabilityReport
    removed_files: tuple[str, ...] = ()
    deleted_event_rows: int = 0
    deleted_event_row_bytes: int = 0
    archived: bool = False
    archive_dir: str | None = None
    archive_manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the completed or dry-run cleanup receipt.

        Intro:
            Produces the stable JSON-compatible action result used by the CLI.

        Examples:
            `payload = result.to_dict()`

        Args:
            None.

        Returns:
            dict[str, Any]: Report, deletion, and archive outcome fields.

        Notes:
            An archived result names its persisted archive manifest.
        """
        return {
            "dry_run": self.dry_run,
            "report": self.report.to_dict(),
            "removed_files": list(self.removed_files),
            "deleted_event_rows": self.deleted_event_rows,
            "deleted_event_row_bytes": self.deleted_event_row_bytes,
            "archived": self.archived,
            "archive_dir": self.archive_dir,
            "archive_manifest": self.archive_manifest,
        }


@dataclass(frozen=True)
class _LegacyEventRow:
    row_id: int
    payload_json: str
    tags: tuple[str, ...]
    logical_bytes: int


def scan_legacy_observability(workspace: str | Path) -> LegacyObservabilityReport:
    """Report unsupported observability data at the historical fixed locations.

    Intro:
        Finds legacy engine trace files, the LLM JSONL sink, and generic tracer
        rows without opening them as supported observability history.

    Examples:
        Inspect a workspace without changing it:
        ```python
        report = scan_legacy_observability("./aethergraph_workspace")
        print(report.candidate_bytes)
        ```

    Args:
        workspace: Existing AetherGraph workspace root.

    Returns:
        LegacyObservabilityReport: Contained cleanup candidates and byte totals.

    Notes:
        Only the historical fixed paths are considered. Custom legacy paths
        must be handled manually so the cleanup never performs a broad search.
    """
    root = _resolve_workspace(workspace)
    report, _ = _build_report(root)
    return report


def cleanup_legacy_observability(
    workspace: str | Path,
    *,
    apply: bool = False,
    archive_dir: str | Path | None = None,
) -> LegacyObservabilityCleanupResult:
    """Explicitly remove contained legacy observability data from one workspace.

    Intro:
        Runs as a dry report by default. Apply mode holds the workspace lock,
        optionally archives every candidate, then removes only reported data.

    Examples:
        Preview cleanup:
        ```python
        result = cleanup_legacy_observability("./workspace")
        ```

        Archive and apply cleanup:
        ```python
        result = cleanup_legacy_observability(
            "./workspace",
            apply=True,
            archive_dir="../workspace-observability-archive",
        )
        ```

    Args:
        workspace: Existing AetherGraph workspace root.
        apply: Whether to execute deletion. Defaults to a read-only report.
        archive_dir: Optional empty directory outside the workspace for copies.

    Returns:
        LegacyObservabilityCleanupResult: Exact candidates and completed actions.

    Notes:
        This is cleanup only: it never imports, migrates, or serves legacy data.
        SQLite row capacity becomes reusable but the event database is not
        automatically vacuumed.
    """
    root = _resolve_workspace(workspace)
    if not apply:
        report, _ = _build_report(root)
        return LegacyObservabilityCleanupResult(dry_run=True, report=report)

    from aethergraph.server.server_state import workspace_lock

    try:
        with workspace_lock(root, timeout_s=0.25):
            return _apply_cleanup(root, archive_dir=archive_dir)
    except TimeoutError as exc:
        raise LegacyObservabilityWorkspaceActiveError(
            f"Workspace is active; stop its server before cleanup: {root}"
        ) from exc


def _apply_cleanup(
    root: Path,
    *,
    archive_dir: str | Path | None,
) -> LegacyObservabilityCleanupResult:
    report, event_rows = _build_report(root)
    archive_root = _prepare_archive(root, archive_dir) if archive_dir is not None else None

    resolved_files: list[tuple[LegacyFileCandidate, Path]] = []
    for candidate in report.files:
        source = _resolve_contained_file(root, candidate.relative_path)
        resolved_files.append((candidate, source))
        if archive_root is not None:
            destination = archive_root / Path(candidate.relative_path)
            _copy_archive_file(source, destination, archive_root)

    deleted_event_rows = 0
    deleted_event_row_bytes = 0
    event_db = root / Path(_EVENT_DATABASE_PATH)
    if event_rows:
        archive_rows_path = None
        if archive_root is not None:
            archive_rows_path = archive_root / "events" / "events.db.legacy-trace-rows.jsonl"
        deleted_event_rows, deleted_event_row_bytes = _delete_event_rows(
            event_db,
            archive_path=archive_rows_path,
        )

    removed_files: list[str] = []
    for candidate, source in resolved_files:
        source.unlink()
        removed_files.append(candidate.relative_path)

    result = LegacyObservabilityCleanupResult(
        dry_run=False,
        report=report,
        removed_files=tuple(removed_files),
        deleted_event_rows=deleted_event_rows,
        deleted_event_row_bytes=deleted_event_row_bytes,
        archived=archive_root is not None,
        archive_dir=str(archive_root) if archive_root is not None else None,
        archive_manifest=(
            str(archive_root / _ARCHIVE_MANIFEST) if archive_root is not None else None
        ),
    )
    if archive_root is not None:
        (archive_root / _ARCHIVE_MANIFEST).write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result


def _build_report(
    root: Path,
) -> tuple[LegacyObservabilityReport, tuple[_LegacyEventRow, ...]]:
    files: list[LegacyFileCandidate] = []
    for relative_path, kind in _LEGACY_FILE_LAYOUT:
        candidate = root / Path(relative_path)
        if not candidate.exists():
            continue
        resolved = _resolve_contained_file(root, relative_path)
        files.append(
            LegacyFileCandidate(
                relative_path=relative_path,
                kind=kind,
                physical_bytes=resolved.stat().st_size,
            )
        )

    event_candidate: tuple[LegacyEventRowsCandidate, ...] = ()
    event_rows: tuple[_LegacyEventRow, ...] = ()
    event_db = root / Path(_EVENT_DATABASE_PATH)
    if event_db.exists():
        resolved_db = _resolve_contained_file(root, _EVENT_DATABASE_PATH)
        event_rows = _read_legacy_event_rows(resolved_db, read_only=True)
        if event_rows:
            event_candidate = (
                LegacyEventRowsCandidate(
                    relative_path=_EVENT_DATABASE_PATH,
                    row_count=len(event_rows),
                    logical_bytes=sum(row.logical_bytes for row in event_rows),
                ),
            )

    return (
        LegacyObservabilityReport(
            workspace=str(root),
            generated_at=datetime.now(UTC).isoformat(),
            files=tuple(files),
            event_rows=event_candidate,
        ),
        event_rows,
    )


def _read_legacy_event_rows(path: Path, *, read_only: bool) -> tuple[_LegacyEventRow, ...]:
    uri = f"{path.as_uri()}?mode=ro" if read_only else str(path)
    try:
        conn = sqlite3.connect(uri, uri=read_only)
    except sqlite3.Error as exc:
        raise LegacyObservabilityCleanupError(f"Cannot open event database: {path}") from exc
    conn.row_factory = sqlite3.Row
    try:
        return _read_legacy_event_rows_from_connection(conn)
    except sqlite3.Error as exc:
        raise LegacyObservabilityCleanupError(f"Cannot inspect event database: {path}") from exc
    finally:
        conn.close()


def _read_legacy_event_rows_from_connection(
    conn: sqlite3.Connection,
) -> tuple[_LegacyEventRow, ...]:
    tables = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "events" not in tables:
        return ()
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(events)")}
    if not {"id", "kind", "payload"} <= columns:
        return ()
    tag_table = "event_tags" in tables
    rows: list[_LegacyEventRow] = []
    for event in conn.execute("SELECT * FROM events WHERE kind = 'trace' ORDER BY id"):
        payload_json = str(event["payload"] or "")
        if not _is_legacy_generic_trace(payload_json):
            continue
        tags: tuple[str, ...] = ()
        if tag_table:
            tags = tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT tag FROM event_tags WHERE event_row_id = ? ORDER BY tag",
                    (event["id"],),
                )
            )
        logical_bytes = sum(_sqlite_value_bytes(value) for value in tuple(event))
        logical_bytes += sum(8 + len(tag.encode("utf-8")) for tag in tags)
        rows.append(
            _LegacyEventRow(
                row_id=int(event["id"]),
                payload_json=payload_json,
                tags=tags,
                logical_bytes=logical_bytes,
            )
        )
    return tuple(rows)


def _delete_event_rows(
    path: Path,
    *,
    archive_path: Path | None,
) -> tuple[int, int]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = _read_legacy_event_rows_from_connection(conn)
        if archive_path is not None:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with archive_path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(
                            {
                                "event_row_id": row.row_id,
                                "payload_json": row.payload_json,
                                "tags": list(row.tags),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        ids = [(row.row_id,) for row in rows]
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if ids and "event_tags" in tables:
            conn.executemany("DELETE FROM event_tags WHERE event_row_id = ?", ids)
        if ids:
            conn.executemany("DELETE FROM events WHERE id = ?", ids)
        conn.commit()
        return len(rows), sum(row.logical_bytes for row in rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _is_legacy_generic_trace(payload_json: str) -> bool:
    try:
        event = json.loads(payload_json)
    except (TypeError, ValueError):
        return False
    payload = event.get("payload") if isinstance(event, dict) else None
    if not isinstance(payload, dict):
        return False
    return (
        event.get("kind") == "trace"
        and payload.get("schema_version") == 1
        and isinstance(payload.get("trace_id"), str)
        and isinstance(payload.get("span_id"), str)
        and payload.get("phase") in {"start", "end", "error", "wait", "resume"}
        and isinstance(payload.get("service"), str)
        and isinstance(payload.get("operation"), str)
    )


def _resolve_workspace(workspace: str | Path) -> Path:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def _resolve_contained_file(root: Path, relative_path: str) -> Path:
    candidate = root / Path(relative_path)
    current = root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise LegacyObservabilityContainmentError(
                f"Legacy cleanup does not follow symlinks: {current}"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise LegacyObservabilityContainmentError(
            f"Legacy cleanup target is outside the workspace: {candidate}"
        )
    return resolved


def _prepare_archive(root: Path, archive_dir: str | Path) -> Path:
    archive_root = Path(archive_dir).expanduser().resolve()
    if archive_root == root or archive_root.is_relative_to(root):
        raise LegacyObservabilityContainmentError(
            "Legacy observability archives must be outside the workspace"
        )
    if archive_root.exists():
        if not archive_root.is_dir():
            raise NotADirectoryError(archive_root)
        if any(archive_root.iterdir()):
            raise FileExistsError(f"Archive directory must be empty: {archive_root}")
    else:
        archive_root.mkdir(parents=True)
    return archive_root


def _copy_archive_file(source: Path, destination: Path, archive_root: Path) -> None:
    resolved_parent = destination.parent.resolve()
    if not resolved_parent.is_relative_to(archive_root):
        raise LegacyObservabilityContainmentError(
            f"Archive target is outside the archive directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise OSError(f"Archived file size does not match source: {source}")


def _sqlite_value_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, int | float):
        return 8
    return len(str(value).encode("utf-8"))


__all__ = [
    "LegacyEventRowsCandidate",
    "LegacyFileCandidate",
    "LegacyObservabilityCleanupError",
    "LegacyObservabilityCleanupResult",
    "LegacyObservabilityContainmentError",
    "LegacyObservabilityReport",
    "LegacyObservabilityWorkspaceActiveError",
    "cleanup_legacy_observability",
    "scan_legacy_observability",
]
