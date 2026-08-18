"""Deterministic directory archive creation and bounded safe materialization."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile

from aethergraph.storage.contracts import StorageCapacityError, StorageIntegrityError


@dataclass(frozen=True, slots=True)
class DirectoryArchiveStats:
    """Logical content statistics for one canonical directory archive."""

    entry_count: int
    file_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    relative_path: str
    path: Path
    is_directory: bool
    size_bytes: int
    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _ArchiveEntry:
    member: tarfile.TarInfo
    parts: tuple[str, ...]


def write_directory_archive(
    source: Path,
    archive_path: Path,
    *,
    max_entries: int,
    max_total_bytes: int,
) -> DirectoryArchiveStats:
    """Write a deterministic uncompressed POSIX tar archive for one directory."""
    entries = _collect_source_entries(
        source,
        max_entries=max_entries,
        max_total_bytes=max_total_bytes,
    )
    file_count = 0
    total_bytes = 0
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            info = tarfile.TarInfo(entry.relative_path)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            if entry.is_directory:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                archive.addfile(info)
                continue

            info.type = tarfile.REGTYPE
            info.mode = 0o644
            info.size = entry.size_bytes
            with entry.path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if not _matches_source_entry(opened, entry):
                    raise StorageIntegrityError(
                        f"Directory source changed while packaging: {entry.relative_path!r}"
                    )
                archive.addfile(info, handle)
                if not _matches_source_entry(os.fstat(handle.fileno()), entry):
                    raise StorageIntegrityError(
                        f"Directory source changed while packaging: {entry.relative_path!r}"
                    )
            file_count += 1
            total_bytes += entry.size_bytes
    return DirectoryArchiveStats(
        entry_count=len(entries),
        file_count=file_count,
        total_bytes=total_bytes,
    )


def extract_directory_archive(
    archive_path: Path,
    destination: Path,
    *,
    max_entries: int,
    max_total_bytes: int,
) -> DirectoryArchiveStats:
    """Validate and materialize one canonical tar archive without link traversal."""
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {destination.parent}")

    created = False
    try:
        with tarfile.open(archive_path, mode="r:", errorlevel=2) as archive:
            entries, stats = _validated_archive_entries(
                archive,
                max_entries=max_entries,
                max_total_bytes=max_total_bytes,
            )
            destination.mkdir()
            created = True
            actual_total = 0
            for entry in entries:
                target = destination.joinpath(*entry.parts)
                if entry.member.isdir():
                    target.mkdir(parents=True, exist_ok=False)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(entry.member)
                if source is None:
                    raise StorageIntegrityError(
                        f"Archive file has no readable payload: {entry.member.name!r}"
                    )
                written = 0
                with source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        actual_total += len(chunk)
                        if written > entry.member.size or actual_total > max_total_bytes:
                            raise StorageCapacityError(
                                "Directory archive exceeds the explicit materialization bound"
                            )
                        output.write(chunk)
                if written != entry.member.size:
                    raise StorageIntegrityError(
                        f"Archive file size does not match its header: {entry.member.name!r}"
                    )
            if actual_total != stats.total_bytes:
                raise StorageIntegrityError("Directory archive materialized byte count changed")
            return stats
    except Exception as exc:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        if isinstance(
            exc,
            (StorageCapacityError, StorageIntegrityError, FileExistsError, FileNotFoundError),
        ):
            raise
        if isinstance(exc, (OSError, tarfile.TarError)):
            raise StorageIntegrityError("Directory archive is malformed or unreadable") from exc
        raise


def _collect_source_entries(
    source: Path,
    *,
    max_entries: int,
    max_total_bytes: int,
) -> tuple[_SourceEntry, ...]:
    root_stat = source.lstat()
    if _is_link_like(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("source must identify a non-linked directory")

    entries: list[_SourceEntry] = []
    total_bytes = 0
    for current, directory_names, file_names in os.walk(source, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            metadata = path.lstat()
            relative = path.relative_to(source).as_posix()
            if _is_link_like(metadata):
                raise StorageIntegrityError(f"Directory archives do not permit links: {relative!r}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise StorageIntegrityError(
                    f"Directory archive entry is not a directory: {relative!r}"
                )
            entries.append(
                _SourceEntry(
                    relative,
                    path,
                    True,
                    0,
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                )
            )
            if len(entries) > max_entries:
                raise StorageCapacityError("Directory source exceeds the explicit entry bound")
        for name in file_names:
            path = current_path / name
            metadata = path.lstat()
            relative = path.relative_to(source).as_posix()
            if _is_link_like(metadata):
                raise StorageIntegrityError(f"Directory archives do not permit links: {relative!r}")
            if not stat.S_ISREG(metadata.st_mode):
                raise StorageIntegrityError(
                    f"Directory archive entry is not a regular file: {relative!r}"
                )
            entries.append(
                _SourceEntry(
                    relative,
                    path,
                    False,
                    metadata.st_size,
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                )
            )
            total_bytes += metadata.st_size
            if len(entries) > max_entries:
                raise StorageCapacityError("Directory source exceeds the explicit entry bound")
            if total_bytes > max_total_bytes:
                raise StorageCapacityError("Directory source exceeds the explicit byte bound")
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))


def _validated_archive_entries(
    archive: tarfile.TarFile,
    *,
    max_entries: int,
    max_total_bytes: int,
) -> tuple[tuple[_ArchiveEntry, ...], DirectoryArchiveStats]:
    members: list[tarfile.TarInfo] = []
    for member in archive:
        members.append(member)
        if len(members) > max_entries:
            raise StorageCapacityError("Directory archive exceeds the explicit entry bound")

    entries: list[_ArchiveEntry] = []
    seen: dict[str, bool] = {}
    file_count = 0
    total_bytes = 0
    for member in sorted(members, key=lambda item: (item.name.casefold(), item.name)):
        if not member.isdir() and not member.isfile():
            raise StorageIntegrityError(
                f"Directory archives permit only directories and regular files: {member.name!r}"
            )
        parts = _safe_member_parts(member.name)
        canonical = "/".join(parts).casefold()
        if canonical in seen:
            raise StorageIntegrityError(f"Directory archive path is duplicated: {member.name!r}")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index]).casefold()
            if seen.get(parent) is False:
                raise StorageIntegrityError(
                    f"Directory archive file is used as a parent: {member.name!r}"
                )
        seen[canonical] = member.isdir()
        if member.isdir():
            if member.size != 0:
                raise StorageIntegrityError(
                    f"Directory archive directory has a payload: {member.name!r}"
                )
        else:
            if member.size < 0:
                raise StorageIntegrityError(
                    f"Directory archive file has a negative size: {member.name!r}"
                )
            file_count += 1
            total_bytes += member.size
            if total_bytes > max_total_bytes:
                raise StorageCapacityError(
                    "Directory archive exceeds the explicit materialization bound"
                )
        entries.append(_ArchiveEntry(member, parts))

    entries.sort(key=lambda entry: (len(entry.parts), not entry.member.isdir(), entry.parts))
    return (
        tuple(entries),
        DirectoryArchiveStats(
            entry_count=len(entries),
            file_count=file_count,
            total_bytes=total_bytes,
        ),
    )


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise StorageIntegrityError(f"Directory archive path is unsafe: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise StorageIntegrityError(f"Directory archive path is unsafe: {name!r}")
    if any(":" in part for part in parts):
        raise StorageIntegrityError(f"Directory archive path is not portable: {name!r}")
    return tuple(parts)


def _is_link_like(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _matches_source_entry(metadata: os.stat_result, entry: _SourceEntry) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size == entry.size_bytes
        and metadata.st_dev == entry.device
        and metadata.st_ino == entry.inode
        and metadata.st_mtime_ns == entry.modified_ns
    )
