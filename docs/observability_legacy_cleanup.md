# Legacy observability cleanup

Observability v2 does not read or migrate pre-v2 engine traces, LLM JSONL
observations, or generic service-trace rows. They are unsupported offline data
after the coordinated cutover.

Use the administrative command first without `--apply`:

```powershell
python -m aethergraph observability legacy --workspace C:\path\to\workspace
```

The JSON report identifies only the historical fixed locations:

- `trace/trace.sqlite3` and its `-wal` / `-shm` sidecars;
- `events/llm/llm_calls.jsonl`; and
- rows in `events/events.db` that match the exact v1 `EventLogTracer` payload
  shape.

The report does not recursively guess custom paths, treat every `kind=trace`
event as obsolete, or open this data through the product observability facade.
`file_bytes` are physical file bytes. `event_row_bytes` are logical live bytes
for the matched event and tag rows. `candidate_bytes` is their dry-run total.

## Archive, then clean

Stop the workspace server before applying cleanup. To preserve a recoverable
offline copy, provide a new or empty directory outside the workspace:

```powershell
python -m aethergraph observability legacy `
  --workspace C:\path\to\workspace `
  --apply `
  --archive-dir C:\archives\workspace-observability-pre-v2
```

The archive preserves the legacy files at their relative paths, exports the
matched mixed-database rows to JSONL, and writes
`legacy_observability_cleanup_manifest.json`. The command's JSON result also
records every removed path, the deleted row count and logical bytes, and whether
an archive was created. Save standard output separately if an unarchived audit
receipt is required.

Applying without `--archive-dir` permanently removes the candidates. Cleanup
holds the workspace runtime lock and refuses an active server. It never runs on
ordinary startup.

Deleting generic rows makes their SQLite capacity reusable; it does not
automatically `VACUUM` the canonical event database. Schedule any desired event
database compaction separately while the workspace is stopped. If an old build
configured a nonstandard trace or JSONL path, archive and remove that explicit
path manually—the command intentionally does not search broadly or follow
symlinks.
