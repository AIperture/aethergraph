---
id: ag-graph-builder-checkpoint-pattern
title: Checkpoint Pattern
---

# Artifact checkpointing pattern (expensive steps)

Use checkpointing when:
- a tool step is expensive (minutes+), iterative, or depends on external resources
- reruns are likely during development

## Default approach

- key: `ckpt_key = f"{TOOL_NAME}:{stable_key_fields}"`
- load: `rows = await context.artifacts().search(kind="checkpoint", labels={"ckpt_key": ckpt_key}, limit=1)`
- load content: `await context.artifacts().load_json_by_id(rows[0].artifact_id)`
- save: `await context.artifacts().save_json(payload, kind="checkpoint", labels={"ckpt_key": ckpt_key, ...})`
- scope note: `search()` defaults to run-level scope. For cross-run reuse, set `level="scope"` (or explicit labels) intentionally.

## Best practices

- Include `tool` and `tool_ver` in labels.
- Keep checkpoint payload minimal (store big binaries as files via save_file).
- Use `await context.channel().send_text(...)` on cache hit/miss boundaries (not per-iteration).
- Build `ckpt_key` from deterministic business inputs only (no timestamps/random UUIDs).
- Keep checkpointing logic in the same tool that runs the expensive step.
- Keep orchestration in `@graphify`; do not create checkpoint orchestration by calling tool A from inside tool B.
