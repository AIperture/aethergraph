---
id: ag-graph-builder-context-node-api
title: ContextNode API
---

# NodeContext Core (curated)

Use only the APIs below unless the user explicitly enables advanced services.

## Allowed calls

- `context.channel()`
- `context.artifacts()`
- `context.logger()` (optional)

## Rules

- If a tool calls `channel()` or `artifacts()`, it MUST include `*, context: NodeContext`.
- Prefer async tools only when needed (I/O, artifact access, interactive channel calls).
