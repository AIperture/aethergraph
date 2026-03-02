---
id: ag-graph-builder-channel-api
title: Channel API
---

# context.channel() (curated)

This pack lists channel methods safe to use by default.

## Allowed methods

- `await context.channel().send_text(text: str, *, meta: dict | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"] = "assistant", memory_tags: list[str] | None = None, memory_data: dict | None = None)`
- `await context.channel().send_phase(phase: str, status: Literal["pending", "active", "done", "failed", "skipped"], *, label: str | None = None, detail: str | None = None, code: str | None = None, channel: str | None = None, key_suffix: str | None = None)`
- `async with context.channel().stream(channel: str | None = None) as s: ...`
  - `await s.delta("...")`
  - `await s.end(full_text: str | None = None, memory_tags: list[str] | None = None, memory_log: bool = True)`

(Advanced; only if user requests interactive waits)
- `await context.channel().ask_text(prompt: str | None, *, timeout_s: int = 3600, silent: bool = False, channel: str | None = None) -> str`

## Recommended usage in generated workflows

- Use `send_text` for start/end of expensive steps or major phase transitions.
- Use `send_phase` when the UI needs a durable status block.
- Use `stream` only when token-by-token or incremental output is required.

## Example

```python
from aethergraph import NodeContext, tool

@tool(name="notify", outputs=["ok"])
async def notify(text: str, *, context: NodeContext) -> dict:
    await context.channel().send_text(text)
    return {"ok": True}
```
