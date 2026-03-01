---
id: ag-graph-builder-channel-api
title: Channel API

---

# context.channel() (curated)

This pack lists the channel methods you may use by default.

## Allowed methods

- `await context.channel().send_text(text: str, *, meta: dict | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"]="assistant", memory_tags: list[str] | None = None, memory_data: dict | None = None)`

- `async with context.channel().stream(channel: str | None = None) as s: ...`
  - `await s.delta("...")`
  - `await s.end(full_text: str | None = None, memory_tags: list[str] | None = None, memory_log: bool = True)`

(Advanced / optional; only if user requests interaction)
- `await context.channel().ask_text(prompt: str | None, *, timeout_s: int = 3600, silent: bool = False, channel: str | None = None) -> str`

## Recommended usage in generated workflows

- Use `send_text` for start/end of expensive steps.
- Use `stream` only when streaming LLM output or incremental progress is required.

## Example

```python
from aethergraph import NodeContext, tool

@tool(name="notify", outputs=["ok"])
async def notify(text: str, *, context: NodeContext) -> dict:
    await context.channel().send_text(f"📣 {text}")
    return {"ok": True}
```
