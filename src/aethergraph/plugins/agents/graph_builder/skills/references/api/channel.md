---
id: ag-graph-builder-channel-api
title: Channel API
---

# context.channel() (curated)

This pack lists channel methods safe to use by default.

## Allowed methods

- `await context.channel().send_text(text: str, *, meta: dict | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"] = "assistant", memory_tags: list[str] | None = None, memory_data: dict | None = None)`
- `await context.channel().send_rich(text: str | None = None, *, rich: dict | None = None, meta: dict | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"] = "assistant", memory_tags: list[str] | None = None, memory_data: dict | None = None)`
- `await context.channel().send_phase(phase: str, status: Literal["pending", "active", "done", "failed", "skipped"], *, label: str | None = None, detail: str | None = None, code: str | None = None, channel: str | None = None, key_suffix: str | None = None)`
- `await context.channel().send_file(url: str | None = None, *, file_bytes: bytes | None = None, filename: str = "file.bin", title: str | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"] = "assistant")`
- `await context.channel().send_buttons(text: str, buttons: list[Button], *, meta: dict | None = None, channel: str | None = None, memory_log: bool = True, memory_role: Literal["user","assistant","system","tool"] = "assistant")`
- `async with context.channel().stream(channel: str | None = None) as s: ...`
  - `await s.delta("...")`
  - `await s.end(full_text: str | None = None, memory_tags: list[str] | None = None, memory_log: bool = True)`

## Recommended usage in generated workflows

- Use `send_text` for short updates.
- Use `send_rich` for structured cards.
- Use `send_phase` for durable long-thinking status, and always close active phases.
- Use `send_buttons` for plan/register decisions.
- Use `send_file` to deliver generated code artifacts to UI.
- Use `stream` only when token-by-token output is explicitly required.
