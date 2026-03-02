---
id: ag-graph-builder-artifact-api
title: Artifact API
---

# context.artifacts() (curated)

This pack defines the minimal artifacts surface for checkpointing and caching.

## Save (checkpoint writes)

- `await context.artifacts().save_json(payload: dict, *, suggested_uri: str | None = None, name: str | None = None, kind: str = "json", tags: list[str] | None = None, labels: dict | None = None, metrics: dict | None = None, pin: bool = False) -> Artifact`
- `await context.artifacts().save_text(payload: str, *, suggested_uri: str | None = None, name: str | None = None, kind: str = "text", tags: list[str] | None = None, labels: dict | None = None, metrics: dict | None = None, pin: bool = False) -> Artifact`
- `await context.artifacts().save_file(path: str, *, kind: str, tags: list[str] | None = None, mime: str | None = None, labels: dict | None = None, metrics: dict | None = None, suggested_uri: str | None = None, name: str | None = None, pin: bool = False, cleanup: bool = True) -> Artifact`

## Search (checkpoint reads)

Use structured search for checkpoints:
- `rows = await context.artifacts().search(kind="checkpoint", labels={"ckpt_key": ckpt_key}, limit=1)`
- `search()` defaults to run-level scope (`level="run"`). Use `level="scope"` if you want cross-run checkpoint reuse.

(You may also filter by tags, metrics, and optionally include `query/mode` for semantic search.)

## Load (checkpoint reads)

- `data = await context.artifacts().load_json_by_id(artifact_id: str, *, encoding: str = "utf-8", errors: str = "strict") -> dict`
- `text = await context.artifacts().load_text_by_id(artifact_id: str, *, encoding: str = "utf-8", errors: str = "strict") -> str`

## Checkpointing example (recommended)

```python
from __future__ import annotations
from aethergraph import NodeContext, tool

TOOL_NAME = "expensive_step"
TOOL_VER = "0.1.0"

async def _try_load_ckpt(*, ckpt_key: str, context: NodeContext) -> dict | None:
    rows = await context.artifacts().search(
        kind="checkpoint",
        labels={"ckpt_key": ckpt_key},
        limit=1,
    )
    if not rows:
        return None
    return await context.artifacts().load_json_by_id(rows[0].artifact_id)

async def _save_ckpt(*, ckpt_key: str, payload: dict, context: NodeContext) -> None:
    await context.artifacts().save_json(
        payload,
        kind="checkpoint",
        labels={"ckpt_key": ckpt_key, "tool": TOOL_NAME, "tool_ver": TOOL_VER},
        name=f"{ckpt_key}.json",
    )

@tool(name=TOOL_NAME, outputs=["result"])
async def expensive_step(x: str, *, context: NodeContext) -> dict:
    ckpt_key = f"{TOOL_NAME}:{x}"
    cached = await _try_load_ckpt(ckpt_key=ckpt_key, context=context)
    if cached is not None:
        await context.channel().send_text("loaded checkpoint")
        return {"result": cached["result"]}

    await context.channel().send_text("running expensive step")
    result = x.upper()  # replace with real work
    await _save_ckpt(ckpt_key=ckpt_key, payload={"result": result}, context=context)
    await context.channel().send_text("saved checkpoint")
    return {"result": result}
```

## Constraints

- Always label checkpoints with `ckpt_key`.
- Prefer `kind="checkpoint"` for checkpoint artifacts.
- Keep payload JSON-serializable.
- Keep `ckpt_key` deterministic and based on stable inputs.
