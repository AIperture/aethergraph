from __future__ import annotations

from aethergraph import NodeContext, graphify, tool

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
        await context.channel().send_text("✅ Loaded checkpoint for expensive_step")
        return {"result": cached["result"]}

    await context.channel().send_text("⏳ Running expensive_step...")
    # TODO: replace with real expensive call
    result = x.upper()

    await _save_ckpt(ckpt_key=ckpt_key, payload={"result": result}, context=context)
    await context.channel().send_text("💾 Saved checkpoint for expensive_step")
    return {"result": result}


@graphify(name="my_workflow", inputs=["x"], outputs=["result"])
def my_workflow(x):
    out = expensive_step(x=x)
    return {"result": out.result}
