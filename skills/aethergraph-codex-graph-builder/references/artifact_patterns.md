# Artifact Patterns

Source of truth:
- `src/aethergraph/plugins/agents/graph_builder/skills/references/api/artifacts.md`

Use artifacts whenever graph edges would otherwise carry large blobs or complex data.

## Default Policy

- Persist large text, JSON, files, media, generated code, and checkpoints as artifacts.
- Pass artifact ids or uris between nodes as strings.
- Prefer saving the final workflow result to artifacts and returning the resulting artifact id or uri as the graph output.
- Only write to output folders or custom file paths when the user explicitly asks for that delivery mode.

## Save / Load Pattern

```python
from aethergraph import NodeContext, tool


@tool(name="save_payload", outputs=["artifact_id"])
async def save_payload(payload_text: str, *, context: NodeContext) -> dict[str, str]:
    artifact = await context.artifacts().save_text(
        payload_text,
        kind="text",
        name="payload.txt",
        tags=["workflow-output"],
    )
    return {"artifact_id": str(artifact.artifact_id)}


@tool(name="load_payload", outputs=["payload_text"])
async def load_payload(artifact_id: str, *, context: NodeContext) -> dict[str, str]:
    payload_text = await context.artifacts().load_text_by_id(artifact_id)
    return {"payload_text": payload_text}
```

## Checkpoint Pattern

- Use `kind="checkpoint"` for resumable expensive steps.
- Search by a stable `ckpt_key`.
- Save the expensive result once, then load it on subsequent runs.
- Use memory state for the latest checkpoint pointer or lightweight progress metadata.

```python
from aethergraph import NodeContext, tool


def _state_key(tool_name: str) -> str:
    return f"checkpoint:{tool_name}"


@tool(name="load_checkpoint", outputs=["checkpoint_artifact_id"])
async def load_checkpoint(job_id: str, *, context: NodeContext) -> dict[str, str]:
    ckpt_key = f"expensive-step:{job_id}"
    mem = context.memory()

    latest = await mem.latest_state(_state_key("expensive-step"))
    if isinstance(latest, dict) and latest.get("ckpt_key") == ckpt_key:
        artifact_id = str(latest.get("artifact_id") or "")
        if artifact_id:
            return {"checkpoint_artifact_id": artifact_id}

    rows = await context.artifacts().search(
        kind="checkpoint",
        labels={"ckpt_key": ckpt_key},
        level="scope",
        limit=1,
    )
    artifact_id = str(rows[0].artifact_id) if rows else ""
    return {"checkpoint_artifact_id": artifact_id}
```

Use an empty string sentinel only if the next node explicitly handles it.

## Checkpoint + Memory State Pattern

Use this when the node is iterative and you want fast resume checks without pushing full state into memory events.

```python
from aethergraph import NodeContext, tool


def _state_key(tool_name: str) -> str:
    return f"checkpoint:{tool_name}"


@tool(name="iterative_worker", outputs=["result_artifact_id"])
async def iterative_worker(job_id: str, *, context: NodeContext) -> dict[str, str]:
    ckpt_key = f"iterative-worker:{job_id}"
    mem = context.memory()
    artifacts = context.artifacts()

    state = await mem.latest_state(_state_key("iterative-worker"))
    if not isinstance(state, dict) or state.get("ckpt_key") != ckpt_key:
        state = {"ckpt_key": ckpt_key, "next_index": 0, "items": [], "artifact_id": ""}

    if state["artifact_id"]:
        try:
            checkpoint_payload = await artifacts.load_json_by_id(state["artifact_id"])
            state["next_index"] = int(checkpoint_payload.get("next_index", state["next_index"]))
            state["items"] = list(checkpoint_payload.get("items", state["items"]))
        except Exception:
            rows = await artifacts.search(
                kind="checkpoint",
                labels={"ckpt_key": ckpt_key},
                level="scope",
                limit=1,
            )
            if rows:
                checkpoint_payload = await artifacts.load_json_by_id(rows[0].artifact_id)
                state["artifact_id"] = str(rows[0].artifact_id)
                state["next_index"] = int(checkpoint_payload.get("next_index", 0))
                state["items"] = list(checkpoint_payload.get("items", []))

    while state["next_index"] < 10:
        i = state["next_index"]
        state["items"].append(f"processed-{i}")
        state["next_index"] = i + 1

        checkpoint = await artifacts.save_json(
            {
                "job_id": job_id,
                "next_index": state["next_index"],
                "items": state["items"],
            },
            kind="checkpoint",
            labels={"ckpt_key": ckpt_key, "tool": "iterative_worker"},
            suggested_uri=f"./checkpoints/{job_id}.json",
            pin=True,
        )
        state["artifact_id"] = str(checkpoint.artifact_id)
        await mem.record_state(
            key=_state_key("iterative-worker"),
            value=state,
            tags=["checkpoint", "iterative-worker"],
            meta={"ckpt_key": ckpt_key},
            severity=1,
        )

    final_artifact = await artifacts.save_json(
        state,
        kind="result",
        name=f"{job_id}-final.json",
        tags=["workflow-output"],
    )
    return {"result_artifact_id": str(final_artifact.artifact_id)}
```

Notes:
- `record_state(...)` stores small structured resume state only.
- `latest_state(...)` is the fast-path resume pointer.
- The artifact remains the source of truth for the full checkpoint payload.
- Use `level="scope"` only when you intentionally want cross-run reuse.

## Iterative Long-Running Node Pattern

If one node does repeated expensive work, do not wait until the end of the node to save state. Save resumable progress inside the node, then resume from the latest checkpoint when the node is retried.

Pattern:
- Compute a stable `ckpt_key` from the workflow input or job id.
- At node start, check memory state first for the latest checkpoint pointer, then fall back to artifact search if needed.
- Load prior progress if it exists.
- Process one batch or iteration chunk.
- Save updated progress back to artifacts before the next chunk.
- Update memory state with the latest checkpoint artifact id and minimal progress counters.
- On retry, reload and continue instead of restarting the whole node.

This pattern matters because node retries restart from the node boundary. Durable intra-node checkpoints let the node continue from saved progress instead of losing the whole long-running iteration.

## Long-Run Guidance

- Avoid progress-only side channels as the primary durability mechanism.
- Save important intermediate results as artifacts before moving to the next expensive step.
- Use memory state to track the latest checkpoint artifact id, iteration counters, or planner state.
- For iterative jobs inside one node, checkpoint within the node body, not only between graph nodes.
