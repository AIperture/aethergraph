# Artifact Patterns

Source of truth:
- `src/aethergraph/plugins/agents/graph_builder/skills/references/api/artifacts.md`

Use artifacts whenever graph edges would otherwise carry large blobs or complex data.

## Default Policy

- Graph inputs and outputs exposed to the user should be strings.
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

```python
@tool(name="load_checkpoint", outputs=["artifact_id"])
async def load_checkpoint(ckpt_key: str, *, context: NodeContext) -> dict[str, str]:
    rows = await context.artifacts().search(
        kind="checkpoint",
        labels={"ckpt_key": ckpt_key},
        limit=1,
    )
    artifact_id = str(rows[0].artifact_id) if rows else ""
    return {"artifact_id": artifact_id}
```

Use an empty string sentinel only if the next node explicitly handles it.

## Iterative Long-Running Node Pattern

If one node does repeated expensive work, do not wait until the end of the node to save state. Save resumable progress inside the node, then resume from the latest checkpoint when the node is retried.

Pattern:
- Compute a stable `ckpt_key` from the workflow input or job id.
- At node start, search for the latest checkpoint artifact.
- Load prior progress if it exists.
- Process one batch or iteration chunk.
- Save updated progress back to artifacts before the next chunk.
- On retry, reload and continue instead of restarting the whole node.

```python
@tool(name="iterative_worker", outputs=["result_artifact_id"])
async def iterative_worker(job_id: str, *, context: NodeContext) -> dict[str, str]:
    ckpt_key = f"iterative-worker:{job_id}"
    rows = await context.artifacts().search(
        kind="checkpoint",
        labels={"ckpt_key": ckpt_key},
        limit=1,
    )

    if rows:
        state = await context.artifacts().load_json_by_id(rows[0].artifact_id)
    else:
        state = {"next_index": 0, "items": []}

    while state["next_index"] < 10:
        i = state["next_index"]
        state["items"].append(f"processed-{i}")
        state["next_index"] = i + 1
        await context.artifacts().save_json(
            state,
            kind="checkpoint",
            labels={"ckpt_key": ckpt_key},
            name=f"{job_id}-progress.json",
        )

    final_artifact = await context.artifacts().save_json(
        state,
        kind="result",
        name=f"{job_id}-final.json",
        tags=["workflow-output"],
    )
    return {"result_artifact_id": str(final_artifact.artifact_id)}
```

This pattern matters because node retries restart from the node boundary. Durable intra-node checkpoints let the node continue from saved progress instead of losing the whole long-running iteration.

## Long-Run Guidance

- Avoid progress-only side channels as the primary durability mechanism.
- Save important intermediate results as artifacts before moving to the next expensive step.
- Prefer one artifact per durable stage boundary.
- For iterative jobs inside one node, checkpoint within the node body, not only between graph nodes.
