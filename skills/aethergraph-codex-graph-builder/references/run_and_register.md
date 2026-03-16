# Run And Register

Source of truth:
- `src/aethergraph/runner/__init__.py`
- `src/aethergraph/core/runtime/graph_runner.py`
- `src/aethergraph/__main__.py`
- `src/aethergraph/server/start.py`

Use this after the workflow module is valid and registered.

## Preflight

- The user may not have the AetherGraph repository locally.
- They only need a Python environment where `aethergraph` is installed.
- If `python -c "import aethergraph"` fails, tell them execution and UI serving cannot work yet.

## Local Run

Prefer the CLI in `src/aethergraph/__main__.py`.

When the final graph file is known, the simplest local run path is:

```bash
python -m aethergraph run ./aethergraph_graphs/my-workflow/workflow.py \
  --workspace ./aethergraph_workspace \
  --project-root . \
  --inputs '{"request_text":"summarize this"}'
```

Notes:
- When `target` is a `.py` file, `run` automatically uses API mode, registers the file with the running server, and polls by default unless `--no-poll` is set.
- If the file defines multiple graphs, pass `--graph <graph_name>`.
- This path expects a running server for the workspace. If one is not running, start it with `python -m aethergraph serve ...` first.

## In-Process Run

Use this when the user wants a direct execution without a server:

```bash
python -m aethergraph run my_workflow \
  --workspace ./aethergraph_workspace \
  --project-root . \
  --load-path ./aethergraph_graphs/my-workflow/workflow.py \
  --inputs '{"request_text":"summarize this"}'
```

Notes:
- This path runs in-process and does not require `--via-api`.
- The `target` is the graph name, not the file path.
- Prefer this when you want a one-shot local validation run without standing up the UI server.

## Local AG UI Server

Use `python -m aethergraph serve` when the user wants to interact through AG UI.

Ask for approval before starting a long-running local server process.

## CLI Command For UI

When the final graph file is known, give the user the direct CLI form from `python -m aethergraph serve`.

For a graph file at `./aethergraph_graphs/my-workflow/workflow.py`:

```bash
python -m aethergraph serve \
  --workspace ./aethergraph_workspace \
  --project-root . \
  --load-path ./aethergraph_graphs/my-workflow/workflow.py \
  --port 8745 \
  --reuse
```

Notes:
- `--load-path` imports the graph file before server startup, so `as_app` registrations appear in UI immediately.
- `--reuse` prints the existing server URL and exits if that workspace is already running.
- `--port 8745` keeps the local UI/server URL stable. Use `0` only if the user wants an auto-picked port.
- The UI URL will be `http://127.0.0.1:8745/ui` unless the user changes host or port.

If the user wants live reload during development:

```bash
python -m aethergraph serve \
  --workspace ./aethergraph_workspace \
  --project-root . \
  --load-path ./aethergraph_graphs/my-workflow/workflow.py \
  --port 8745 \
  --reload
```

## Optional CLI Registration

If you want an explicit registration step before serving or running, use:

```bash
python -m aethergraph register \
  --workspace ./aethergraph_workspace \
  --source file \
  --path ./aethergraph_graphs/my-workflow/workflow.py
```

Serving with `--load-path` is still the simplest path for getting the app into UI.

## Auto-Run Guidance For This Skill

After code generation and registration succeed:
- Prefer giving or running `python -m aethergraph run <workflow.py> ...` for direct execution requests.
- Prefer giving or running `python -m aethergraph serve ... --load-path <workflow.py>` for UI requests.
- Use `python -m aethergraph register ...` only when the user explicitly wants a separate registration action or when the workflow must be registered before a non-serve path.

## Post-Generation Prompt

After registration succeeds, ask the user:

1. Run locally with `python -m aethergraph run`
2. Start AG UI with `python -m aethergraph serve`
3. Cancel

Only start the server after approval.
